"""
pointcloud_action_datasets.py

Folder layout (typical):
Each observation only have one point cloud


# so you should shuffle over episodes, not inside the episodes
  obs_root/
    episode_001/
      point.ply
    episode_002/
      point.ply

  tactile_root/
    episode_001/
    tactile.logs
    episode_002/
    tactile.logs
  act_root/
    episode_001/
      nova2.txt
      nova5.txt
      left.txt
      right.txt
    episode_002/
      ...

"""
from __future__ import annotations

import ast
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# ============================================================
# Interpolation helper (unchanged from your code)
# ============================================================

def _interp_seq_torch(seq: torch.Tensor, target_len: int, mode: str = "linear") -> torch.Tensor:
    """
    Resample a sequence [L, D] -> [target_len, D] using torch interpolation.
    - mode: "linear" (good for continuous) or "nearest" (good for discrete-ish controls)
    """
    if target_len <= 0:
        raise ValueError("target_len must be > 0")
    if seq.ndim != 2:
        raise ValueError(f"Expected seq [L,D], got {tuple(seq.shape)}")

    L, D = seq.shape
    seq = seq.to(dtype=torch.float32)

    if L == target_len:
        return seq
    if L == 0:
        return torch.zeros((target_len, D), dtype=torch.float32)
    if L == 1:
        return seq.repeat(target_len, 1)

    x = seq.transpose(0, 1).unsqueeze(0)  # [1, D, L]
    if mode == "linear":
        y = F.interpolate(x, size=target_len, mode="linear", align_corners=True)
    elif mode == "nearest":
        y = F.interpolate(x, size=target_len, mode="nearest")
    else:
        raise ValueError("mode must be 'linear' or 'nearest'")
    return y.squeeze(0).transpose(0, 1)  # [target_len, D]


# ============================================================
# Action conventions (same as your current code)
# ============================================================

EXPECTED_DIMS_ALL: Dict[str, int] = {"n2": 6, "n5": 6, "lh": 10, "rh": 10}

DEFAULT_INITIALS_ALL: Dict[str, np.ndarray] = {
    "n2": np.zeros(6, dtype=np.float64),             # degrees
    "n5": np.zeros(6, dtype=np.float64),             # degrees
    "lh": np.full(10, 255.0, dtype=np.float64),      # 0..255 controls
    "rh": np.full(10, 255.0, dtype=np.float64),      # 0..255 controls
}


def _make_constant_seq(num_steps: int, init_vec: np.ndarray) -> np.ndarray:
    return np.tile(init_vec[None, :], (num_steps, 1)).astype(np.float64, copy=False)


def _expand_or_trim_to_dim(q: np.ndarray, target_dim: int, init_vec: np.ndarray) -> np.ndarray:
    q = np.asarray(q)
    if q.ndim != 2:
        raise ValueError(f"Expected q to be 2D (N,D). Got shape {q.shape}.")
    N, D = q.shape
    if D == target_dim:
        return q
    out = np.empty((N, target_dim), dtype=q.dtype)
    m = min(D, target_dim)
    out[:, :m] = q[:, :m]
    if target_dim > D:
        out[:, m:] = init_vec[m:].reshape(1, -1)
    return out


def _fill_nans_with_init(q: np.ndarray, init_vec: np.ndarray) -> np.ndarray:
    q = np.asarray(q).copy()
    bad = ~np.isfinite(q)
    if bad.any():
        for j in range(q.shape[1]):
            col_bad = bad[:, j]
            if col_bad.any():
                q[col_bad, j] = init_vec[j]
    return q


# ============================================================
# ROS-like TXT action loading (same logic as your current code)
# ============================================================

def load_ros_txt_positions(txt_path: Union[str, Path], key: str = "position") -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse ROS-like text with blocks:
      seq: <int>
      secs: <int>
      nsecs: <int>
      position: [ ... ]

    Returns:
      t: (N,) seconds from 0
      q: (N, D)
    """
    txt_path = str(txt_path)
    t_list: List[float] = []
    q_list: List[np.ndarray] = []
    secs: Optional[int] = None
    nsecs: int = 0

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s.startswith("secs:"):
                try:
                    secs = int(s.split(":", 1)[1].strip())
                except Exception:
                    secs = None
            elif s.startswith("nsecs:"):
                try:
                    nsecs = int(s.split(":", 1)[1].strip())
                except Exception:
                    nsecs = 0
            elif s.startswith(f"{key}:"):
                payload = s.split(":", 1)[1].strip()
                arr_list = ast.literal_eval(payload)
                arr = np.asarray(arr_list, dtype=np.float64).reshape(-1)
                q_list.append(arr)
                if secs is not None:
                    t_list.append(float(secs) + float(nsecs or 0) * 1e-9)
                else:
                    t_list.append(len(t_list) * 1.0)

    if not q_list:
        raise RuntimeError(f"No '{key}:' entries found in {txt_path}")

    t = np.asarray(t_list, dtype=np.float64)
    q = np.vstack(q_list)

    t -= t[0]
    uniq, idx = np.unique(t, return_index=True)
    return uniq, q[idx]


def _load_actions_dir(action_dir: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Loads available action groups from one episode action directory.

    Files:
      n2 -> nova2.txt
      n5 -> nova5.txt
      lh -> left.txt
      rh -> right.txt
    """
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    mapping = {"n2": "nova2.txt", "n5": "nova5.txt", "lh": "left.txt", "rh": "right.txt"}

    for k, fname in mapping.items():
        p = action_dir / fname
        if p.exists():
            try:
                t, q = load_ros_txt_positions(p)
                out[k] = (t, q.astype(np.float64))
            except Exception:
                # Treat unreadable file as missing, instead of killing the dataset
                continue
    return out


# ============================================================
# Point cloud loading (Open3D) - same as your code
# ============================================================

def _read_point_cloud_open3d(path: Path) -> "tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]":
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Open3D read unexpected points shape for {path}: {pts.shape}")
    if pts.shape[0] == 0:
        raise ValueError(f"Empty point cloud: {path}")

    cols = None
    nors = None
    if pcd.has_colors():
        cols = np.asarray(pcd.colors, dtype=np.float32)
        if cols.shape != pts.shape:
            cols = None
    if pcd.has_normals():
        nors = np.asarray(pcd.normals, dtype=np.float32)
        if nors.shape != pts.shape:
            nors = None

    return pts, cols, nors


def _sample_or_pad(
    pts: np.ndarray,
    cols: Optional[np.ndarray],
    nors: Optional[np.ndarray],
    num_points: int,
    rng: np.random.Generator,
    method: str = "random",
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    if num_points <= 0:
        raise ValueError("num_points must be > 0")
    N = pts.shape[0]
    if N == num_points:
        return pts, cols, nors

    if N > num_points:
        if method == "first":
            idx = np.arange(num_points)
        elif method == "random":
            idx = rng.choice(N, size=num_points, replace=False)
        else:
            raise ValueError("sampling method must be 'random' or 'first'")
    else:
        idx = rng.choice(N, size=num_points, replace=True)

    pts2 = pts[idx]
    cols2 = cols[idx] if cols is not None else None
    nors2 = nors[idx] if nors is not None else None
    return pts2, cols2, nors2


def _pack_features(
    pts: np.ndarray,
    cols: Optional[np.ndarray],
    nors: Optional[np.ndarray],
    features: Tuple[str, ...],
) -> np.ndarray:
    blocks: List[np.ndarray] = []
    for f in features:
        f = f.lower()
        if f == "xyz":
            blocks.append(pts)
        elif f == "rgb":
            if cols is None:
                blocks.append(np.zeros_like(pts, dtype=np.float32))
            else:
                blocks.append(cols.astype(np.float32, copy=False))
        elif f in ("normal", "normals"):
            if nors is None:
                blocks.append(np.zeros_like(pts, dtype=np.float32))
            else:
                blocks.append(nors.astype(np.float32, copy=False))
        else:
            raise ValueError(f"Unknown point cloud feature '{f}'. Use ('xyz','rgb','normal').")
    return np.concatenate(blocks, axis=1).astype(np.float32, copy=False)


# ============================================================
# Tactile CSV/LOG loading (based on your tactile scripts)
#   - Supports GUI log: timestamp_ms + ch{ch}_p{k}
#   - Also supports vector-style header: ch{ch}_p{p}_r{row}c{col}
# ============================================================

_TACTILE_RE_FLAT = re.compile(r"^ch(?P<ch>\d+)_p(?P<p>\d+)$")
_TACTILE_RE_GRID = re.compile(r"^ch(?P<ch>\d+)_p(?P<p>\d+)_r(?P<row>\d+)c(?P<col>\d+)$")


def _safe_float(s: Optional[str]) -> float:
    if s is None:
        return 0.0
    s = s.strip()
    if s == "":
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


@dataclass
class TactileLog:
    t_sec: np.ndarray          # (T,)
    data: np.ndarray           # (T, C, P) float32
    channels: int
    points: int
    rows: Optional[int]
    cols: Optional[int]
    header_style: str          # "flat" | "grid" | "unknown"


def load_tactile_log(path: Union[str, Path]) -> TactileLog:
    """
    Loads tactile.logs (CSV-like).

    Supported header styles:
      1) GUI style (from gui_control_tactile_record.py):
         timestamp_ms, ch0_p0, ..., ch0_p95, ch1_p0, ..., ch4_p95

      2) Grid style (from draw_tactile_vectors.py style):
         timestamp_ms, ch0_p0_r0c0, ..., ch0_p?_r11c7, ch1_p0_r0c0, ...

    Returns:
      TactileLog with data shaped (T, C, P).
    """
    path = Path(path)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader, None)
        if headers is None:
            raise ValueError(f"Empty tactile log: {path}")

        # Timestamp column
        if "timestamp_ms" in headers:
            t_idx = headers.index("timestamp_ms")
        else:
            # Fallback: assume col0 is time
            t_idx = 0

        # Detect header style and build mapping
        any_grid = False
        any_flat = False

        # grid maps: ch -> {(row,col): col_index}
        grid_map: Dict[int, Dict[Tuple[int, int], int]] = {}
        max_row: Dict[int, int] = {}
        max_col: Dict[int, int] = {}

        # flat maps: ch -> {p: col_index}
        flat_map: Dict[int, Dict[int, int]] = {}
        max_p: Dict[int, int] = {}

        for j, name in enumerate(headers):
            if j == t_idx:
                continue

            m = _TACTILE_RE_GRID.match(name)
            if m:
                any_grid = True
                ch = int(m.group("ch"))
                row = int(m.group("row"))
                col = int(m.group("col"))
                grid_map.setdefault(ch, {})[(row, col)] = j
                max_row[ch] = max(max_row.get(ch, -1), row)
                max_col[ch] = max(max_col.get(ch, -1), col)
                continue

            m = _TACTILE_RE_FLAT.match(name)
            if m:
                any_flat = True
                ch = int(m.group("ch"))
                p = int(m.group("p"))
                flat_map.setdefault(ch, {})[p] = j
                max_p[ch] = max(max_p.get(ch, -1), p)
                continue

        if any_grid:
            header_style = "grid"
            channels = (max(grid_map.keys()) + 1) if grid_map else 0
            rows = max(max_row.values()) + 1 if max_row else None
            cols = max(max_col.values()) + 1 if max_col else None
            if rows is None or cols is None:
                # something is wrong in the header
                rows, cols = None, None
                points = 0
            else:
                points = rows * cols

            # Precompute column indices for each (ch, p_flat)
            col_idx = [[-1] * points for _ in range(channels)]
            if rows is not None and cols is not None:
                for ch, idx_map in grid_map.items():
                    if ch < 0 or ch >= channels:
                        continue
                    for (r, c), j in idx_map.items():
                        if 0 <= r < rows and 0 <= c < cols:
                            p_flat = r * cols + c
                            col_idx[ch][p_flat] = j

        elif any_flat:
            header_style = "flat"
            channels = (max(flat_map.keys()) + 1) if flat_map else 0
            points = (max(max_p.values()) + 1) if max_p else 0
            rows, cols = None, None

            col_idx = [[-1] * points for _ in range(channels)]
            for ch, pmap in flat_map.items():
                if ch < 0 or ch >= channels:
                    continue
                for p, j in pmap.items():
                    if 0 <= p < points:
                        col_idx[ch][p] = j
        else:
            header_style = "unknown"
            channels, points, rows, cols = 0, 0, None, None
            col_idx = []

        # Load rows
        t_ms_list: List[float] = []
        frames: List[np.ndarray] = []

        for row in reader:
            if not row:
                continue

            # timestamp
            if t_idx < len(row):
                t_ms = _safe_float(row[t_idx])
            else:
                t_ms = float(len(t_ms_list)) * 1.0
            t_ms_list.append(t_ms)

            if channels <= 0 or points <= 0:
                continue

            mat = np.zeros((channels, points), dtype=np.float32)
            for ch in range(channels):
                idxs = col_idx[ch]
                for p in range(points):
                    j = idxs[p]
                    if 0 <= j < len(row):
                        mat[ch, p] = float(_safe_float(row[j]))
                    else:
                        mat[ch, p] = 0.0
            frames.append(mat)

    if not frames:
        # Still return something consistent (length 1), so training doesn't explode
        t_sec = np.array([0.0], dtype=np.float64)
        data = np.zeros((1, max(channels, 1), max(points, 1)), dtype=np.float32)
        return TactileLog(t_sec=t_sec, data=data, channels=max(channels, 1), points=max(points, 1),
                         rows=rows, cols=cols, header_style=header_style)

    t_ms_arr = np.asarray(t_ms_list, dtype=np.float64)
    t_ms_arr -= t_ms_arr[0]
    t_sec = t_ms_arr / 1000.0
    data = np.stack(frames, axis=0)  # (T, C, P)
    return TactileLog(t_sec=t_sec, data=data, channels=data.shape[1], points=data.shape[2],
                     rows=rows, cols=cols, header_style=header_style)


def _pad_or_trim_tactile(data: np.ndarray, channels: int, points: int, fill: float = 0.0) -> np.ndarray:
    """
    data: (T, C, P)
    Returns: (T, channels, points)
    """
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected tactile data (T,C,P), got {data.shape}")

    T, C, P = data.shape
    out = np.full((T, channels, points), fill, dtype=np.float32)
    c_use = min(C, channels)
    p_use = min(P, points)
    out[:, :c_use, :p_use] = data[:, :c_use, :p_use]
    return out


# ============================================================
# Episode dataset: ONE point cloud per episode
# ============================================================

@dataclass
class EpisodeOnePC:
    episode_id: str
    obs_dir: Path
    pc_path: Path
    act_dir: Path
    tactile_path: Optional[Path]
    actions_raw: Dict[str, Tuple[np.ndarray, np.ndarray]]  # group -> (t, q)


def _discover_action_dir(act_root: Path, episode_id: str, expected_files: Sequence[str]) -> Optional[Path]:
    expected = set(expected_files)

    def has_any(p: Path) -> bool:
        return any((p / name).exists() for name in expected)

    if act_root.is_file():
        return act_root.parent

    if act_root.is_dir():
        cand = act_root / episode_id
        if cand.is_dir() and has_any(cand):
            return cand
        if has_any(act_root):
            return act_root

        cands = [d for d in act_root.glob("*") if d.is_dir() and episode_id in d.name]
        for d in cands:
            if has_any(d):
                return d

        for d in act_root.glob("*"):
            if d.is_dir() and has_any(d):
                return d

    return None


class PointCloudTactileActionEpisodeDataset(Dataset):
    """
    Episode-level dataset: each episode has EXACTLY ONE point cloud + tactile log + action txts.

    Folder layout:
      obs_root/
        episode_001/point.ply
      tactile_root/
        episode_001/tactile.logs
      act_root/
        episode_001/{nova2.txt,nova5.txt,left.txt,right.txt}

    __getitem__ returns ONE EPISODE.
    """

    def __init__(
        self,
        obs_root: Union[str, Path],
        tactile_root: Union[str, Path],
        act_root: Union[str, Path],
        pc_name: str = "point.ply",
        tactile_name: str = "tactile.logs",
        groups: Tuple[str, ...] = ("n2", "n5", "lh", "rh"),
        missing_policy: str = "initial",  # {"initial","zero","nan"} for missing action files
        initial_overrides: Optional[Dict[str, np.ndarray]] = None,

        # Tactile shaping (default matches tactile_stream.py: 5 channels, 12*8 points)
        tactile_channels: int = 5,
        tactile_points: int = 96,
        tactile_missing_fill: float = 0.0,  # if tactile file missing/unreadable
        tactile_preload: bool = False,       # preload all tactile logs into RAM

        # Point cloud shaping:
        to_tensor: bool = True,
        pc_transform: Optional[Callable[[np.ndarray], Any]] = None,
        num_points: Optional[int] = None,
        sampling: str = "random",
        features: Tuple[str, ...] = ("xyz",),

        include_episodes: Optional[List[str]] = None,
        seed: int = 0,
    ):
        super().__init__()
        self.obs_root = Path(obs_root).expanduser()
        self.tactile_root = Path(tactile_root).expanduser()
        self.act_root = Path(act_root).expanduser()

        self.pc_name = pc_name
        self.tactile_name = tactile_name
        self.groups = tuple(groups)
        self.missing_policy = str(missing_policy)
        self.include_episodes = include_episodes

        self.tactile_channels = int(tactile_channels)
        self.tactile_points = int(tactile_points)
        self.tactile_missing_fill = float(tactile_missing_fill)
        self.tactile_preload = bool(tactile_preload)

        self.to_tensor = bool(to_tensor)
        self.pc_transform = pc_transform
        self.num_points = num_points
        self.sampling = sampling
        self.features = tuple(features)

        self._rng = np.random.default_rng(int(seed))

        # Expected dims / initials
        self.expected_dims = dict(EXPECTED_DIMS_ALL)
        self.initials = {k: DEFAULT_INITIALS_ALL[k].copy() for k in DEFAULT_INITIALS_ALL}
        if initial_overrides:
            for k, v in initial_overrides.items():
                v = np.asarray(v, dtype=np.float64).reshape(-1)
                exp = self.expected_dims.get(k)
                if exp is None:
                    continue
                if v.shape[0] != exp:
                    raise ValueError(f"initial_overrides['{k}'] must have length {exp}, got {v.shape[0]}")
                self.initials[k] = v

        if not self.obs_root.is_dir():
            raise NotADirectoryError(f"obs_root must be a directory: {self.obs_root}")

        # Discover episodes: obs_root/*/pc_name
        episodes: List[EpisodeOnePC] = []

        # Case A: obs_root itself is an episode folder
        pc_here = self.obs_root / self.pc_name
        if pc_here.exists():
            ep_id = self.obs_root.name
            if self.include_episodes and ep_id not in self.include_episodes:
                raise FileNotFoundError(f"Requested episode '{ep_id}' not in include_episodes.")

            act_dir = _discover_action_dir(
                self.act_root, ep_id,
                expected_files=("nova2.txt", "nova5.txt", "left.txt", "right.txt")
            )
            if act_dir is None:
                raise FileNotFoundError(f"Could not find action files for episode '{ep_id}' under '{self.act_root}'")

            tactile_path = (self.tactile_root / ep_id / self.tactile_name)
            if not tactile_path.exists():
                tactile_path = None

            actions_raw = _load_actions_dir(act_dir)
            episodes.append(EpisodeOnePC(
                episode_id=ep_id, obs_dir=self.obs_root, pc_path=pc_here,
                act_dir=act_dir, tactile_path=tactile_path, actions_raw=actions_raw
            ))

        else:
            # Case B: obs_root is a parent directory containing episode subfolders
            subdirs = sorted([p for p in self.obs_root.iterdir() if p.is_dir()])
            for ep_dir in subdirs:
                ep_id = ep_dir.name
                if self.include_episodes and ep_id not in self.include_episodes:
                    continue

                pc_path = ep_dir / self.pc_name
                if not pc_path.exists():
                    # fallback: if user didn't name it point.ply, pick first ply/pcd-like file
                    candidates = sorted(ep_dir.glob("*.ply")) + sorted(ep_dir.glob("*.pcd"))
                    if not candidates:
                        continue
                    pc_path = candidates[0]

                act_dir = _discover_action_dir(
                    self.act_root, ep_id,
                    expected_files=("nova2.txt", "nova5.txt", "left.txt", "right.txt")
                )
                if act_dir is None:
                    raise FileNotFoundError(f"Could not find action files for episode '{ep_id}' under '{self.act_root}'")

                tactile_path = (self.tactile_root / ep_id / self.tactile_name)
                if not tactile_path.exists():
                    tactile_path = None

                actions_raw = _load_actions_dir(act_dir)

                episodes.append(EpisodeOnePC(
                    episode_id=ep_id, obs_dir=ep_dir, pc_path=pc_path,
                    act_dir=act_dir, tactile_path=tactile_path, actions_raw=actions_raw
                ))

        if not episodes:
            raise FileNotFoundError(f"No episodes found under {self.obs_root} with pc_name='{self.pc_name}'")

        self.episodes = episodes

        # Optional tactile preload + cache
        self._tactile_cache: Dict[int, TactileLog] = {}
        if self.tactile_preload:
            for i in range(len(self.episodes)):
                _ = self._get_tactile(i)

    def __len__(self) -> int:
        return len(self.episodes)

    def _get_tactile(self, idx: int) -> TactileLog:
        if idx in self._tactile_cache:
            return self._tactile_cache[idx]

        ep = self.episodes[idx]
        if ep.tactile_path is None:
            # missing tactile -> length 1 filler
            t = np.array([0.0], dtype=np.float64)
            data = np.full((1, self.tactile_channels, self.tactile_points),
                           self.tactile_missing_fill, dtype=np.float32)
            tl = TactileLog(t_sec=t, data=data, channels=self.tactile_channels, points=self.tactile_points,
                            rows=None, cols=None, header_style="missing")
            self._tactile_cache[idx] = tl
            return tl

        try:
            tl = load_tactile_log(ep.tactile_path)
            # enforce shape
            tl_data = _pad_or_trim_tactile(tl.data, self.tactile_channels, self.tactile_points,
                                           fill=self.tactile_missing_fill)
            tl = TactileLog(
                t_sec=tl.t_sec,
                data=tl_data,
                channels=self.tactile_channels,
                points=self.tactile_points,
                rows=tl.rows,
                cols=tl.cols,
                header_style=tl.header_style,
            )
        except Exception:
            t = np.array([0.0], dtype=np.float64)
            data = np.full((1, self.tactile_channels, self.tactile_points),
                           self.tactile_missing_fill, dtype=np.float32)
            tl = TactileLog(t_sec=t, data=data, channels=self.tactile_channels, points=self.tactile_points,
                            rows=None, cols=None, header_style="unreadable")

        self._tactile_cache[idx] = tl
        return tl

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ep = self.episodes[idx]

        # ---- point cloud ----
        pts, cols, nors = _read_point_cloud_open3d(ep.pc_path)

        if self.num_points is not None:
            pts, cols, nors = _sample_or_pad(
                pts, cols, nors,
                num_points=int(self.num_points),
                rng=self._rng,
                method=self.sampling,
            )

        feat = _pack_features(pts, cols, nors, self.features)  # (N,F)

        if self.pc_transform is not None:
            feat_out = self.pc_transform(feat)
        else:
            feat_out = feat

        if self.to_tensor and isinstance(feat_out, np.ndarray):
            pc_tensor = torch.from_numpy(feat_out).float()
        else:
            pc_tensor = feat_out

        # ---- actions (raw sequences, per file) ----
        action_seq: Dict[str, torch.Tensor] = {}
        action_t: Dict[str, torch.Tensor] = {}

        for k in self.groups:
            expD = self.expected_dims.get(k)
            if expD is None:
                continue
            init_vec = self.initials.get(k, np.zeros(expD, dtype=np.float64))

            if k in ep.actions_raw:
                t_src, q_src = ep.actions_raw[k]
                q_src = _expand_or_trim_to_dim(q_src, expD, init_vec)
                q_src = _fill_nans_with_init(q_src, init_vec)
                # if empty, fallback to length 1
                if q_src.shape[0] == 0:
                    q_src = init_vec[None, :]
                    t_src = np.array([0.0], dtype=np.float64)
            else:
                if self.missing_policy == "initial":
                    q_src = init_vec[None, :]
                elif self.missing_policy == "zero":
                    q_src = np.zeros((1, expD), dtype=np.float64)
                elif self.missing_policy == "nan":
                    q_src = np.full((1, expD), np.nan, dtype=np.float64)
                else:
                    raise ValueError("missing_policy must be one of {'initial','zero','nan'}")
                t_src = np.array([0.0], dtype=np.float64)

            action_seq[k] = torch.from_numpy(np.asarray(q_src, dtype=np.float32))
            action_t[k] = torch.from_numpy(np.asarray(t_src, dtype=np.float32))

        # ---- tactile ----
        tl = self._get_tactile(idx)
        tactile_seq = torch.from_numpy(tl.data.astype(np.float32, copy=False))  # (T,C,P)
        tactile_t = torch.from_numpy(tl.t_sec.astype(np.float32, copy=False))   # (T,)

        sample: Dict[str, Any] = {
            "point_cloud": pc_tensor,
            "action_seq": action_seq,
            "action_t": action_t,
            "tactile_seq": tactile_seq,
            "tactile_t": tactile_t,
            "t_frame": 0.0,  # single observation point cloud
            "meta": {
                "episode_id": ep.episode_id,
                "episode_idx": int(idx),
                "pc_path": str(ep.pc_path),
                "obs_dir": str(ep.obs_dir),
                "act_dir": str(ep.act_dir),
                "tactile_path": None if ep.tactile_path is None else str(ep.tactile_path),
                "tactile_header_style": tl.header_style,
                "tactile_channels": int(self.tactile_channels),
                "tactile_points": int(self.tactile_points),
            },
        }
        return sample


# ============================================================
# Collate: (B episodes) -> fixed chunk_len sequences
#   - Resamples action_seq (per group) to chunk_len
#   - Resamples tactile_seq to chunk_len
# ============================================================

def collate_pointcloud_to_action_chunk(
    batch: List[Dict[str, Any]],
    chunk_len: int = 64,
    interp_mode_map: Optional[Dict[str, str]] = None,
    prefer_action_seq_key: str = "action_seq",
    tactile_key: str = "tactile_seq",
) -> Dict[str, Any]:
    """
    Episode-level collate:
      - Each item is ONE point cloud
      - Each item provides an action SEQUENCE (variable length)
      - Each item provides a tactile SEQUENCE (variable length)
      - We interpolate/resample each sequence to fixed chunk_len

    Output:
      {
        "point_cloud": Tensor [B,N,F] or List[Tensor [Ni,F]],
        "actions": {k: Tensor [B,chunk_len,Dk]},
        "tactile": Tensor [B,chunk_len,C,P],
        "t_frame": Tensor [B],
        "meta": List[dict],
        "action_seq_lens": {k: Tensor [B]},
        "tactile_seq_lens": Tensor [B],
      }
    """
    if len(batch) == 0:
        raise ValueError("Empty batch for collate_pointcloud_to_action_chunk")

    # --- point clouds ---
    pcs = [b["point_cloud"] for b in batch]
    if isinstance(pcs[0], torch.Tensor):
        try:
            pc_out: Union[torch.Tensor, List[torch.Tensor]] = torch.stack(pcs, dim=0)
        except RuntimeError:
            pc_out = pcs
    else:
        pc_out = pcs

    # --- times + meta ---
    t_frame = torch.tensor([float(b.get("t_frame", 0.0)) for b in batch], dtype=torch.float32)
    meta = [b.get("meta", {}) for b in batch]

    # Decide interpolation mode per action group
    default_interp_mode_map = {"n2": "linear", "n5": "linear", "lh": "nearest", "rh": "nearest"}
    interp_mode_map = {**default_interp_mode_map, **(interp_mode_map or {})}

    # --- actions ---
    all_keys = set()
    for b in batch:
        if prefer_action_seq_key in b and isinstance(b[prefer_action_seq_key], dict):
            all_keys |= set(b[prefer_action_seq_key].keys())
        if "actions" in b and isinstance(b["actions"], dict):
            all_keys |= set(b["actions"].keys())

    actions_out: Dict[str, torch.Tensor] = {}
    seq_lens_out: Dict[str, torch.Tensor] = {}

    for k in sorted(all_keys):
        mode = interp_mode_map.get(k, "linear")
        seqs_k: List[torch.Tensor] = []
        lens_k: List[int] = []

        ref_D: Optional[int] = None
        for b in batch:
            q = None
            if prefer_action_seq_key in b and k in b[prefer_action_seq_key]:
                q = b[prefer_action_seq_key][k]
            elif "actions" in b and k in b["actions"]:
                q = b["actions"][k]
            if isinstance(q, torch.Tensor):
                if q.ndim == 1:
                    ref_D = int(q.shape[0])
                elif q.ndim == 2:
                    ref_D = int(q.shape[1])
                break
        if ref_D is None:
            continue

        for b in batch:
            if prefer_action_seq_key in b and k in b[prefer_action_seq_key]:
                q = b[prefer_action_seq_key][k]
            elif "actions" in b and k in b["actions"]:
                q1 = b["actions"][k]
                q = q1.unsqueeze(0) if q1.ndim == 1 else q1
            else:
                q = torch.full((1, ref_D), float("nan"), dtype=torch.float32)

            if q.ndim == 1:
                q = q.unsqueeze(0)
            if q.ndim != 2:
                raise ValueError(f"Bad action tensor for key={k}: shape={tuple(q.shape)}")

            L = int(q.shape[0])
            lens_k.append(L)
            q_rs = _interp_seq_torch(q, target_len=chunk_len, mode=mode)
            seqs_k.append(q_rs)

        actions_out[k] = torch.stack(seqs_k, dim=0)  # [B,chunk_len,D]
        seq_lens_out[k] = torch.tensor(lens_k, dtype=torch.int64)

    # --- tactile ---
    tactile_out = None
    tactile_lens: List[int] = []

    # Find reference C,P
    ref_C = None
    ref_P = None
    for b in batch:
        ts = b.get(tactile_key, None)
        if isinstance(ts, torch.Tensor) and ts.ndim == 3:
            ref_C = int(ts.shape[1])
            ref_P = int(ts.shape[2])
            break

    if ref_C is not None and ref_P is not None:
        tac_rs_list: List[torch.Tensor] = []
        for b in batch:
            ts = b.get(tactile_key, None)
            if not isinstance(ts, torch.Tensor):
                ts = torch.full((1, ref_C, ref_P), float("nan"), dtype=torch.float32)

            if ts.ndim != 3:
                raise ValueError(f"{tactile_key} must be Tensor [T,C,P], got {tuple(ts.shape)}")

            Lt = int(ts.shape[0])
            tactile_lens.append(Lt)

            # flatten -> interpolate -> unflatten
            flat = ts.reshape(Lt, ref_C * ref_P)
            flat_rs = _interp_seq_torch(flat, target_len=chunk_len, mode="linear")
            tac_rs = flat_rs.reshape(chunk_len, ref_C, ref_P)
            tac_rs_list.append(tac_rs)

        tactile_out = torch.stack(tac_rs_list, dim=0)  # [B,chunk_len,C,P]

    out = {
        "point_cloud": pc_out,
        "actions": actions_out,
        "t_frame": t_frame,
        "meta": meta,
        "action_seq_lens": seq_lens_out,
    }
    if tactile_out is not None:
        out["tactile"] = tactile_out
        out["tactile_seq_lens"] = torch.tensor(tactile_lens, dtype=torch.int64)
    return out
