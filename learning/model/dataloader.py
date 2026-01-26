"""
pointcloud_action_datasets.py

Folder layout (typical):
Each observation only has ONE point cloud per episode.

# so you should shuffle over episodes, not inside the episodes
  obs_root/
    episode_001/
      point.ply
    episode_002/
      point.ply

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import OrderedDict


# ============================================================
# Interpolation helper (unchanged)
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



import numpy as np

def _frontload_time_grid(
    t0: float,
    t1: float,
    L: int,
    early_time_frac: float = 0.25,   # "first quarter of time"
    early_sample_frac: float = 0.80, # "80% of samples"
) -> np.ndarray:
    """
    Return L target times in [t0, t1] such that early_time_frac of the time span
    contains ~early_sample_frac of the samples.

    Early segment includes the pivot; late segment starts after pivot (no duplicate pivot).
    """
    t0 = float(t0); t1 = float(t1)
    if L <= 1:
        return np.array([t0], dtype=np.float64)
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        return np.full((L,), t0, dtype=np.float64)

    # Clamp fractions to safe ranges
    early_time_frac  = float(np.clip(early_time_frac,  0.0, 1.0))
    early_sample_frac = float(np.clip(early_sample_frac, 0.0, 1.0))

    # If L is tiny, you can't really enforce 80/20—just return endpoints.
    if L <= 2 or early_time_frac in (0.0, 1.0) or early_sample_frac in (0.0, 1.0):
        return np.linspace(t0, t1, L, dtype=np.float64)

    # Number of samples in early segment (ensure at least 1 late sample)
    n_early = int(round(early_sample_frac * L))
    n_early = max(1, min(L - 1, n_early))
    n_late  = L - n_early  # >= 1 guaranteed

    pivot = early_time_frac  # normalized pivot in [0,1]

    # Early normalized times: [0 .. pivot] (includes pivot if n_early>1)
    u_early = np.linspace(0.0, pivot, n_early, endpoint=True, dtype=np.float64)

    # Late normalized times: (pivot .. 1] (exclude pivot to avoid duplication)
    u_late = np.linspace(pivot, 1.0, n_late + 1, endpoint=True, dtype=np.float64)[1:]

    u = np.concatenate([u_early, u_late])  # length L
    return t0 + u * (t1 - t0)



# ============================================================
# Action conventions
# ============================================================

EXPECTED_DIMS_ALL: Dict[str, int] = {"n2": 6, "n5": 6, "lh": 10, "rh": 10}

DEFAULT_INITIALS_ALL: Dict[str, np.ndarray] = {
    "n2": np.zeros(6, dtype=np.float64),             # degrees
    "n5": np.zeros(6, dtype=np.float64),             # degrees
    "lh": np.full(10, 255.0, dtype=np.float64),      # 0..255 controls
    "rh": np.full(10, 255.0, dtype=np.float64),      # 0..255 controls
}


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
# ROS-like TXT action loading
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
# Point cloud loading (Open3D)
# ============================================================

def _read_point_cloud_open3d(path: Path) -> "tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]":
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.vertices) == 0:
        raise ValueError(f"Empty mesh/ply: {path}")

    pts = np.asarray(mesh.vertices, dtype=np.float32)

    cols: Optional[np.ndarray] = None
    nors: Optional[np.ndarray] = None

    if mesh.has_vertex_colors():
        cols = np.asarray(mesh.vertex_colors, dtype=np.float32)
        if cols.shape != pts.shape:
            cols = None

    if mesh.has_vertex_normals():
        nors = np.asarray(mesh.vertex_normals, dtype=np.float32)
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
            blocks.append(cols.astype(np.float32, copy=False) if cols is not None else np.zeros_like(pts, dtype=np.float32))
        elif f in ("normal", "normals"):
            blocks.append(nors.astype(np.float32, copy=False) if nors is not None else np.zeros_like(pts, dtype=np.float32))
        else:
            raise ValueError(f"Unknown point cloud feature '{f}'. Use ('xyz','rgb','normal').")
    return np.concatenate(blocks, axis=1).astype(np.float32, copy=False)


# ============================================================
# Episode dataset: ONE point cloud per episode
# ============================================================

@dataclass
class EpisodeOnePC:
    episode_id: str
    obs_dir: Path
    pc_path: Path
    act_dir: Path
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

import numpy as np
from typing import Tuple
def _resample_action_np(
    t_src: np.ndarray,
    q_src: np.ndarray,
    target_len: int,
    mode: str = "linear",
    time_strategy: str = "uniform",     # NEW
    early_time_frac: float = 0.25,      # NEW
    early_sample_frac: float = 0.80,    # NEW
):
    t_src = np.asarray(t_src, dtype=np.float64).reshape(-1)
    q_src = np.asarray(q_src, dtype=np.float64)

    # Basic safety
    if q_src.ndim == 1:
        q_src = q_src[:, None]
    if t_src.size != q_src.shape[0]:
        raise ValueError(f"t_src len {t_src.size} != q_src len {q_src.shape[0]}")
    if t_src.size == 0:
        t0 = 0.0
        t_rs = np.full((target_len,), t0, dtype=np.float64)
        q_rs = np.zeros((target_len, q_src.shape[1]), dtype=np.float64)
        return t_rs, q_rs

    # Sort by time
    order = np.argsort(t_src)
    t_src = t_src[order]
    q_src = q_src[order]

    # Drop duplicate timestamps (keep last occurrence)
    # (np.interp behaves poorly with repeated x values)
    rev = t_src[::-1]
    _, idx_rev = np.unique(rev, return_index=True)
    keep = (t_src.size - 1 - idx_rev)
    keep = np.sort(keep)
    t_src = t_src[keep]
    q_src = q_src[keep]

    t0, t1 = float(t_src[0]), float(t_src[-1])

    # --- NEW: choose target times ---
    if time_strategy == "uniform":
        t_rs = np.linspace(t0, t1, int(target_len), dtype=np.float64)
    elif time_strategy in ("frontload", "early80_first25"):
        t_rs = _frontload_time_grid(
            t0, t1, int(target_len),
            early_time_frac=early_time_frac,
            early_sample_frac=early_sample_frac,
        )
    else:
        raise ValueError(f"Unknown time_strategy: {time_strategy}")

    # --- interpolate ---
    D = q_src.shape[1]
    q_rs = np.empty((t_rs.size, D), dtype=np.float64)

    if mode == "linear":
        for d in range(D):
            q_rs[:, d] = np.interp(t_rs, t_src, q_src[:, d])
    elif mode == "nearest":
        # nearest neighbor in time
        idx = np.searchsorted(t_src, t_rs, side="left")
        idx = np.clip(idx, 1, len(t_src) - 1)
        left = idx - 1
        right = idx
        choose_right = (t_rs - t_src[left]) >= (t_src[right] - t_rs)
        nn = np.where(choose_right, right, left)
        q_rs = q_src[nn]
    elif mode in ("previous", "zoh"):
        # zero-order hold (previous sample)
        idx = np.searchsorted(t_src, t_rs, side="right") - 1
        idx = np.clip(idx, 0, len(t_src) - 1)
        q_rs = q_src[idx]
    else:
        raise ValueError(f"Unknown interp mode: {mode}")

    return t_rs, q_rs


_MESH_EXTS = {".stl", ".obj", ".off", ".glb", ".gltf"}


def _action_dir_has_expected_files(act_dir: Path, expected_files: Tuple[str, ...]) -> bool:
    return act_dir.is_dir() and all((act_dir / f).is_file() for f in expected_files)


def _read_geometry_as_point_cloud_open3d(
    path: Path,
    *,
    mesh_sample_points: int = 8192,
    mesh_sampling: str = "surface",  # "surface" | "vertices"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (pts, cols, nors) as numpy arrays:
      pts  (N,3) float64
      cols (N,3) float64 (zeros if unavailable)
      nors (N,3) float64 (zeros if unavailable)

    - For point clouds: uses your existing _read_point_cloud_open3d(path)
    - For meshes: reads triangle mesh, then converts to point samples.
    """
    suf = path.suffix.lower()

    # Mesh -> sample points
    if suf in _MESH_EXTS:
        import open3d as o3d

        mesh = o3d.io.read_triangle_mesh(str(path))
        if mesh.is_empty():
            raise ValueError(f"Mesh is empty or unreadable: {path}")

        mesh_sampling = str(mesh_sampling).lower()

        if mesh_sampling == "vertices":
            # Deterministic, uses mesh vertices (may be non-uniform depending on STL)
            mesh.compute_vertex_normals()
            pts = np.asarray(mesh.vertices, dtype=np.float64)
            nors = np.asarray(mesh.vertex_normals, dtype=np.float64) if len(mesh.vertex_normals) else np.zeros_like(pts)
            cols = np.asarray(mesh.vertex_colors, dtype=np.float64) if len(mesh.vertex_colors) else np.zeros_like(pts)
            if pts.size == 0:
                raise ValueError(f"Mesh has no vertices: {path}")
            return pts, cols, nors

        # Default: sample from surface
        n = int(mesh_sample_points)
        n = max(n, 1)
        pcd = mesh.sample_points_uniformly(number_of_points=n, use_triangle_normal=True)

        pts = np.asarray(pcd.points, dtype=np.float64)
        nors = np.asarray(pcd.normals, dtype=np.float64) if len(pcd.normals) else np.zeros_like(pts)
        cols = np.asarray(pcd.colors, dtype=np.float64) if len(pcd.colors) else np.zeros_like(pts)

        if pts.size == 0:
            raise ValueError(f"Surface sampling produced 0 points for mesh: {path}")
        return pts, cols, nors

    # Point cloud -> your existing reader
    return _read_point_cloud_open3d(path)


class PointCloudActionEpisodeDataset(Dataset):
    def __init__(
        self,
        obs_root: Union[str, Path],
        act_root: Optional[Union[str, Path]] = None,  # NEW: can be None -> same as obs_root
        pc_name: str = "point.ply",
        groups: Tuple[str, ...] = ("n2", "n5", "lh", "rh"),
        missing_policy: str = "initial",
        initial_overrides: Optional[Dict[str, np.ndarray]] = None,

        # Point cloud shaping:
        to_tensor: bool = True,
        pc_transform: Optional[Callable[[np.ndarray], Any]] = None,
        num_points: Optional[int] = None,
        sampling: str = "random",
        features: Tuple[str, ...] = ("xyz",),

        include_episodes: Optional[List[str]] = None,
        seed: int = 0,

        # Action resampling
        action_len: Optional[int] = 64,
        interp_mode_map: Optional[Dict[str, str]] = None,
        action_format: str = "concat",   # "dict" | "concat" | "flat"

        # Caching
        cache_point_cloud: bool = True,
        pc_cache_max_items: int = 0,

        # NEW: new folder layout knobs
        mesh_subdir: str = "mesh",
        mesh_name: str = "strawberry.stl",
        control_subdir: str = "control",

        # NEW: mesh->point conversion
        mesh_sampling: str = "surface",  # "surface" (uniform on triangles) | "vertices"
        mesh_points: int = 8192,         # how many points to sample from the mesh surface before final sampling/padding
    ):
        super().__init__()
        self.obs_root = Path(obs_root).expanduser()
        self.act_root = Path(act_root).expanduser() if act_root is not None else self.obs_root

        self.action_len = action_len
        default_modes = {"n2": "linear", "n5": "linear", "lh": "nearest", "rh": "nearest"}
        self.interp_mode_map = {**default_modes, **(interp_mode_map or {})}
        self.action_format = str(action_format)

        self.pc_name = pc_name
        self.groups = tuple(groups)
        self.missing_policy = str(missing_policy)
        self.include_episodes = include_episodes

        self.to_tensor = bool(to_tensor)
        self.pc_transform = pc_transform
        self.num_points = num_points
        self.sampling = sampling
        self.features = tuple(features)

        self.mesh_subdir = str(mesh_subdir)
        self.mesh_name = str(mesh_name)
        self.control_subdir = str(control_subdir)
        self.mesh_sampling = str(mesh_sampling)
        self.mesh_points = int(mesh_points)

        self._rng = np.random.default_rng(int(seed))

        self.cache_point_cloud = bool(cache_point_cloud)
        self.pc_cache_max_items = int(pc_cache_max_items)
        self._pc_cache: "OrderedDict[int, Any]" = OrderedDict()

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

        expected_action_files = ("nova2.txt", "nova5.txt", "left.txt", "right.txt")

        def find_obs_path(ep_dir: Path) -> Optional[Path]:
            # NEW layout: mesh/strawberry.stl
            p_mesh = ep_dir / self.mesh_subdir / self.mesh_name
            if p_mesh.is_file():
                return p_mesh

            # OLD layout: episode_dir/point.ply (or whatever pc_name is)
            p_pc = ep_dir / self.pc_name
            if p_pc.is_file():
                return p_pc

            # Fallback: any point cloud
            candidates = sorted(ep_dir.glob("*.ply")) + sorted(ep_dir.glob("*.pcd"))
            if candidates:
                return candidates[0]

            # Fallback: any mesh
            candidates = sorted(ep_dir.glob("*.stl")) + sorted(ep_dir.glob("*.obj")) + sorted(ep_dir.glob("*.off"))
            if candidates:
                return candidates[0]

            return None

        def find_action_dir(ep_dir: Path, ep_id: str) -> Optional[Path]:
            # NEW layout: episode_dir/control/...
            cand = ep_dir / self.control_subdir
            if _action_dir_has_expected_files(cand, expected_action_files):
                return cand

            # ALSO common: act_root/ep_id/control
            cand2 = self.act_root / ep_id / self.control_subdir
            if _action_dir_has_expected_files(cand2, expected_action_files):
                return cand2

            # OLD behavior fallback (your existing helper)
            act_dir = _discover_action_dir(
                self.act_root,
                ep_id,
                expected_files=expected_action_files,
            )
            if act_dir is not None and _action_dir_has_expected_files(act_dir, expected_action_files):
                return act_dir

            return None

        episodes: List[EpisodeOnePC] = []

        # Case A: obs_root itself is an episode folder (supports new + old layouts)
        obs_path_here = find_obs_path(self.obs_root)
        if obs_path_here is not None:
            ep_id = self.obs_root.name
            if self.include_episodes and ep_id not in self.include_episodes:
                raise FileNotFoundError(f"Requested episode '{ep_id}' not in include_episodes.")

            act_dir = find_action_dir(self.obs_root, ep_id)
            if act_dir is None:
                raise FileNotFoundError(
                    f"Could not find action files for episode '{ep_id}' under '{self.obs_root}' "
                    f"(expected '{self.control_subdir}/' or old layout)."
                )

            actions_raw = _load_actions_dir(act_dir)
            episodes.append(
                EpisodeOnePC(
                    episode_id=ep_id,
                    obs_dir=self.obs_root,
                    pc_path=obs_path_here,   # NOTE: can be .stl now
                    act_dir=act_dir,
                    actions_raw=actions_raw,
                )
            )

        else:
            # Case B: obs_root is a parent directory containing episode subfolders
            subdirs = sorted([p for p in self.obs_root.iterdir() if p.is_dir()])
            for ep_dir in subdirs:
                ep_id = ep_dir.name
                if self.include_episodes and ep_id not in self.include_episodes:
                    continue

                obs_path = find_obs_path(ep_dir)
                if obs_path is None:
                    continue

                act_dir = find_action_dir(ep_dir, ep_id)
                if act_dir is None:
                    raise FileNotFoundError(
                        f"Could not find action files for episode '{ep_id}' (tried '{ep_dir/self.control_subdir}' "
                        f"and old act_root-based layout under '{self.act_root}')."
                    )

                actions_raw = _load_actions_dir(act_dir)
                episodes.append(
                    EpisodeOnePC(
                        episode_id=ep_id,
                        obs_dir=ep_dir,
                        pc_path=obs_path,  # NOTE: can be .stl now
                        act_dir=act_dir,
                        actions_raw=actions_raw,
                    )
                )

        if not episodes:
            raise FileNotFoundError(
                f"No episodes found under {self.obs_root}. "
                f"Tried '{self.mesh_subdir}/{self.mesh_name}' and '{self.pc_name}'."
            )

        self.episodes = episodes

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ep = self.episodes[idx]

        # =========================================================
        # 1) Point cloud (now supports mesh->points)
        # =========================================================
        pc_tensor = None

        if self.cache_point_cloud:
            pc_tensor = self._pc_cache.get(idx, None)
            if pc_tensor is not None:
                self._pc_cache.move_to_end(idx)

        if pc_tensor is None:
            # If mesh: sample mesh_points (or more) first, then your normal sampling/padding handles final num_points.
            # If point cloud: just read it.
            base_sample_n = self.mesh_points
            if self.num_points is not None:
                base_sample_n = max(int(self.num_points), int(self.mesh_points))

            pts, cols, nors = _read_geometry_as_point_cloud_open3d(
                ep.pc_path,
                mesh_sample_points=base_sample_n,
                mesh_sampling=self.mesh_sampling,
            )

            if self.num_points is not None:
                pts, cols, nors = _sample_or_pad(
                    pts, cols, nors,
                    num_points=int(self.num_points),
                    rng=self._rng,
                    method=self.sampling,
                )

            feat = _pack_features(pts, cols, nors, self.features)  # (N,F)
            feat_out = self.pc_transform(feat) if self.pc_transform is not None else feat
            pc_tensor = (
                torch.from_numpy(feat_out).float()
                if (self.to_tensor and isinstance(feat_out, np.ndarray))
                else feat_out
            )

            if self.cache_point_cloud:
                self._pc_cache[idx] = pc_tensor
                if self.pc_cache_max_items > 0:
                    while len(self._pc_cache) > self.pc_cache_max_items:
                        self._pc_cache.popitem(last=False)

        # =========================================================
        # 2) Actions: load + sanitize + resample to fixed length
        # =========================================================
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

            if self.action_len is not None:
                mode = self.interp_mode_map.get(k, "linear")
                t_rs, q_rs = _resample_action_np(
                    t_src, q_src,
                    target_len=int(self.action_len),
                    mode=mode,
                    time_strategy="frontload",
                    early_time_frac=0.25,
                    early_sample_frac=0.80,
                )
            else:
                t_rs, q_rs = np.asarray(t_src, dtype=np.float64), np.asarray(q_src, dtype=np.float64)

            action_seq[k] = torch.from_numpy(np.asarray(q_rs, dtype=np.float32))  # [L,D]
            action_t[k]   = torch.from_numpy(np.asarray(t_rs, dtype=np.float32))  # [L]

        # =========================================================
        # 3) Output formatting (dict / concat / flat)
        # =========================================================
        if self.action_format == "dict":
            out = {"actions": action_seq, "action_t": action_t}
        else:
            parts = [action_seq[k] for k in self.groups if k in action_seq]
            if len(parts) == 0:
                action_cat = torch.empty((int(self.action_len or 1), 0), dtype=torch.float32)
            else:
                action_cat = torch.cat(parts, dim=1)  # [L, sumD]

            if self.action_format == "concat":
                out = {"action": action_cat}
            elif self.action_format == "flat":
                out = {"action": action_cat.reshape(-1)}
            else:
                raise ValueError("action_format must be one of {'dict','concat','flat'}")

        # =========================================================
        # 4) Final sample dict
        # =========================================================
        return {
            "point_cloud": pc_tensor,
            **out,
            "t_frame": 0.0,
            "meta": {
                "episode_id": ep.episode_id,
                "episode_idx": int(idx),
                "pc_path": str(ep.pc_path),   # may be .stl now
                "obs_dir": str(ep.obs_dir),
                "act_dir": str(ep.act_dir),
            },
        }



# ============================================================
# Collate: (B episodes) -> fixed chunk_len sequences
#   - Resamples action_seq (per group) to chunk_len
# ============================================================

# def collate_pointcloud_to_action_chunk(
#     batch: List[Dict[str, Any]],
#     chunk_len: int = 64,
#     interp_mode_map: Optional[Dict[str, str]] = None,
#     prefer_action_seq_key: str = "action_seq",
# ) -> Dict[str, Any]:
#     """
#     Episode-level collate:
#       - Each item is ONE point cloud
#       - Each item provides an action SEQUENCE (variable length)
#       - We interpolate/resample each action sequence to fixed chunk_len

#     Output:
#       {
#         "point_cloud": Tensor [B,N,F] or List[Tensor [Ni,F]],
#         "actions": {k: Tensor [B,chunk_len,Dk]},
#         "t_frame": Tensor [B],
#         "meta": List[dict],
#         "action_seq_lens": {k: Tensor [B]},
#       }
#     """
#     if len(batch) == 0:
#         raise ValueError("Empty batch for collate_pointcloud_to_action_chunk")

#     # --- point clouds ---
#     pcs = [b["point_cloud"] for b in batch]
#     if isinstance(pcs[0], torch.Tensor):
#         try:
#             pc_out: Union[torch.Tensor, List[torch.Tensor]] = torch.stack(pcs, dim=0)
#         except RuntimeError:
#             pc_out = pcs
#     else:
#         pc_out = pcs

#     # --- times + meta ---
#     t_frame = torch.tensor([float(b.get("t_frame", 0.0)) for b in batch], dtype=torch.float32)
#     meta = [b.get("meta", {}) for b in batch]

#     # Decide interpolation mode per action group
#     default_interp_mode_map = {"n2": "linear", "n5": "linear", "lh": "nearest", "rh": "nearest"}
#     interp_mode_map = {**default_interp_mode_map, **(interp_mode_map or {})}

#     # --- actions ---
#     all_keys = set()
#     for b in batch:
#         if prefer_action_seq_key in b and isinstance(b[prefer_action_seq_key], dict):
#             all_keys |= set(b[prefer_action_seq_key].keys())
#         if "actions" in b and isinstance(b["actions"], dict):
#             all_keys |= set(b["actions"].keys())

#     actions_out: Dict[str, torch.Tensor] = {}
#     seq_lens_out: Dict[str, torch.Tensor] = {}

#     for k in sorted(all_keys):
#         mode = interp_mode_map.get(k, "linear")
#         seqs_k: List[torch.Tensor] = []
#         lens_k: List[int] = []

#         # find ref dim
#         ref_D: Optional[int] = None
#         for b in batch:
#             q = None
#             if prefer_action_seq_key in b and k in b[prefer_action_seq_key]:
#                 q = b[prefer_action_seq_key][k]
#             elif "actions" in b and k in b["actions"]:
#                 q = b["actions"][k]
#             if isinstance(q, torch.Tensor):
#                 ref_D = int(q.shape[0]) if q.ndim == 1 else int(q.shape[1])
#                 break
#         if ref_D is None:
#             continue

#         for b in batch:
#             if prefer_action_seq_key in b and k in b[prefer_action_seq_key]:
#                 q = b[prefer_action_seq_key][k]
#             elif "actions" in b and k in b["actions"]:
#                 q1 = b["actions"][k]
#                 q = q1.unsqueeze(0) if q1.ndim == 1 else q1
#             else:
#                 q = torch.full((1, ref_D), float("nan"), dtype=torch.float32)

#             if q.ndim == 1:
#                 q = q.unsqueeze(0)
#             if q.ndim != 2:
#                 raise ValueError(f"Bad action tensor for key={k}: shape={tuple(q.shape)}")

#             L = int(q.shape[0])
#             lens_k.append(L)
#             q_rs = _interp_seq_torch(q, target_len=chunk_len, mode=mode)
#             seqs_k.append(q_rs)

#         actions_out[k] = torch.stack(seqs_k, dim=0)  # [B,chunk_len,D]
#         seq_lens_out[k] = torch.tensor(lens_k, dtype=torch.int64)

#     return {
#         "point_cloud": pc_out,
#         "actions": actions_out,
#         "t_frame": t_frame,
#         "meta": meta,
#         "action_seq_lens": seq_lens_out,
#     }
