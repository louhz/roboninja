# =============================================================================
# Right-hand MULTI-MODAL dataset: point cloud + tactile (+ rh actions)
# Paste this BELOW the point-cloud dataset code I sent earlier.
#
# Requires the earlier module to define:
#   - RightHandPointCloudActionDataset
#   - _timestamp_from_name   (or similar filename timestamp helper)
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from learning.model.dataloader import RightHandPointCloudActionDataset

# -----------------------------------------------------------------------------
# TACTILE load/save utilities (adapted from your pasted thermal utilities)
# -----------------------------------------------------------------------------

def _srgb_to_linear(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    a = 0.055
    return np.where(arr <= 0.04045, arr / 12.92, ((arr + a) / (1 + a)) ** 2.4)


def _jet_lut(n: int = 256) -> np.ndarray:
    x = np.linspace(0.0, 1.0, n, dtype=np.float32)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0.0, 1.0)
    return np.stack([r, g, b], axis=1)


def _map_rgb_via_lut(
    arr_rgb: np.ndarray,
    lut: np.ndarray,
    assume_srgb: bool = True,
    chunk: int = 200_000,
) -> np.ndarray:
    """
    Invert a colormap by nearest-neighbor in RGB space.
    Returns scalar map in [0,1], shape [H,W].
    """
    H, W = arr_rgb.shape[:2]

    # Normalize to [0,1]
    if arr_rgb.dtype == np.uint8:
        rgb = arr_rgb.astype(np.float32) / 255.0
    elif arr_rgb.dtype == np.uint16:
        rgb = arr_rgb.astype(np.float32) / 65535.0
    else:
        rgb = arr_rgb.astype(np.float32, copy=False)
        mx, mn = float(rgb.max()), float(rgb.min())
        if mx > 1.0 or mn < 0.0:
            rng = mx - mn if mx > mn else 1.0
            rgb = (rgb - mn) / rng

    if assume_srgb:
        rgb = _srgb_to_linear(rgb)

    flat = rgb.reshape(-1, 3)               # [P,3]
    lut = lut.astype(np.float32, copy=False)  # [N,3]
    P = flat.shape[0]
    N = lut.shape[0]

    out = np.empty(P, dtype=np.float32)
    for i in range(0, P, chunk):
        block = flat[i : i + chunk]  # [B,3]
        d2 = ((block[:, None, :] - lut[None, :, :]) ** 2).sum(axis=2)  # [B,N]
        idx = np.argmin(d2, axis=1).astype(np.float32)                 # [B]
        out[i : i + chunk] = idx / (N - 1.0)

    return out.reshape(H, W)


def _rgb_pseudothermal_to_scalar(
    arr_rgb: np.ndarray,
    method: str = "rb_ratio",     # "rb_ratio" | "nearest"
    colormap: str = "jet",
    custom_lut: Optional[np.ndarray] = None,
    assume_srgb: bool = True,
    nearest_chunk: int = 200_000,
) -> np.ndarray:
    """
    Convert pseudo-colored tactile/thermal RGB (red=hot/high, blue=cold/low) to scalar [0,1].
    """
    assert arr_rgb.ndim == 3 and arr_rgb.shape[-1] >= 3, "Expected RGB image array."

    if method == "rb_ratio":
        # Normalize to [0,1]
        if arr_rgb.dtype == np.uint8:
            rgb = arr_rgb.astype(np.float32) / 255.0
        elif arr_rgb.dtype == np.uint16:
            rgb = arr_rgb.astype(np.float32) / 65535.0
        else:
            rgb = arr_rgb.astype(np.float32, copy=False)
            mx, mn = float(rgb.max()), float(rgb.min())
            if mx > 1.0 or mn < 0.0:
                rng = mx - mn if mx > mn else 1.0
                rgb = (rgb - mn) / rng

        if assume_srgb:
            rgb = _srgb_to_linear(rgb)

        R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        eps = 1e-6

        rb = R / (R + B + eps)
        g_chroma = G / (R + G + B + eps)
        hotness = rb + 0.10 * (g_chroma - 1.0 / 3.0)
        return np.clip(hotness, 0.0, 1.0).astype(np.float32)

    if method == "nearest":
        lut = custom_lut
        if lut is None:
            lut = _jet_lut(256) if colormap.lower() == "jet" else _jet_lut(256)
        return _map_rgb_via_lut(arr_rgb, lut, assume_srgb=assume_srgb, chunk=nearest_chunk).astype(np.float32)

    raise ValueError("method must be one of {'rb_ratio','nearest'}")


def _load_tactile_array(path: Path, keep_channels: bool = True) -> np.ndarray:
    """
    Loads a tactile frame into a numpy array.
    Supports:
      - .npy: HxW or HxWx{1,3}
      - .npz: 'tactile' or 'thermal' or first array
      - image files: PNG/TIFF/JPG etc
    """
    from PIL import Image  # local import

    ext = path.suffix.lower()

    if ext == ".npy":
        arr = np.load(path)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        return arr.astype(np.float64, copy=False)

    if ext == ".npz":
        z = np.load(path)
        if "tactile" in z:
            arr = z["tactile"]
        elif "thermal" in z:
            arr = z["thermal"]
        else:
            arr = z[list(z.keys())[0]]
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        return arr.astype(np.float64, copy=False)

    # Image files
    with Image.open(path) as im:
        arr = np.array(im)

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if not keep_channels and arr.ndim == 3:
        arr = arr.mean(axis=-1)

    return arr.astype(np.float64, copy=False)


def _normalize_tactile(
    arr: np.ndarray,
    mode: str = "percentile",     # "percentile", "minmax", "fixed", "none"
    p_low: float = 2.0,
    p_high: float = 98.0,
    fixed_range: Optional[Tuple[float, float]] = None,
    global_range: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Normalize scalar tactile to [0,1].
    Returns (scaled, info_dict with lo/hi).
    """
    info: Dict[str, Any] = {}
    a = arr

    if global_range is not None:
        lo, hi = global_range
    elif mode == "percentile":
        lo = float(np.percentile(a, p_low))
        hi = float(np.percentile(a, p_high))
    elif mode == "minmax":
        lo = float(np.min(a))
        hi = float(np.max(a))
    elif mode == "fixed":
        if fixed_range is None:
            raise ValueError("fixed_range must be provided when mode='fixed'")
        lo, hi = fixed_range
    elif mode == "none":
        # Heuristic by dtype-ish range
        lo, hi = float(np.min(a)), float(np.max(a))
    else:
        raise ValueError("mode must be one of {'percentile','minmax','fixed','none'}")

    if hi <= lo:
        hi = lo + 1.0

    scaled = (a - lo) / (hi - lo)
    scaled = np.clip(scaled, 0.0, 1.0).astype(np.float32, copy=False)
    info.update({"lo": lo, "hi": hi})
    return scaled, info


def load_tactile_tensor(
    path: Path,
    channels: int = 1,                 # 1 or 3
    rgb_pseudo: bool = True,           # treat RGB as pseudo-colored tactile/thermal
    rgb_method: str = "rb_ratio",      # "rb_ratio" or "nearest"
    colormap: str = "jet",
    custom_lut: Optional[np.ndarray] = None,
    assume_srgb: bool = True,
    nearest_chunk: int = 200_000,
    norm_mode: str = "percentile",
    p_low: float = 2.0,
    p_high: float = 98.0,
    fixed_range: Optional[Tuple[float, float]] = None,
    global_range: Optional[Tuple[float, float]] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    HOOK: If you have your own tactile script, replace THIS function body to call it.

    Returns:
      tactile_tensor: FloatTensor [C,H,W] in 0..1
      info: dict with lo/hi (normalization stats)
    """
    raw = _load_tactile_array(path, keep_channels=True)

    # Convert to scalar map
    if raw.ndim == 2:
        scalar = raw.astype(np.float32, copy=False)
    elif raw.ndim == 3 and raw.shape[-1] >= 3:
        if rgb_pseudo:
            scalar = _rgb_pseudothermal_to_scalar(
                raw,
                method=rgb_method,
                colormap=colormap,
                custom_lut=custom_lut,
                assume_srgb=assume_srgb,
                nearest_chunk=nearest_chunk,
            ).astype(np.float32, copy=False)
        else:
            scalar = raw[..., :3].mean(axis=-1).astype(np.float32, copy=False)
    elif raw.ndim == 3 and raw.shape[-1] == 1:
        scalar = raw[..., 0].astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unsupported tactile array shape: {raw.shape} for {path}")

    scaled, info = _normalize_tactile(
        scalar,
        mode=norm_mode,
        p_low=p_low,
        p_high=p_high,
        fixed_range=fixed_range,
        global_range=global_range,
    )

    if channels == 1:
        t = torch.from_numpy(scaled)[None, ...]  # [1,H,W]
    elif channels == 3:
        t = torch.from_numpy(np.stack([scaled, scaled, scaled], axis=0))  # [3,H,W]
    else:
        raise ValueError("channels must be 1 or 3")

    return t.float(), info


def save_tactile(
    path: Union[str, Path],
    tactile: np.ndarray,
    key: str = "tactile",
) -> None:
    """
    HOOK: If you have your own tactile save script, replace THIS function body to call it.
    """
    from PIL import Image  # local import

    path = Path(path)
    tactile = np.asarray(tactile)

    ext = path.suffix.lower()
    if ext == ".npy":
        np.save(path, tactile)
        return
    if ext == ".npz":
        np.savez_compressed(path, **{key: tactile})
        return

    # image save fallback: assumes tactile is [H,W] in 0..1 or 0..255
    t = tactile.astype(np.float32)
    if t.ndim == 3 and t.shape[0] in (1, 3):
        # [C,H,W] -> [H,W] if 1ch, else take mean
        if t.shape[0] == 1:
            t = t[0]
        else:
            t = t.mean(axis=0)

    if t.max() <= 1.5:
        t = np.clip(t, 0.0, 1.0) * 255.0
    t8 = np.clip(t, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(t8).save(path)


# -----------------------------------------------------------------------------
# Tactile frame listing + alignment helpers
# -----------------------------------------------------------------------------

def _list_tactile_sorted(root: Path, tactile_glob: str) -> List[Path]:
    """
    Lists tactile frames. Supports images and arrays.
    """
    paths: List[Path] = []
    if tactile_glob:
        paths.extend(sorted(root.glob(tactile_glob)))

    if not paths:
        for ext in [
            "**/*.png", "**/*.jpg", "**/*.jpeg", "**/*.tif", "**/*.tiff",
            "**/*.npy", "**/*.npz"
        ]:
            paths.extend(sorted(root.glob(ext)))

    return paths


def _build_timebase_from_names_or_fps(
    paths: List[Path],
    default_fps: float,
    timestamp_fn,
) -> Tuple[np.ndarray, bool]:
    """
    Returns (t, have_real_timestamps).
    - If filename timestamps exist for all frames: uses them and returns True.
    - Else: returns a fixed fps timebase and returns False.
    """
    ts: List[float] = []
    ok = True
    for p in paths:
        t = timestamp_fn(p)
        if t is None:
            ok = False
            break
        ts.append(float(t))

    if ok and ts:
        arr = np.asarray(ts, dtype=np.float64)
        arr -= arr[0]
        return arr, True

    dt = 1.0 / default_fps if default_fps > 0 else 1.0 / 30.0
    return np.arange(len(paths), dtype=np.float64) * dt, False


def _nearest_index_map(t_query: np.ndarray, t_src: np.ndarray) -> np.ndarray:
    """
    For each t_query, find nearest index in t_src (both sorted).
    """
    t_query = np.asarray(t_query, dtype=np.float64).reshape(-1)
    t_src = np.asarray(t_src, dtype=np.float64).reshape(-1)

    if len(t_src) == 0:
        return np.full((len(t_query),), -1, dtype=np.int64)

    idx = np.searchsorted(t_src, t_query, side="left")
    idx = np.clip(idx, 0, len(t_src) - 1)
    prev = np.clip(idx - 1, 0, len(t_src) - 1)
    use_prev = (idx > 0) & (np.abs(t_query - t_src[prev]) <= np.abs(t_src[idx] - t_query))
    idx = np.where(use_prev, prev, idx)
    return idx.astype(np.int64)


# -----------------------------------------------------------------------------
# Collate: episode batch with tactile
# -----------------------------------------------------------------------------

def collate_episode_pointcloud_tactile(frames: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Episode-collate for multimodal frames (same-episode requirement, like your original).
    Returns:
      {
        "point_cloud": Tensor [T,N,F] OR list[Tensor [Ni,F]],
        "tactile":     Tensor [T,C,H,W] OR list[Tensor [C,H,W]],
        "actions":     {k: Tensor [T,Dk]},
        "t_frame":     Tensor [T],
        "meta":        dict(...)
      }
    """
    if len(frames) == 0:
        raise ValueError("Empty batch for collate_episode_pointcloud_tactile")

    ep_id = frames[0]["meta"]["episode_id"]
    ep_idx = frames[0]["meta"]["episode_idx"]
    for f in frames[1:]:
        if f["meta"]["episode_id"] != ep_id:
            raise ValueError("collate_episode_pointcloud_tactile got frames from multiple episodes.")

    # Point clouds
    pcs = [f["point_cloud"] for f in frames]
    if isinstance(pcs[0], torch.Tensor):
        try:
            point_cloud = torch.stack(pcs, dim=0)
        except RuntimeError:
            point_cloud = pcs
    else:
        point_cloud = pcs

    # Tactile
    tacs = [f["tactile"] for f in frames]
    if isinstance(tacs[0], torch.Tensor):
        try:
            tactile = torch.stack(tacs, dim=0)
        except RuntimeError:
            tactile = tacs
    else:
        tactile = tacs

    # Actions
    all_keys = set().union(*(f["actions"].keys() for f in frames))
    actions: Dict[str, torch.Tensor] = {}
    for k in sorted(all_keys):
        mats = [f["actions"][k] for f in frames if k in f["actions"]]
        if len(mats) != len(frames):
            D = mats[0].shape[-1]
            seq = []
            for f in frames:
                if k in f["actions"]:
                    seq.append(f["actions"][k])
                else:
                    seq.append(torch.full((D,), float("nan"), dtype=torch.float32))
            actions[k] = torch.stack(seq, dim=0)
        else:
            actions[k] = torch.stack(mats, dim=0)

    t_frame = torch.tensor([f["t_frame"] for f in frames], dtype=torch.float32)

    meta = {
        "episode_id": ep_id,
        "episode_idx": int(ep_idx),
        "num_frames": len(frames),
        "frame_indices": [int(f["meta"]["frame_idx"]) for f in frames],
        "frame_paths": [f["meta"]["frame_path"] for f in frames],
        "tactile_paths": [f["meta"].get("tactile_path", None) for f in frames],
        "act_dir": frames[0]["meta"]["act_dir"],
        "obs_dir": frames[0]["meta"]["obs_dir"],
        "tactile_dir": frames[0]["meta"].get("tactile_dir", None),
    }

    return {"point_cloud": point_cloud, "tactile": tactile, "actions": actions, "t_frame": t_frame, "meta": meta}


# -----------------------------------------------------------------------------
# New dataset: RH point cloud + tactile (+ rh actions)
# -----------------------------------------------------------------------------

@dataclass
class _TactileEpisodeIndex:
    tactile_dir: Path
    tactile_paths: List[Path]
    t_tactile: np.ndarray
    idx_map: np.ndarray              # len = T_pc, maps pc frame index -> tactile frame index
    have_ts: bool


class RightHandPointCloudTactileActionDataset(torch.utils.data.Dataset):
    """
    Right-hand multimodal dataset:
      - point cloud (Open3D, from your point cloud dataset)
      - tactile frame (loaded using the tactile loader above)
      - actions: {'rh': Tensor[10]}

    It wraps the already-working RightHandPointCloudActionDataset (point_cloud + rh actions),
    and *adds tactile* aligned per episode by timestamp (or by index fallback).

    Output per __getitem__:
      {
        "point_cloud": Tensor [N,F],
        "tactile":     Tensor [C,H,W],
        "actions":     {"rh": Tensor[10]},
        "t_frame":     float,
        "meta":        {... plus tactile_path ...}
      }
    """

    def __init__(
        self,
        # point cloud + actions config (passed into the base dataset)
        obs_root: Union[str, Path],
        act_root: Union[str, Path],
        pc_glob: str = "*.ply",
        default_fps: float = 30.0,
        align_mode: str = "nearest",
        num_points: Optional[int] = None,
        sampling: str = "random",
        features: Tuple[str, ...] = ("xyz",),
        resample_if_len_mismatch: bool = True,

        # tactile config
        tactile_root: Optional[Union[str, Path]] = None,
        tactile_subdir: Optional[str] = None,        # if tactile frames live under <episode>/<subdir>/
        tactile_glob: str = "*.png",
        tactile_channels: int = 1,
        tactile_rgb_pseudo: bool = True,
        tactile_rgb_method: str = "rb_ratio",
        tactile_norm_mode: str = "percentile",
        tactile_p_low: float = 2.0,
        tactile_p_high: float = 98.0,
        tactile_fixed_range: Optional[Tuple[float, float]] = None,
        tactile_assume_srgb: bool = True,
        tactile_nearest_chunk: int = 200_000,

        # what to do if tactile missing in some episode
        require_tactile: bool = True,
        missing_tactile_policy: str = "zero",  # "zero" or "nan"
    ):
        super().__init__()

        # --- base dataset: point cloud + RH actions ---
        self.base = RightHandPointCloudActionDataset(
            obs_root=str(obs_root),
            act_root=str(act_root),
            pc_glob=pc_glob,
            default_fps=default_fps,
            align_mode=align_mode,
            num_points=num_points,
            sampling=sampling,
            features=features,
            resample_if_len_mismatch=resample_if_len_mismatch,
        )

        self.default_fps = float(default_fps)

        # tactile configuration
        self.tactile_channels = int(tactile_channels)
        self.tactile_rgb_pseudo = bool(tactile_rgb_pseudo)
        self.tactile_rgb_method = str(tactile_rgb_method)
        self.tactile_norm_mode = str(tactile_norm_mode)
        self.tactile_p_low = float(tactile_p_low)
        self.tactile_p_high = float(tactile_p_high)
        self.tactile_fixed_range = tactile_fixed_range
        self.tactile_assume_srgb = bool(tactile_assume_srgb)
        self.tactile_nearest_chunk = int(tactile_nearest_chunk)

        self.require_tactile = bool(require_tactile)
        self.missing_tactile_policy = str(missing_tactile_policy)

        # tactile root resolution
        self.tactile_root = Path(tactile_root).expanduser() if tactile_root is not None else None
        self.tactile_subdir = tactile_subdir
        self.tactile_glob = tactile_glob

        # We reuse the timestamp helper from the base module:
        # your earlier module had `_timestamp_from_name(p: Path) -> Optional[float]`
        try:
            timestamp_fn = _timestamp_from_name  # type: ignore[name-defined]
        except NameError:
            raise RuntimeError(
                "This add-on expects the earlier point-cloud module to define `_timestamp_from_name(Path)`."
            )

        # --- Build tactile index per episode ---
        self._tactile_ep: List[_TactileEpisodeIndex] = []

        # Pre-infer a fill shape by looking at the first tactile file found (for missing episodes)
        self._fill_hw: Optional[Tuple[int, int]] = None

        for ep in self.base.episodes:
            # Resolve tactile episode directory
            if self.tactile_root is None:
                # tactile frames live alongside point cloud frames by default
                tactile_dir = ep.obs_dir
            else:
                cand = self.tactile_root / ep.episode_id
                tactile_dir = cand if cand.is_dir() else self.tactile_root

            if self.tactile_subdir:
                tactile_dir = tactile_dir / self.tactile_subdir

            tactile_paths = _list_tactile_sorted(tactile_dir, self.tactile_glob)

            if (not tactile_paths) and self.require_tactile:
                raise FileNotFoundError(
                    f"No tactile frames found for episode '{ep.episode_id}' under: {tactile_dir} (glob='{self.tactile_glob}')"
                )

            # Timebases
            t_tactile, tactile_has_ts = _build_timebase_from_names_or_fps(
                tactile_paths, self.default_fps, timestamp_fn=timestamp_fn
            )
            t_pc = ep.t_frame
            pc_has_ts = True  # base always returns a timebase (may be fps-based); we treat that as usable

            # Build mapping from PC frame index -> tactile frame index
            if len(tactile_paths) == 0:
                idx_map = np.full((len(t_pc),), -1, dtype=np.int64)
            else:
                if tactile_has_ts and pc_has_ts:
                    # nearest by time
                    idx_map = _nearest_index_map(t_pc, t_tactile)
                else:
                    # fallback: index-based mapping
                    if len(tactile_paths) == len(t_pc):
                        idx_map = np.arange(len(t_pc), dtype=np.int64)
                    else:
                        idx_map = np.round(
                            np.linspace(0, len(tactile_paths) - 1, num=len(t_pc), endpoint=True)
                        ).astype(np.int64)

            self._tactile_ep.append(_TactileEpisodeIndex(
                tactile_dir=tactile_dir,
                tactile_paths=tactile_paths,
                t_tactile=t_tactile,
                idx_map=idx_map,
                have_ts=bool(tactile_has_ts),
            ))

            # Infer fill H,W
            if self._fill_hw is None and tactile_paths:
                tac0, _info0 = load_tactile_tensor(
                    tactile_paths[0],
                    channels=self.tactile_channels,
                    rgb_pseudo=self.tactile_rgb_pseudo,
                    rgb_method=self.tactile_rgb_method,
                    assume_srgb=self.tactile_assume_srgb,
                    nearest_chunk=self.tactile_nearest_chunk,
                    norm_mode=self.tactile_norm_mode,
                    p_low=self.tactile_p_low,
                    p_high=self.tactile_p_high,
                    fixed_range=self.tactile_fixed_range,
                    global_range=None,
                )
                _, H, W = tac0.shape
                self._fill_hw = (H, W)

        if (not self.require_tactile) and self._fill_hw is None:
            # If tactile missing everywhere, still define something safe
            self._fill_hw = (1, 1)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.base[idx]  # includes point_cloud, actions{'rh'}, t_frame, meta

        # Locate episode/frame index
        ep_idx, frame_i = self.base._global_to_local[idx]  # uses base internal mapping
        te = self._tactile_ep[ep_idx]
        j = int(te.idx_map[frame_i])

        if j < 0 or j >= len(te.tactile_paths):
            # Missing tactile frame: fill tensor
            H, W = self._fill_hw if self._fill_hw is not None else (1, 1)
            if self.missing_tactile_policy == "zero":
                tactile = torch.zeros((self.tactile_channels, H, W), dtype=torch.float32)
            elif self.missing_tactile_policy == "nan":
                tactile = torch.full((self.tactile_channels, H, W), float("nan"), dtype=torch.float32)
            else:
                raise ValueError("missing_tactile_policy must be 'zero' or 'nan'")
            tactile_path = None
            tac_info: Dict[str, Any] = {}
        else:
            tactile_path = te.tactile_paths[j]
            tactile, tac_info = load_tactile_tensor(
                tactile_path,
                channels=self.tactile_channels,
                rgb_pseudo=self.tactile_rgb_pseudo,
                rgb_method=self.tactile_rgb_method,
                assume_srgb=self.tactile_assume_srgb,
                nearest_chunk=self.tactile_nearest_chunk,
                norm_mode=self.tactile_norm_mode,
                p_low=self.tactile_p_low,
                p_high=self.tactile_p_high,
                fixed_range=self.tactile_fixed_range,
                global_range=None,
            )

        # Attach tactile to sample
        sample["tactile"] = tactile

        # Expand meta
        sample["meta"]["tactile_dir"] = str(te.tactile_dir)
        sample["meta"]["tactile_path"] = str(tactile_path) if tactile_path is not None else None
        sample["meta"]["tactile_lo"] = tac_info.get("lo", None)
        sample["meta"]["tactile_hi"] = tac_info.get("hi", None)
        sample["meta"]["tactile_rgb_pseudo"] = self.tactile_rgb_pseudo
        sample["meta"]["tactile_rgb_method"] = self.tactile_rgb_method

        return sample


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    ds = RightHandPointCloudTactileActionDataset(
        obs_root="path/to/pc_obs_root",
        act_root="path/to/act_root",
        pc_glob="*.ply",
        num_points=2048,
        features=("xyz",),

        tactile_root="path/to/tactile_root",   # can be None if tactile is in same episode folder
        tactile_subdir=None,                   # or "tactile" if frames are under <episode>/tactile/
        tactile_glob="*.png",
        tactile_channels=1,
        tactile_rgb_pseudo=True,
        tactile_rgb_method="rb_ratio",
    )

    # IMPORTANT:
    # - If you want *episode sequences*, you need an episode-aware sampler like you had before.
    # - Otherwise (random frames), use a normal collate or keep batch as list.
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0, collate_fn=lambda x: x)

    batch = next(iter(dl))
    print("Example keys:", batch[0].keys())
    print("PC shape:", batch[0]["point_cloud"].shape)   # [N,F]
    print("Tac shape:", batch[0]["tactile"].shape)      # [C,H,W]
    print("RH action:", batch[0]["actions"]["rh"].shape)
