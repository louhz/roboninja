# similar to the train model and config, but this time


# i need to input the point cloud and only output the prediction value,

# please note that i need to save those prediction value back to left.txt,nova2.txt,nova5.txt,right.txt





# predict_pointcloud_action.py
import os
import re
import sys
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

# ---------------------------------------------------------------------
# Path setup (same pattern as your training script)
# ---------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, PARENT_DIR)

# ---------------------------------------------------------------------
# Imports: model (+ dataset only if you use episodes mode)
# ---------------------------------------------------------------------
try:
    from model.model_raw import FourierFeatures, MultiModalPolicy
except Exception:
    from model_raw import FourierFeatures, MultiModalPolicy

# Optional dataset import (only needed for --episodes_root mode)
_HAS_DATASET = True
try:
    from model.dataloader import PointCloudActionEpisodeDataset
except Exception:
    try:
        from dataloader import PointCloudActionEpisodeDataset
    except Exception:
        _HAS_DATASET = False


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def seed_everything(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_latest_ckpt(save_dir: str) -> Optional[str]:
    if not os.path.isdir(save_dir):
        return None
    candidates = []
    for fn in os.listdir(save_dir):
        m = re.match(r"ckpt_step_(\d+)\.pt$", fn)
        if m:
            candidates.append((int(m.group(1)), os.path.join(save_dir, fn)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def load_point_cloud_file(path: str) -> np.ndarray:
    """
    Loads a point cloud from:
      - .npy / .npz
      - .pt / .pth (Tensor or dict containing point cloud)
      - .txt / .csv (whitespace or comma separated)

    Returns: float32 array of shape [N, F]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Point cloud file not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext in [".npy"]:
        arr = np.load(path)
    elif ext in [".npz"]:
        data = np.load(path)
        if len(data.files) == 0:
            raise ValueError(f"Empty .npz file: {path}")
        arr = data[data.files[0]]
    elif ext in [".pt", ".pth"]:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            arr = obj.detach().cpu().numpy()
        elif isinstance(obj, dict):
            # common keys people use
            for k in ["point_cloud", "pc", "points", "xyz"]:
                if k in obj:
                    v = obj[k]
                    if isinstance(v, torch.Tensor):
                        arr = v.detach().cpu().numpy()
                    else:
                        arr = np.asarray(v)
                    break
            else:
                raise ValueError(
                    f"Unsupported dict format in {path}. Expected a key like "
                    f"'point_cloud'/'pc'/'points'/'xyz'. Keys: {list(obj.keys())}"
                )
        else:
            raise ValueError(f"Unsupported torch object type in {path}: {type(obj)}")
    else:
        # txt/csv: try comma, then whitespace
        try:
            arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
        except Exception:
            arr = np.loadtxt(path, dtype=np.float32)

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Point cloud must be 2D [N,F]. Got shape {arr.shape} from {path}")
    if arr.shape[0] < 1:
        raise ValueError(f"Point cloud has 0 points: {path}")
    return arr




_MESH_EXTS = {
    ".stl", ".obj", ".ply", ".off", ".dae", ".glb", ".gltf", ".3ds",
}
_PC_EXTS = {".npy", ".npz", ".pt", ".pth", ".txt", ".csv"}


def _pad_or_crop_np(pc: np.ndarray, expected_dim: int) -> np.ndarray:
    pc = np.asarray(pc, dtype=np.float32)
    if pc.ndim != 2:
        raise ValueError(f"Point cloud must be 2D [N,F], got {pc.shape}")
    n, f = pc.shape
    if f == expected_dim:
        return pc
    if f > expected_dim:
        return pc[:, :expected_dim]
    # pad
    pad = np.zeros((n, expected_dim - f), dtype=np.float32)
    return np.concatenate([pc, pad], axis=1)


def sample_point_cloud_from_mesh_file(
    mesh_path: str,
    *,
    meta: Dict[str, Any],
    seed: int,
) -> np.ndarray:
    """
    Loads a mesh (or PLY point cloud) and returns a point cloud array [N,F].
    Uses meta fields when available:
      - mesh_sampling: "surface" (default)
      - mesh_points: int (default 8192)
      - features: e.g. ("xyz",) or ("xyz","normal")
      - point_dim: expected feature dim
    """
    try:
        import trimesh
    except Exception as e:
        raise RuntimeError(
            "Mesh input detected but trimesh is not available. Install with:\n"
            "  pip install trimesh"
        ) from e

    if not os.path.exists(mesh_path):
        raise FileNotFoundError(mesh_path)

    # Deterministic mesh sampling (trimesh uses numpy global RNG)
    old_state = np.random.get_state()
    np.random.seed(int(seed))
    try:
        obj = trimesh.load(mesh_path, force=None, process=False)
    finally:
        np.random.set_state(old_state)

    # If scene, merge geometries
    if isinstance(obj, trimesh.Scene):
        if not obj.geometry:
            raise ValueError(f"Empty mesh scene: {mesh_path}")
        # concatenate all Trimesh geometries
        meshes = []
        pclouds = []
        for g in obj.geometry.values():
            if isinstance(g, trimesh.Trimesh):
                meshes.append(g)
            elif isinstance(g, trimesh.points.PointCloud):
                pclouds.append(g)

        if meshes:
            mesh = trimesh.util.concatenate(meshes)
        elif pclouds:
            # fallback: treat as point cloud
            pts = np.asarray(pclouds[0].vertices, dtype=np.float32)
            return pts
        else:
            raise ValueError(f"Unsupported scene geometry types in: {mesh_path}")

    elif isinstance(obj, trimesh.Trimesh):
        mesh = obj

    elif isinstance(obj, trimesh.points.PointCloud):
        # mesh file might actually contain points
        pts = np.asarray(obj.vertices, dtype=np.float32)
        return pts

    else:
        raise ValueError(f"Unsupported mesh object type from trimesh.load: {type(obj)}")

    if mesh.is_empty:
        raise ValueError(f"Loaded mesh is empty: {mesh_path}")

    mesh_sampling = str(meta.get("mesh_sampling", "surface")).lower()
    mesh_points = int(meta.get("mesh_points", 8192))
    features = meta.get("features", ("xyz",))
    if isinstance(features, str):
        features = (features,)
    features = tuple(str(x).lower() for x in features)

    # Sample points
    old_state = np.random.get_state()
    np.random.seed(int(seed))
    try:
        if mesh_sampling == "surface":
            # returns (points, face_index)
            pts, face_idx = trimesh.sample.sample_surface(mesh, mesh_points)
            pts = np.asarray(pts, dtype=np.float32)
            face_idx = np.asarray(face_idx, dtype=np.int64)
            face_normals = None
            if face_idx.size > 0 and hasattr(mesh, "face_normals") and mesh.face_normals is not None:
                face_normals = np.asarray(mesh.face_normals[face_idx], dtype=np.float32)
        else:
            # fallback: trimesh's generic sampler (usually surface)
            pts = np.asarray(mesh.sample(mesh_points), dtype=np.float32)
            face_normals = None
    finally:
        np.random.set_state(old_state)

    # Build feature matrix with xyz first (model assumes xyz in first 3 dims)
    cols: List[np.ndarray] = []
    cols.append(pts[:, :3].astype(np.float32))  # xyz always first

    # Add other requested features if possible
    for feat in features:
        if feat in ("xyz", "pos", "position"):
            # already added as first columns
            continue
        elif feat in ("normal", "normals", "n"):
            if face_normals is None:
                # if we can't compute normals, fill zeros
                cols.append(np.zeros_like(pts[:, :3], dtype=np.float32))
            else:
                cols.append(face_normals[:, :3].astype(np.float32))
        else:
            # unknown feature -> fill zeros (keeps shape consistent vs training)
            # We don't know expected per-feature dim; pad later to meta["point_dim"]
            pass

    pc = np.concatenate(cols, axis=1).astype(np.float32)

    # Match expected point_dim
    expected_dim = int(meta.get("point_dim", pc.shape[1]))
    pc = _pad_or_crop_np(pc, expected_dim)
    return pc


def load_point_cloud_or_mesh_file(
    path: str,
    *,
    meta: Dict[str, Any],
    seed: int,
) -> np.ndarray:
    """
    Auto-detect input type:
      - If extension indicates mesh -> sample from mesh
      - Else try point cloud loader first; if it fails, try mesh loader
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in _MESH_EXTS:
        return sample_point_cloud_from_mesh_file(path, meta=meta, seed=seed)

    if ext in _PC_EXTS:
        return load_point_cloud_file(path)

    # Unknown extension: try PC first, then mesh
    try:
        return load_point_cloud_file(path)
    except Exception as pc_err:
        try:
            return sample_point_cloud_from_mesh_file(path, meta=meta, seed=seed)
        except Exception as mesh_err:
            raise ValueError(
                f"Failed to load input as point cloud OR mesh.\n"
                f"- Point cloud error: {pc_err}\n"
                f"- Mesh error: {mesh_err}\n"
                f"Input: {path}"
            )



def sample_or_pad_points(
    pc: torch.Tensor,
    *,
    num_points: Optional[int],
    sampling: str,
    seed: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    pc: [N, F]
    Returns:
      pc_out: [1, Nout, F]
      mask:   [1, Nout] bool or None
    """
    if pc.dim() != 2:
        raise ValueError(f"pc must be [N,F], got {tuple(pc.shape)}")

    N, F = int(pc.shape[0]), int(pc.shape[1])

    if num_points is None:
        # no padding/sampling
        pc_out = pc.unsqueeze(0)
        return pc_out, None

    M = int(num_points)
    if M <= 0:
        raise ValueError(f"num_points must be >0 or None. Got {num_points}")

    # deterministic sampling
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    if N >= M:
        if sampling == "random":
            idx = torch.randperm(N, generator=g)[:M]
        elif sampling == "first":
            idx = torch.arange(M)
        else:
            raise ValueError("sampling must be 'random' or 'first'")
        pc_sel = pc[idx]
        mask = torch.ones((M,), dtype=torch.bool)
        return pc_sel.unsqueeze(0), mask.unsqueeze(0)

    # N < M: pad with zeros
    pc_out = torch.zeros((M, F), dtype=pc.dtype)
    pc_out[:N] = pc
    mask = torch.zeros((M,), dtype=torch.bool)
    mask[:N] = True
    return pc_out.unsqueeze(0), mask.unsqueeze(0)


def build_model_from_ckpt_meta(
    ckpt_meta: Dict[str, Any],
    *,
    latent_dim: int,
    trunk_dropout: float,
    device: torch.device,
) -> torch.nn.Module:
    head_names = ckpt_meta["head_names"]
    head_dims_flat = ckpt_meta["head_dims_flat"]
    joint_dim = int(ckpt_meta.get("joint_dim", 1))
    point_dim = int(ckpt_meta["point_dim"])

    # Fourier feature settings mirror your training script
    joint_pe = FourierFeatures(joint_dim, num_bands=6, max_freq=20.0)
    point_pe = FourierFeatures(3, num_bands=6, max_freq=10.0)

    try:
        model = MultiModalPolicy(
            joint_dim=joint_dim,
            point_dim=point_dim,
            latent_dim=int(latent_dim),
            joint_pe=joint_pe,
            point_xyz_pe=point_pe,
            trunk_dropout=float(trunk_dropout),
            head_dims=list(head_dims_flat),
            head_names=list(head_names),
            ignore_joint=True,
        ).to(device)
    except TypeError:
        # fallback for older versions
        model = MultiModalPolicy(
            joint_dim=joint_dim,
            point_dim=point_dim,
            latent_dim=int(latent_dim),
            joint_pe=joint_pe,
            point_xyz_pe=point_pe,
            trunk_dropout=float(trunk_dropout),
            head_dims=list(head_dims_flat),
            head_names=list(head_names),
        ).to(device)

    return model


def head_to_filename(head: str) -> str:
    # Required mapping per your request (support common aliases too)
    mapping = {
        # left
        "lh": "left.txt",
        "left": "left.txt",
        "l": "left.txt",

        # right
        "rh": "right.txt",
        "right": "right.txt",
        "r": "right.txt",

        # nova2
        "n2": "nova2.txt",
        "nova2": "nova2.txt",

        # nova5
        "n5": "nova5.txt",
        "nova5": "nova5.txt",
    }
    return mapping.get(str(head).lower(), f"{head}.txt")


def save_pred_txt(
    out_dir: str,
    head: str,
    pred_flat: np.ndarray,
    *,
    action_len: int,
    d_step: int,
    float_fmt: str = "%.8f",
):
    os.makedirs(out_dir, exist_ok=True)
    T = int(action_len)
    D = int(d_step)

    if pred_flat.size != T * D:
        raise ValueError(
            f"Pred size mismatch for head={head}: got {pred_flat.size}, expected {T*D}"
        )

    pred_td = pred_flat.reshape(T, D)

    out_path = os.path.join(out_dir, head_to_filename(head))

    # If D==1, write one value per line; if D>1, write space-separated columns per line
    if D == 1:
        np.savetxt(out_path, pred_td.reshape(-1), fmt=float_fmt)
    else:
        np.savetxt(out_path, pred_td, fmt=float_fmt)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass
class InferConfig:
    # checkpoint
    ckpt_path: Optional[str] = None
    ckpt_dir: str = "./checkpoints"

    # input (choose one)
    point_cloud_path: Optional[str] = None
    episodes_root: Optional[str] = None  # optional batch mode over episodes_*

    # point processing (match training defaults)
    num_points: Optional[int] = 4096
    sampling: str = "random"  # "random" | "first"

    # model build knobs (must match training)
    latent_dim: int = 512
    trunk_dropout: float = 0.1
    device: str = "cuda"
    seed: int = 0

    # output
    out_dir: str = "."
    float_fmt: str = "%.8f"
    print_to_stdout: bool = False

    # episodes mode writing
    # - reads point clouds via your dataset pipeline
    # - writes predicted control files into each episode directory
    mesh_subdir: str = "mesh"
    mesh_name: str = "strawberry.stl"
    control_subdir: str = "control"
    write_control_subdir: Optional[str] = None  # None -> overwrite control_subdir


# ---------------------------------------------------------------------
# Main inference
# ---------------------------------------------------------------------
@torch.no_grad()
def run_single_point_cloud(cfg: InferConfig):
    seed_everything(cfg.seed)

    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")

    ckpt_path = cfg.ckpt_path or find_latest_ckpt(cfg.ckpt_dir)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No checkpoint found. Provide --ckpt_path or put ckpt_step_*.pt under {cfg.ckpt_dir}"
        )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    meta = ckpt["meta"]

    model = build_model_from_ckpt_meta(
        meta,
        latent_dim=cfg.latent_dim,
        trunk_dropout=cfg.trunk_dropout,
        device=device,
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # load point cloud
    pc_np = load_point_cloud_or_mesh_file(cfg.point_cloud_path, meta=meta, seed=cfg.seed)

    pc_t = torch.from_numpy(pc_np).to(dtype=torch.float32)


    point_dim_expected = int(meta["point_dim"])
    F_in = int(pc_t.shape[1])
    if F_in < point_dim_expected:
        pad = torch.zeros((pc_t.shape[0], point_dim_expected - F_in), dtype=pc_t.dtype)
        pc_t = torch.cat([pc_t, pad], dim=1)
    elif F_in > point_dim_expected:
        pc_t = pc_t[:, :point_dim_expected]

    pc_b, mask_b = sample_or_pad_points(
        pc_t,
        num_points=cfg.num_points,
        sampling=cfg.sampling,
        seed=cfg.seed,
    )

    pc_b = pc_b.to(device, non_blocking=True)
    mask_b = mask_b.to(device, non_blocking=True) if mask_b is not None else None

    joint_dim = int(meta.get("joint_dim", 1))
    joint = torch.zeros((1, joint_dim), device=device, dtype=torch.float32)

    preds: Dict[str, torch.Tensor] = model(joint=joint, point_cloud=pc_b, point_mask=mask_b)

    # save predictions to left.txt / nova2.txt / nova5.txt / right.txt
    head_names: List[str] = list(meta["head_names"])
    head_dims_step: List[int] = list(meta["head_dims_step"])
    action_len = int(meta["action_len"])

    for hn, d_step in zip(head_names, head_dims_step):
        pred_flat = preds[hn][0].detach().cpu().numpy().astype(np.float32)
        save_pred_txt(
            cfg.out_dir,
            hn,
            pred_flat,
            action_len=action_len,
            d_step=int(d_step),
            float_fmt=cfg.float_fmt,
        )

        if cfg.print_to_stdout:
            # Print ONLY numeric values (no extra logs)
            # D==1 prints T lines; D>1 prints T rows
            pred_td = pred_flat.reshape(action_len, int(d_step))
            if int(d_step) == 1:
                for v in pred_td.reshape(-1):
                    print(float(v))
            else:
                for row in pred_td:
                    print(" ".join(str(float(x)) for x in row))


@torch.no_grad()
def run_episodes_root(cfg: InferConfig):
    if not _HAS_DATASET:
        raise RuntimeError(
            "PointCloudActionEpisodeDataset import failed. "
            "Episodes mode requires your dataset module to be importable."
        )

    seed_everything(cfg.seed)
    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")

    ckpt_path = cfg.ckpt_path or find_latest_ckpt(cfg.ckpt_dir)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No checkpoint found. Provide --ckpt_path or put ckpt_step_*.pt under {cfg.ckpt_dir}"
        )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    meta = ckpt["meta"]

    model = build_model_from_ckpt_meta(
        meta,
        latent_dim=cfg.latent_dim,
        trunk_dropout=cfg.trunk_dropout,
        device=device,
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # Build dataset in the same way as training (reads mesh->point cloud, etc.)
    ds = PointCloudActionEpisodeDataset(
        obs_root=cfg.episodes_root,
        act_root=None,
        mesh_subdir=cfg.mesh_subdir,
        mesh_name=cfg.mesh_name,
        control_subdir=cfg.control_subdir,
        mesh_sampling=meta.get("mesh_sampling", "surface"),
        mesh_points=int(meta.get("mesh_points", 8192)),
        groups=tuple(meta.get("groups", meta["head_names"])),
        missing_policy=meta.get("missing_policy", "initial"),
        action_len=int(meta["action_len"]),
        action_format=meta.get("action_format", "dict"),
        interp_mode_map=None,
        num_points=cfg.num_points,
        sampling=cfg.sampling,
        features=tuple(meta.get("features", ("xyz",))),
        seed=cfg.seed,
        cache_point_cloud=True,
        pc_cache_max_items=0,
    )

    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    head_names: List[str] = list(meta["head_names"])
    head_dims_step: List[int] = list(meta["head_dims_step"])
    action_len = int(meta["action_len"])
    write_subdir = cfg.write_control_subdir or cfg.control_subdir  # default overwrite

    for batch in loader:
        pc_raw = batch["point_cloud"]
        # dataset may return Tensor [B,N,F] or list; handle both
        if isinstance(pc_raw, list):
            pc = pc_raw[0].unsqueeze(0)
            mask = torch.ones((1, pc.shape[1]), dtype=torch.bool)
        else:
            pc = pc_raw
            mask = None

        pc = pc.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True) if mask is not None else None

        joint_dim = int(meta.get("joint_dim", 1))
        joint = torch.zeros((1, joint_dim), device=device, dtype=torch.float32)

        preds: Dict[str, torch.Tensor] = model(joint=joint, point_cloud=pc, point_mask=mask)

        # resolve episode output directory
        out_dir = None
        meta_list = batch.get("meta", None)
        if isinstance(meta_list, list) and len(meta_list) > 0 and isinstance(meta_list[0], dict):
            m0 = meta_list[0]
            for k in ["episode_dir", "episode_path", "episode_root", "path"]:
                if k in m0 and isinstance(m0[k], str) and os.path.isdir(m0[k]):
                    out_dir = os.path.join(m0[k], write_subdir)
                    break
            if out_dir is None and "episode_id" in m0:
                # best-effort fallback
                cand = os.path.join(cfg.episodes_root, str(m0["episode_id"]))
                if os.path.isdir(cand):
                    out_dir = os.path.join(cand, write_subdir)

        if out_dir is None:
            # last-resort fallback
            out_dir = os.path.join(cfg.out_dir, write_subdir)

        for hn, d_step in zip(head_names, head_dims_step):
            pred_flat = preds[hn][0].detach().cpu().numpy().astype(np.float32)
            save_pred_txt(
                out_dir,
                hn,
                pred_flat,
                action_len=action_len,
                d_step=int(d_step),
                float_fmt=cfg.float_fmt,
            )


def parse_args() -> InferConfig:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", type=str, default=None, help="Path to ckpt_step_*.pt")
    ap.add_argument("--ckpt_dir", type=str, default="./checkpoints", help="Directory to auto-pick latest ckpt")
    ap.add_argument(
        "--point_cloud",
        type=str,
        default=None,
        help="Input point cloud OR mesh file (.txt/.csv/.npy/.npz/.pt OR .stl/.obj/.ply/...)",
    )
    ap.add_argument("--episodes_root", type=str, default=None, help="Run inference over episodes_* dataset root")

    ap.add_argument("--out_dir", type=str, default=".", help="Output directory (single PC mode) or fallback dir")
    ap.add_argument("--num_points", type=int, default=4096, help="Downsample/pad to this many points (set -1 to disable)")
    ap.add_argument("--sampling", type=str, default="random", choices=["random", "first"])
    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    ap.add_argument("--seed", type=int, default=0)

    # must match training
    ap.add_argument("--latent_dim", type=int, default=512)
    ap.add_argument("--trunk_dropout", type=float, default=0.1)

    ap.add_argument("--float_fmt", type=str, default="%.8f")
    ap.add_argument("--print", dest="print_to_stdout", action="store_true", help="Print ONLY numeric predictions to stdout")

    # episodes mode layout
    ap.add_argument("--mesh_subdir", type=str, default="mesh")
    ap.add_argument("--mesh_name", type=str, default="strawberry.stl")
    ap.add_argument("--control_subdir", type=str, default="control")
    ap.add_argument(
        "--write_control_subdir",
        type=str,
        default=None,
        help="Where to write outputs inside each episode. Default overwrites --control_subdir.",
    )

    args = ap.parse_args()

    num_points = args.num_points
    if num_points is not None and int(num_points) < 0:
        num_points = None

    return InferConfig(
        ckpt_path=args.ckpt_path,
        ckpt_dir=args.ckpt_dir,
        point_cloud_path=args.point_cloud,
        episodes_root=args.episodes_root,
        out_dir=args.out_dir,
        num_points=num_points,
        sampling=args.sampling,
        device=args.device,
        seed=args.seed,
        latent_dim=args.latent_dim,
        trunk_dropout=args.trunk_dropout,
        float_fmt=args.float_fmt,
        print_to_stdout=bool(args.print_to_stdout),
        mesh_subdir=args.mesh_subdir,
        mesh_name=args.mesh_name,
        control_subdir=args.control_subdir,
        write_control_subdir=args.write_control_subdir,
    )


def main():
    cfg = parse_args()

    if cfg.point_cloud_path:
        run_single_point_cloud(cfg)
        return

    if cfg.episodes_root:
        run_episodes_root(cfg)
        return

    raise SystemExit("You must provide either --point_cloud or --episodes_root")


if __name__ == "__main__":
    main()



