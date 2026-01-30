#!/usr/bin/env python3
"""
Inference for MultiModalPolicy (point-cloud -> multi-head action).

- Loads a checkpoint produced by your train_pointcloud_action_wandb.py
- Samples a point cloud from a given mesh (.stl/.obj/.ply, etc.)
- Runs the model forward (joint=None if ignore_joint=True)
- Saves predictions to .txt files using your required head->filename mapping

Example:
  python infer_pointcloud_action.py \
    --ckpt checkpointnewest2/ckpt_step_2000.pt \
    --mesh /home/louhz/Desktop/Rss/roboninja/example_dataset/sim2real_2/episodes_1/mesh/mesh.stl \
    --out_dir ./pred_txt
"""

import os
import re
import sys
import argparse
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------
# Path setup (same pattern as your training script)
# ---------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, PARENT_DIR)

# ---------------------------------------------------------------------
# Imports: model
# ---------------------------------------------------------------------
try:
    from model.model_raw import FourierFeatures, MultiModalPolicy
except Exception:
    from model_raw import FourierFeatures, MultiModalPolicy


# ---------------------------------------------------------------------
# Your REQUIRED output helpers (verbatim format you asked for)
# ---------------------------------------------------------------------
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
# Utils
# ---------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _infer_latent_dim_from_state_dict(state, head_names):
    for hn in head_names:
        k = f"heads.heads.{hn}.0.weight"
        if k in state and state[k].ndim == 2:
            return int(state[k].shape[1])  # in_features == latent_dim
    return None

def _ckpt_uses_dropout(state):
    # If dropout existed, your trunk linear indices tend to be 0/4/8 (rather than 0/3/6)
    return ("trunk.8.weight" in state) or ("pc_enc.per_point.8.weight" in state)

def _as_bool_tri(x: str) -> Optional[bool]:
    """
    Parses 'auto' -> None, 'true'/'1' -> True, 'false'/'0' -> False
    """
    s = str(x).strip().lower()
    if s in ("auto", "none", ""):
        return None
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f"):
        return False
    raise ValueError(f"Invalid value '{x}'. Use auto/true/false.")


def load_pointcloud_from_mesh(
    mesh_path: str,
    *,
    num_points: int,
    mesh_sampling: str = "surface",  # "surface" | "vertices"
    seed: int = 0,
) -> np.ndarray:
    """
    Returns points as float32 array [N,3] sampled from mesh.
    """
    try:
        import trimesh
    except Exception as e:
        raise RuntimeError(
            "trimesh is required for mesh sampling in this inference script.\n"
            "Install it with: pip install trimesh"
        ) from e

    rng = np.random.default_rng(int(seed))

    m = trimesh.load(mesh_path, force="mesh")
    if isinstance(m, trimesh.Scene):
        # merge all geometry
        geoms = list(m.geometry.values())
        if len(geoms) == 0:
            raise ValueError(f"Loaded a Scene with 0 geometries: {mesh_path}")
        m = trimesh.util.concatenate(geoms)

    if not isinstance(m, trimesh.Trimesh):
        raise ValueError(f"Unsupported mesh type from {mesh_path}: {type(m)}")

    mesh_sampling = str(mesh_sampling).lower().strip()
    N = int(num_points)

    if mesh_sampling == "surface":
        pts, _ = trimesh.sample.sample_surface(m, count=N)
        pts = np.asarray(pts, dtype=np.float32)
    elif mesh_sampling == "vertices":
        verts = np.asarray(m.vertices, dtype=np.float32)
        if verts.shape[0] == 0:
            raise ValueError(f"Mesh has 0 vertices: {mesh_path}")
        replace = verts.shape[0] < N
        idx = rng.choice(verts.shape[0], size=N, replace=replace)
        pts = verts[idx]
    else:
        raise ValueError(f"mesh_sampling must be 'surface' or 'vertices', got '{mesh_sampling}'")

    if pts.shape != (N, 3):
        pts = pts.reshape(N, 3).astype(np.float32, copy=False)
    return pts


def build_model_from_checkpoint(
    ckpt: Dict[str, Any],
    *,
    device: torch.device,
    use_modality_ln: Optional[bool] = None,  # None -> auto-try True then False
    latent_dim_override: Optional[int] = None,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Reconstructs MultiModalPolicy using checkpoint meta + state_dict shapes.
    Returns (model, meta_used).
    """
    if "model" not in ckpt:
        raise KeyError("Checkpoint missing key 'model' (state_dict).")
    state = ckpt["model"]
    meta = ckpt.get("meta", {}) or {}

    # Required meta (normally present from your training script)
    head_names = meta.get("head_names", None)
    head_dims_flat = meta.get("head_dims_flat", None)
    head_dim_step = meta.get("head_dim_step", None)

    if head_names is None or head_dims_flat is None:
        raise KeyError(
            "Checkpoint meta must contain 'head_names' and 'head_dims_flat'. "
            "Please ensure you saved checkpoints from the provided training script."
        )
    head_names = list(head_names)
    head_dims_flat = [int(x) for x in head_dims_flat]

    action_len = int(meta.get("action_len", 64))
    point_dim = int(meta.get("point_dim", 3))
    ignore_joint = bool(meta.get("ignore_joint", True))
    joint_dim = int(meta.get("joint_dim", 1))

    # pc Fourier params (present in your training meta)
    pc_num_bands = int(meta.get("pc_num_bands", 8))
    pc_max_freq = float(meta.get("pc_max_freq", 30.0))

    # policy knobs (present in your training meta)
    pc_center_xyz = bool(meta.get("pc_center_xyz", True))
    pc_xyz_scale = float(meta.get("pc_xyz_scale", 10.0))
    pc_feat_scale = float(meta.get("pc_feat_scale", 2.0))

    # ------------------------------------------------------------------
    # Backwards-compatibility: detect which optional submodules exist in the ckpt
    # ------------------------------------------------------------------
    ckpt_has_pc_ln = any(k.startswith("pc_ln.") for k in state)
    ckpt_has_joint_ln = any(k.startswith("joint_ln.") for k in state)
    ckpt_has_pc_global_head = any(k.startswith("pc_enc.global_head.") for k in state)

    # Infer latent_dim from weights unless user overrides
    latent_dim = None
    if latent_dim_override is not None:
        latent_dim = int(latent_dim_override)
    else:
        latent_dim = _infer_latent_dim_from_state_dict(state, head_names) or 512

    # Build Fourier features
    joint_pe = None
    if not ignore_joint:
        # These are defaults; if you trained with different joint PE settings, pass overrides and match them.
        joint_pe = FourierFeatures(joint_dim, num_bands=6, max_freq=20.0)
    point_xyz_pe = FourierFeatures(3, num_bands=pc_num_bands, max_freq=pc_max_freq)

    trunk_dropout = 0.1 if _ckpt_uses_dropout(state) else 0.0


    def _make_model(use_ln: bool) -> torch.nn.Module:
        model = MultiModalPolicy(
            joint_dim=joint_dim,
            point_dim=point_dim,
            latent_dim=latent_dim,
            joint_pe=joint_pe,
            point_xyz_pe=point_xyz_pe,
            head_dims=head_dims_flat,
            head_names=head_names,
            ignore_joint=ignore_joint,
            pc_center_xyz=pc_center_xyz,
            pc_xyz_scale=pc_xyz_scale,
            pc_feat_scale=pc_feat_scale,
            use_modality_ln=bool(use_ln),
            trunk_dropout=trunk_dropout,
        ).to(device)

        # If the checkpoint was trained before PointNetEncoder gained the xyz-mean fusion head,
        # disable it so state_dict keys match exactly.
        if not ckpt_has_pc_global_head and hasattr(model, "pc_enc") and hasattr(model.pc_enc, "add_xyz_mean"):
            model.pc_enc.add_xyz_mean = False
            model.pc_enc.global_head = torch.nn.Identity()

        return model

        # Auto-detect (and still fallback-try) the LayerNorm setting if not forced
    tried = []
    last_err = None

    if use_modality_ln is None:
        # If the ckpt contains LN parameters, it was trained with use_modality_ln=True.
        preferred = bool(ckpt_has_pc_ln or ckpt_has_joint_ln)
        candidates = (preferred, not preferred)
    else:
        candidates = (bool(use_modality_ln),)

    model = None
    use_ln_used: Optional[bool] = None

    for use_ln in candidates:
        tried.append(bool(use_ln))
        try:
            model = _make_model(bool(use_ln))
            model.load_state_dict(state, strict=True)
            use_ln_used = bool(use_ln)
            break
        except Exception as e:
            last_err = e

    if model is None:
        raise RuntimeError(
            f"Failed to load checkpoint with use_modality_ln in {tried}. "
            f"Last error: {last_err}"
        ) from last_err


    # If head_dim_step missing, derive it
    if head_dim_step is None:
        head_dim_step = {}
        for hn, flat in zip(head_names, head_dims_flat):
            if flat % action_len != 0:
                raise ValueError(f"Cannot derive d_step for head={hn}: flat={flat} not divisible by action_len={action_len}")
            head_dim_step[hn] = int(flat // action_len)

    meta_used = dict(meta)
    meta_used["latent_dim_inferred"] = int(latent_dim)
    meta_used["use_modality_ln_used"] = bool(use_ln_used)
    meta_used["pc_xyz_mean_fusion_used"] = bool(getattr(getattr(model, "pc_enc", None), "add_xyz_mean", False))
    meta_used["head_dim_step"] = dict(head_dim_step)
    return model, meta_used


def normalize_preds_to_dict(
    preds: Any,
    head_names: Sequence[str],
) -> Dict[str, torch.Tensor]:
    """
    Your training loop assumes dict {head: [B, T*D]}.
    Keep inference robust if model returns list/tuple.
    """
    if isinstance(preds, dict):
        return preds
    if isinstance(preds, (list, tuple)) and len(preds) == len(head_names):
        return {hn: preds[i] for i, hn in enumerate(head_names)}
    raise TypeError(f"Unexpected model output type: {type(preds)}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=str, help="Path to checkpoint .pt (e.g. ckpt_step_2000.pt)")
    ap.add_argument("--mesh", required=True, type=str, help="Path to mesh file (.stl/.obj/...)")
    ap.add_argument("--out_dir", default="./pred_txt", type=str, help="Directory to write output txt files")

    ap.add_argument("--device", default="cuda", type=str, help="cuda | cpu")
    ap.add_argument("--seed", default=0, type=int, help="Random seed (affects mesh sampling if needed)")

    ap.add_argument("--num_points", default=None, type=int, help="Number of points to sample from mesh (default: from ckpt meta or 4096)")
    ap.add_argument("--mesh_sampling", default=None, type=str, help="surface | vertices (default: from ckpt meta or 'surface')")

    ap.add_argument("--use_modality_ln", default="auto", type=str, choices=["auto", "true", "false"],
                    help="auto tries True then False; use true/false to force")
    ap.add_argument("--latent_dim", default=None, type=int, help="Override latent_dim. If omitted, inferred from ckpt weights.")

    ap.add_argument("--float_fmt", default="%.8f", type=str, help="np.savetxt float format")
    args = ap.parse_args()

    seed_everything(int(args.seed))

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    use_ln = _as_bool_tri(args.use_modality_ln)

    model, meta_used = build_model_from_checkpoint(
        ckpt,
        device=device,
        use_modality_ln=use_ln,
        latent_dim_override=args.latent_dim,
    )
    model.eval()

    # Resolve sampling params
    meta = ckpt.get("meta", {}) or {}
    action_len = int(meta.get("action_len", 64))
    head_names = list(meta.get("head_names", []))
    head_dim_step = dict(meta_used.get("head_dim_step", {}))
    ignore_joint = bool(meta.get("ignore_joint", True))
    point_dim = int(meta.get("point_dim", 3))

    num_points = args.num_points if args.num_points is not None else meta.get("num_points", 4096)
    if num_points is None:
        num_points = 4096
    num_points = int(num_points)

    mesh_sampling = args.mesh_sampling if args.mesh_sampling is not None else meta.get("mesh_sampling", "surface")
    mesh_sampling = str(mesh_sampling)

    # Sample points
    pts_xyz = load_pointcloud_from_mesh(
        args.mesh,
        num_points=num_points,
        mesh_sampling=mesh_sampling,
        seed=int(args.seed),
    )  # [N,3]

    # Build point features [N, point_dim]
    if point_dim <= 0:
        raise ValueError(f"Invalid point_dim={point_dim}")
    pc_np = np.zeros((num_points, point_dim), dtype=np.float32)
    pc_np[:, : min(3, point_dim)] = pts_xyz[:, : min(3, point_dim)]

    pc = torch.from_numpy(pc_np).unsqueeze(0).to(device=device, dtype=torch.float32)  # [1,N,F]

    # Joint input
    joint = None if ignore_joint else torch.zeros((1, int(meta.get("joint_dim", 1))), device=device, dtype=torch.float32)

    with torch.no_grad():
        preds_any = model(joint=joint, point_cloud=pc, point_mask=None)
        preds = normalize_preds_to_dict(preds_any, head_names)

    os.makedirs(args.out_dir, exist_ok=True)

    # Save per head
    for hn in head_names:
        if hn not in preds:
            print(f"[WARN] Missing prediction for head='{hn}' in model output keys={list(preds.keys())}")
            continue

        pred_flat_t = preds[hn]
        if pred_flat_t.ndim != 2 or pred_flat_t.shape[0] != 1:
            raise ValueError(f"Expected preds['{hn}'] shape [1, T*D], got {tuple(pred_flat_t.shape)}")

        pred_flat = pred_flat_t[0].detach().cpu().numpy()

        d_step = head_dim_step.get(hn, None)
        if d_step is None:
            # try derive from pred size
            if pred_flat.size % action_len != 0:
                raise ValueError(f"Cannot derive d_step for head={hn}: pred.size={pred_flat.size}, action_len={action_len}")
            d_step = int(pred_flat.size // action_len)

        save_pred_txt(
            args.out_dir,
            hn,
            pred_flat,
            action_len=action_len,
            d_step=int(d_step),
            float_fmt=str(args.float_fmt),
        )

        print(f"[OK] Saved head='{hn}' -> {os.path.join(args.out_dir, head_to_filename(hn))}  (T={action_len}, D={int(d_step)})")

    print("\nDone.")
    print(f"Output directory: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
