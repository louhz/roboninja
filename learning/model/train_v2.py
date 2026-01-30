# train_pointcloud_action_wandb.py (ADAPTED FOR NEW MultiModalPolicy + NEW MultiHeadLoss)
#
# Key changes vs your old script:
# - No longer assumes exactly 4 heads (head_names can be any length)
# - Loss weights can be per-head (dict) and we can optionally ignore inactive heads in the loss
# - Uses the new policy knobs to make point-cloud influence/gradients stronger:
#     pc_center_xyz, pc_xyz_scale, pc_feat_scale
# - If ignore_joint=True, we pass joint=None (and the model truly becomes point-cloud-only)

import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader
import wandb

# ---------------------------------------------------------------------
# Path setup (keep your old pattern)
# ---------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, PARENT_DIR)

# ---------------------------------------------------------------------
# Imports: dataloader + model
# ---------------------------------------------------------------------
try:
    from model.dataloader import (
        PointCloudActionEpisodeDataset,
        EXPECTED_DIMS_ALL,
    )
except Exception:
    from dataloader import (
        PointCloudActionEpisodeDataset,
        EXPECTED_DIMS_ALL,
    )


from model.model_raw import (
        FourierFeatures,
        MultiModalPolicy,
        MultiHeadLoss,
        MultiHeadLossConfig,
    )


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def seed_everything(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pad_stack_pointcloud_if_needed(
    pc: Union[torch.Tensor, List[torch.Tensor]],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Accepts either:
      - pc: Tensor [B,N,F]
      - pc: list of length B where each element is Tensor [Ni,F]
    Returns:
      - pc_out: [B,Nmax,F]
      - mask:   [B,Nmax] bool (True=valid), or None if input was already a tensor
    """
    if isinstance(pc, torch.Tensor):
        return pc, None

    if not isinstance(pc, list) or len(pc) == 0:
        raise ValueError("point_cloud must be a Tensor or a non-empty list of Tensors")

    if not all(isinstance(x, torch.Tensor) and x.dim() == 2 for x in pc):
        raise ValueError("point_cloud list must contain only 2D tensors [Ni,F]")

    B = len(pc)
    F = int(pc[0].shape[-1])
    Nmax = max(int(x.shape[0]) for x in pc)

    pc_out = torch.zeros((B, Nmax, F), dtype=pc[0].dtype)
    mask = torch.zeros((B, Nmax), dtype=torch.bool)
    for i, x in enumerate(pc):
        n = int(x.shape[0])
        pc_out[i, :n] = x
        mask[i, :n] = True

    return pc_out, mask


def _move_optimizer_state_to_device(opt: torch.optim.Optimizer, device: torch.device) -> None:
    """Move optimizer state tensors to the same device as model params (important for resume)."""
    for state in opt.state.values():
        for k, v in list(state.items()):
            if torch.is_tensor(v):
                state[k] = v.to(device, non_blocking=True)


def _infer_optimizer_step(opt: torch.optim.Optimizer) -> Optional[int]:
    """
    Adam/AdamW keeps a per-parameter 'step' counter.
    After N optimizer updates, this is typically N (as tensor).
    """
    for state in opt.state.values():
        if "step" in state:
            s = state["step"]
            try:
                if torch.is_tensor(s):
                    return int(s.item())
                return int(s)
            except Exception:
                return None
    return None


def save_ckpt(path: str, model, opt, step: int, meta: Dict[str, Any]):
    """
    Here, 'step' is the *last executed step index* (what your loop calls global_step at save-time).
    We also store 'next_step' for robust resume.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "step": int(step),
            "next_step": int(step + 1),
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "meta": meta,
        },
        path,
    )


def try_load_latest_ckpt(save_dir: str, model, opt, device: torch.device) -> int:
    """
    Returns the correct *next* global_step to run.
    Fixes:
      - off-by-one resume
      - optimizer state device mismatch (CPU vs CUDA)
    """
    if not os.path.isdir(save_dir):
        return 0

    candidates = []
    for fn in os.listdir(save_dir):
        m = re.match(r"ckpt_step_(\d+)\.pt$", fn)
        if m:
            candidates.append((int(m.group(1)), os.path.join(save_dir, fn)))

    if not candidates:
        return 0

    candidates.sort(key=lambda x: x[0])
    _, path = candidates[-1]

    data = torch.load(path, map_location="cpu")
    model.load_state_dict(data["model"])
    opt.load_state_dict(data["opt"])

    opt_step = _infer_optimizer_step(opt)
    _move_optimizer_state_to_device(opt, device)

    if opt_step is not None:
        return int(opt_step)

    if "next_step" in data:
        return int(data["next_step"])

    ckpt_step = int(data.get("step", 0))
    return ckpt_step + 1


def _resolve_run_id_file(cfg: "TrainConfig") -> str:
    """
    Store run_id_file inside save_dir when a relative path is given.
    Prevents accidentally reusing ./wandb_run_id.txt across different runs.
    """
    if cfg.run_id_file is None or str(cfg.run_id_file).strip() == "":
        return os.path.join(cfg.save_dir, "wandb_run_id.txt")
    if os.path.isabs(cfg.run_id_file):
        return cfg.run_id_file
    return os.path.join(cfg.save_dir, os.path.basename(cfg.run_id_file))


def resolve_head_weights(
    head_names: Sequence[str],
    head_weights: Union[Sequence[float], Mapping[str, float]],
    *,
    active_groups: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """
    Returns a dict {head_name: weight} aligned to head_names.

    - If head_weights is a dict: use it (default 1.0 for missing)
    - If head_weights is a list/tuple:
        * len==0 -> all 1.0
        * len==1 -> broadcast
        * len<N -> pad with last
        * len>N -> truncate
    - If active_groups is provided, heads not in active_groups get weight 0.0
    """
    names = list(head_names)
    if isinstance(head_weights, Mapping):
        w = {hn: float(head_weights.get(hn, 1.0)) for hn in names}
    else:
        hw = list(map(float, head_weights)) if head_weights is not None else []
        if len(hw) == 0:
            hw = [1.0] * len(names)
        elif len(hw) == 1:
            hw = [hw[0]] * len(names)
        elif len(hw) < len(names):
            hw = hw + [hw[-1]] * (len(names) - len(hw))
        else:
            hw = hw[: len(names)]
        w = {hn: float(val) for hn, val in zip(names, hw)}

    if active_groups is not None:
        active = set(active_groups)
        for hn in names:
            if hn not in active:
                w[hn] = 0.0
    return w


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass
class TrainConfig:
    base_path: str = "./example_dataset/strawberry_real"

    mesh_subdir: str = "mesh"
    mesh_name: str = "strawberry.stl"
    control_subdir: str = "control"

    mesh_sampling: str = "surface"   # "surface" | "vertices"
    mesh_points: int = 8192

    num_points: Optional[int] = 4096
    sampling: str = "random"  # "random" or "first"
    features: Tuple[str, ...] = ("xyz",)

    cache_point_cloud: bool = True
    pc_cache_max_items: int = 0

    # IMPORTANT: head_names can now be ANY length (e.g., ("n2","n5") to remove robotic hands)
    head_names: Tuple[str, ...] = ("n2", "n5")
    groups: Tuple[str, ...] = ("n2", "n5")
    missing_policy: str = "initial"

    action_len: int = 64
    interp_mode_map: Optional[Dict[str, str]] = None
    action_format: str = "dict"

    batch_size: int = 8
    num_workers: int = 8
    shuffle: bool = True
    drop_last: bool = True
    persistent_workers: bool = True
    pin_memory: bool = True
    prefetch_factor: int = 2

    # Model
    ignore_joint: bool = True
    joint_dim: int = 1  # ignored if ignore_joint=True

    latent_dim: int = 512
    trunk_dropout: float = 0.1

    # Point-cloud gradient / sensitivity knobs (NEW policy)
    pc_center_xyz: bool = False
    pc_xyz_scale: float = 10.0     # try 10 -> 50 -> 100 if still weak gradients
    pc_feat_scale: float = 2.0     # boosts PC influence into trunk
    use_modality_ln: bool = True

    # Fourier features
    joint_num_bands: int = 6
    joint_max_freq: float = 20.0
    pc_num_bands: int = 8
    pc_max_freq: float = 30.0

    # Optim
    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip_norm: float = 1.0
    max_steps: int = 2000
    log_every: int = 20
    ckpt_every: int = 1000

    # Loss (NEW loss supports per-head weights dict, max_scale, ignore_heads)
    loss_type: str = "rmse"
    loss_normalize: str = "per_dim_rms"
    loss_eps: float = 1e-6
    loss_min_scale: float = 1e-3
    loss_max_scale: Optional[float] = None  # set e.g. 100.0 if you suspect vanishing due to huge target scale
    loss_huber_beta: float = 1.0
    loss_detach_scale: bool = True

    # Can be:
    #   - (1.0,) to broadcast
    #   - (w1,w2,...)
    #   - {"n2":10.0,"n5":10.0} etc
    head_weights: Union[Tuple[float, ...], Dict[str, float]] = (1.0,)

    # If True: heads with weight 0.0 are skipped in loss (and targets can be omitted)
    ignore_inactive_heads_in_loss: bool = True

    # W&B
    wandb_project: str = "pointcloud-action"
    wandb_run_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_group: Optional[str] = None
    resume: bool = True
    run_id_file: str = "./wandb_run_id.txt"

    device: str = "cuda"
    seed: int = 0
    save_dir: str = "./checkpoints_new"


# ---------------------------------------------------------------------
# Build model + loss from first batch
# ---------------------------------------------------------------------
@torch.no_grad()
def build_model_and_loss(cfg: TrainConfig, first_batch: Dict[str, Any], device: torch.device):
    if len(cfg.head_names) < 1:
        raise ValueError("cfg.head_names must contain at least one head")
    if len(cfg.groups) > 0:
        for g in cfg.groups:
            if g not in cfg.head_names:
                raise ValueError(f"cfg.groups contains '{g}' not in cfg.head_names={cfg.head_names}")

    if len(cfg.features) < 1 or cfg.features[0].lower() != "xyz":
        raise ValueError(f"cfg.features must start with 'xyz'. Got {cfg.features}")

    pc_raw = first_batch["point_cloud"]
    pc, _ = pad_stack_pointcloud_if_needed(pc_raw)
    point_dim = int(pc.shape[-1])

    if "actions" not in first_batch or not isinstance(first_batch["actions"], dict):
        raise KeyError(
            "This training script expects batch['actions'] as a dict (action_format='dict'). "
            "Set cfg.action_format='dict' in the dataset."
        )
    actions: Dict[str, torch.Tensor] = first_batch["actions"]

    # Infer per-step dims for each head
    head_dim_step: Dict[str, int] = {}
    for hn in cfg.head_names:
        if hn in actions:
            # actions[hn] expected [B,T,D]
            head_dim_step[hn] = int(actions[hn].shape[-1])
        elif hn in EXPECTED_DIMS_ALL:
            head_dim_step[hn] = int(EXPECTED_DIMS_ALL[hn])
        else:
            raise ValueError(f"Cannot infer dim for head '{hn}' (not in batch actions and not in EXPECTED_DIMS_ALL)")

    T = int(cfg.action_len)
    head_dims_flat: List[int] = [int(head_dim_step[hn]) * T for hn in cfg.head_names]

    # Fourier features
    joint_pe = None
    if not cfg.ignore_joint:
        joint_pe = FourierFeatures(int(cfg.joint_dim), num_bands=int(cfg.joint_num_bands), max_freq=float(cfg.joint_max_freq))
    point_pe = FourierFeatures(3, num_bands=int(cfg.pc_num_bands), max_freq=float(cfg.pc_max_freq))

    # Build model (new policy supports pc_center_xyz/pc_xyz_scale/pc_feat_scale/etc)
    model = MultiModalPolicy(
        joint_dim=int(cfg.joint_dim),
        point_dim=point_dim,
        latent_dim=int(cfg.latent_dim),
        joint_pe=joint_pe,
        point_xyz_pe=point_pe,
        trunk_dropout=float(cfg.trunk_dropout),
        head_dims=head_dims_flat,
        head_names=list(cfg.head_names),
        ignore_joint=bool(cfg.ignore_joint),
        pc_center_xyz=bool(cfg.pc_center_xyz),
        pc_xyz_scale=float(cfg.pc_xyz_scale),
        pc_feat_scale=float(cfg.pc_feat_scale),
        use_modality_ln=bool(cfg.use_modality_ln),
    ).to(device)

    # Resolve per-head weights and ignored heads
    weights_dict = resolve_head_weights(
        head_names=cfg.head_names,
        head_weights=cfg.head_weights,
        active_groups=cfg.groups if len(cfg.groups) > 0 else None,
    )
    ignore_heads: Tuple[str, ...] = ()
    if cfg.ignore_inactive_heads_in_loss:
        ignore_heads = tuple([hn for hn in cfg.head_names if float(weights_dict.get(hn, 1.0)) == 0.0])

    # Build loss (new loss supports weights dict + max_scale + ignore_heads)
    loss_cfg = MultiHeadLossConfig(
        loss_type=str(cfg.loss_type),
        normalize=str(cfg.loss_normalize),
        weights=weights_dict,
        eps=float(cfg.loss_eps),
        min_scale=float(cfg.loss_min_scale),
        max_scale=(None if cfg.loss_max_scale is None else float(cfg.loss_max_scale)),
        huber_beta=float(cfg.loss_huber_beta),
        detach_scale=bool(cfg.loss_detach_scale),
        ignore_heads=ignore_heads,
    )
    loss_fn = MultiHeadLoss(head_names=list(cfg.head_names), cfg=loss_cfg).to(device)

    meta = {
        "head_names": list(cfg.head_names),
        "head_dim_step": dict(head_dim_step),
        "head_dims_flat": list(head_dims_flat),
        "action_len": int(cfg.action_len),
        "point_dim": int(point_dim),
        "ignore_joint": bool(cfg.ignore_joint),
        "joint_dim": int(cfg.joint_dim),
        "groups": list(cfg.groups),
        "head_weights_used": dict(weights_dict),
        "ignore_heads_in_loss": list(ignore_heads),
        "features": cfg.features,
        "sampling": cfg.sampling,
        "num_points": cfg.num_points,
        "missing_policy": cfg.missing_policy,
        "action_format": cfg.action_format,
        "mesh_sampling": cfg.mesh_sampling,
        "mesh_points": int(cfg.mesh_points),
        "mesh_name": cfg.mesh_name,
        "mesh_subdir": cfg.mesh_subdir,
        "control_subdir": cfg.control_subdir,
        "loss_type": cfg.loss_type,
        "loss_normalize": cfg.loss_normalize,
        "loss_eps": cfg.loss_eps,
        "loss_min_scale": cfg.loss_min_scale,
        "loss_max_scale": cfg.loss_max_scale,
        "loss_huber_beta": cfg.loss_huber_beta,
        "loss_detach_scale": cfg.loss_detach_scale,
        "pc_center_xyz": bool(cfg.pc_center_xyz),
        "pc_xyz_scale": float(cfg.pc_xyz_scale),
        "pc_feat_scale": float(cfg.pc_feat_scale),
        "pc_num_bands": int(cfg.pc_num_bands),
        "pc_max_freq": float(cfg.pc_max_freq),
    }
    return model, loss_fn, meta


def flatten_targets(
    actions: Dict[str, torch.Tensor],
    include_heads: Sequence[str],
    head_dim_step: Mapping[str, int],
    action_len: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Creates flattened targets dict:
      actions[hn]: [B,T,D] -> targets[hn]: [B, T*D]
    If a head is missing, fills zeros.
    """
    targets: Dict[str, torch.Tensor] = {}

    # infer batch size
    B = None
    for v in actions.values():
        if isinstance(v, torch.Tensor):
            B = int(v.shape[0])
            break
    if B is None:
        raise ValueError("actions dict is empty; cannot infer batch size")

    T = int(action_len)

    for hn in include_heads:
        d_step = int(head_dim_step[hn])
        if hn in actions:
            x = actions[hn]
            if x.dim() != 3:
                raise ValueError(f"actions['{hn}'] must be [B,T,D], got {tuple(x.shape)}")
            if int(x.shape[1]) != T:
                raise ValueError(f"actions['{hn}'] has T={x.shape[1]} but action_len={T}")
            if int(x.shape[2]) != d_step:
                raise ValueError(f"actions['{hn}'] has D={x.shape[2]} but expected {d_step}")
            targets[hn] = x.reshape(B, T * d_step).to(device, dtype=torch.float32, non_blocking=True)
        else:
            targets[hn] = torch.zeros((B, T * d_step), device=device, dtype=torch.float32)
    return targets


# ---------------------------------------------------------------------
# train_one_epoch
# ---------------------------------------------------------------------
def train_one_epoch(
    *,
    cfg: TrainConfig,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    opt: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    meta: Dict[str, Any],
    global_step: int,
    train_start_step: int,
    train_t0: float,
):
    model.train()

    head_names: List[str] = list(meta["head_names"])
    head_dim_step: Dict[str, int] = dict(meta["head_dim_step"])
    head_weights_used: Dict[str, float] = dict(meta["head_weights_used"])
    ignore_heads: set = set(meta.get("ignore_heads_in_loss", []))

    # Only build targets for heads that the loss will actually use (optional optimization)
    if cfg.ignore_inactive_heads_in_loss and len(ignore_heads) > 0:
        target_heads = [hn for hn in head_names if hn not in ignore_heads]
    else:
        target_heads = head_names

    running: Dict[str, float] = {"loss_total": 0.0, "grad_norm": 0.0}
    # Only track per-head losses that the loss_fn returns (safer when ignore_heads is used)
    for hn in target_heads:
        running[f"loss_{hn}"] = 0.0

    n = 0
    num_batches = 0

    for batch in loader:
        if global_step >= cfg.max_steps:
            break

        pc_raw = batch["point_cloud"]
        pc, point_mask = pad_stack_pointcloud_if_needed(pc_raw)
        pc = pc.to(device, dtype=torch.float32, non_blocking=True)
        point_mask = point_mask.to(device, non_blocking=True) if point_mask is not None else None
        B = int(pc.shape[0])

        # Joint input
        if cfg.ignore_joint:
            joint = None
        else:
            # dataset doesn't provide joints in your script; keep zeros
            joint = torch.zeros((B, int(cfg.joint_dim)), device=device, dtype=torch.float32)

        actions: Dict[str, torch.Tensor] = batch["actions"]
        targets = flatten_targets(
            actions=actions,
            include_heads=target_heads,          # only heads that matter
            head_dim_step=head_dim_step,
            action_len=int(cfg.action_len),
            device=device,
        )

        preds = model(joint=joint, point_cloud=pc, point_mask=point_mask)
        total_loss, loss_by_head = loss_fn(preds, targets)

        opt.zero_grad(set_to_none=True)
        total_loss.backward()

        if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            grad_norm = float(gn) if not isinstance(gn, float) else gn
        else:
            grad_norm = 0.0

        opt.step()

        n += B
        num_batches += 1
        running["loss_total"] += float(total_loss.detach().cpu()) * B
        running["grad_norm"] += float(grad_norm) * B
        for hn, l in loss_by_head.items():
            running[f"loss_{hn}"] += float(l.detach().cpu()) * B

        if cfg.log_every and cfg.log_every > 0 and (global_step % cfg.log_every) == 0:
            dt = time.time() - train_t0
            steps_per_sec = (global_step - train_start_step + 1) / max(dt, 1e-9)

            log_dict = {
                "train/step": global_step,
                "train/loss_total": float(total_loss.detach().cpu()),
                "train/grad_norm": float(grad_norm),
                "perf/steps_per_sec": steps_per_sec,
                "data/batch_size": B,
                "data/point_dim": int(pc.shape[-1]),
                "data/num_points": int(pc.shape[1]),
                "data/action_len": int(cfg.action_len),
                "loss/type": str(cfg.loss_type),
                "loss/normalize": str(cfg.loss_normalize),
                "loss/min_scale": float(cfg.loss_min_scale),
                "loss/max_scale": (None if cfg.loss_max_scale is None else float(cfg.loss_max_scale)),
                "model/pc_xyz_scale": float(cfg.pc_xyz_scale),
                "model/pc_feat_scale": float(cfg.pc_feat_scale),
            }

            # Per-head loss
            for hn, l in loss_by_head.items():
                log_dict[f"train/loss_{hn}"] = float(l.detach().cpu())

            # Per-head weights (log all model heads for transparency)
            for hn in head_names:
                log_dict[f"loss_weight/{hn}"] = float(head_weights_used.get(hn, 1.0))
                log_dict[f"loss_ignored/{hn}"] = 1.0 if hn in ignore_heads else 0.0

            meta_list = batch.get("meta", None)
            if isinstance(meta_list, list) and len(meta_list) > 0 and isinstance(meta_list[0], dict):
                if "episode_id" in meta_list[0]:
                    log_dict["data/example_episode_id"] = meta_list[0]["episode_id"]

            if wandb.run is not None:
                wandb.log(log_dict)

        if cfg.ckpt_every and cfg.ckpt_every > 0 and (global_step > 0) and (global_step % cfg.ckpt_every == 0):
            ckpt_path = os.path.join(cfg.save_dir, f"ckpt_step_{global_step}.pt")
            save_ckpt(ckpt_path, model, opt, step=global_step, meta=meta)

            if wandb.run is not None:
                art = wandb.Artifact(name=f"checkpoint-{wandb.run.id}", type="model", metadata={"step": global_step})
                art.add_file(ckpt_path)
                wandb.log_artifact(art)

        global_step += 1

    if num_batches == 0:
        raise RuntimeError(
            "DataLoader produced 0 batches in this epoch. "
            "Common cause: drop_last=True with batch_size > dataset_len."
        )

    denom = max(n, 1)
    for k in list(running.keys()):
        running[k] /= denom
    running["num_samples"] = float(n)
    running["num_batches"] = float(num_batches)
    return running, global_step


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main(cfg: TrainConfig):
    seed_everything(cfg.seed)

    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
    os.makedirs(cfg.save_dir, exist_ok=True)

    run_id_path = _resolve_run_id_file(cfg)

    ds = PointCloudActionEpisodeDataset(
        obs_root=cfg.base_path,
        act_root=None,

        mesh_subdir=cfg.mesh_subdir,
        mesh_name=cfg.mesh_name,
        control_subdir=cfg.control_subdir,

        mesh_sampling=cfg.mesh_sampling,
        mesh_points=cfg.mesh_points,

        groups=cfg.groups,
        missing_policy=cfg.missing_policy,
        action_len=cfg.action_len,
        action_format=cfg.action_format,
        interp_mode_map=cfg.interp_mode_map,
        num_points=cfg.num_points,
        sampling=cfg.sampling,
        features=cfg.features,
        seed=cfg.seed,
        cache_point_cloud=cfg.cache_point_cloud,
        pc_cache_max_items=cfg.pc_cache_max_items,
    )

    dl_kwargs = dict(
        dataset=ds,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        drop_last=cfg.drop_last,
        pin_memory=(cfg.pin_memory and str(device).startswith("cuda")),
        persistent_workers=(cfg.persistent_workers and cfg.num_workers > 0),
    )
    if cfg.num_workers > 0:
        dl_kwargs["prefetch_factor"] = int(cfg.prefetch_factor)

    loader = DataLoader(**dl_kwargs)

    first_batch = next(iter(loader))
    model, loss_fn, meta = build_model_and_loss(cfg, first_batch, device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Resume training state
    start_step = try_load_latest_ckpt(cfg.save_dir, model, opt, device) if cfg.resume else 0
    global_step = int(start_step)

    # Resume W&B run id (safer path)
    run_id = None
    if cfg.resume and os.path.exists(run_id_path):
        try:
            run_id = open(run_id_path, "r", encoding="utf-8").read().strip() or None
        except Exception:
            run_id = None

    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.wandb_run_name,
        group=cfg.wandb_group,
        config={**asdict(cfg), "run_id_file": run_id_path, **meta},
        id=run_id,
        resume="allow" if cfg.resume else None,
        settings=wandb.Settings(start_method="thread"),
    )

    # Define step metrics (prevents monotonic step issues)
    if wandb.run is not None:
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("perf/*", step_metric="train/step")
        wandb.define_metric("data/*", step_metric="train/step")
        wandb.define_metric("loss/*", step_metric="train/step")
        wandb.define_metric("model/*", step_metric="train/step")
        wandb.define_metric("epoch")
        wandb.define_metric("epoch/*", step_metric="epoch")

    try:
        os.makedirs(os.path.dirname(run_id_path), exist_ok=True)
        with open(run_id_path, "w", encoding="utf-8") as f:
            if wandb.run is not None:
                f.write(wandb.run.id)
    except Exception:
        pass

    train_t0 = time.time()
    train_start_step = int(start_step)
    epoch = 0

    try:
        while global_step < cfg.max_steps:
            epoch += 1
            epoch_stats, global_step = train_one_epoch(
                cfg=cfg,
                model=model,
                loss_fn=loss_fn,
                opt=opt,
                loader=loader,
                device=device,
                meta=meta,
                global_step=global_step,
                train_start_step=train_start_step,
                train_t0=train_t0,
            )

            if wandb.run is not None:
                wandb.log({"epoch": epoch, **{f"epoch/{k}": v for k, v in epoch_stats.items()}})

        # Final save (keep your original filename pattern)
        last_step = max(0, int(global_step) - 1)
        final_path = os.path.join(cfg.save_dir, f"ckpt_step_{cfg.max_steps}.pt")
        os.makedirs(cfg.save_dir, exist_ok=True)
        torch.save(
            {
                "step": int(last_step),
                "next_step": int(global_step),
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "meta": meta,
            },
            final_path,
        )

        if wandb.run is not None:
            art = wandb.Artifact(name=f"checkpoint-{wandb.run.id}", type="model", metadata={"step": int(global_step)})
            art.add_file(final_path)
            wandb.log_artifact(art)

    finally:
        wandb.finish()


if __name__ == "__main__":
    # Example: removing robotic hand outputs by only training n2/n5
    cfg = TrainConfig(
        base_path="/home/louhz/Desktop/Rss/roboninja/example_dataset/apple_real",
        mesh_subdir="mesh",
        mesh_name="apple.stl",
        control_subdir="control",

        mesh_sampling="surface",
        mesh_points=8192,

        wandb_project="apple_real",
        wandb_run_name="test",

        # <-- remove hand heads here
        head_names=("n2", "n5"),
        groups=("n2", "n5"),

        missing_policy="initial",
        action_len=64,
        action_format="dict",

        batch_size=4,
        num_workers=1,
        persistent_workers=True,
        num_points=4096,
        features=("xyz",),
        cache_point_cloud=True,
        pc_cache_max_items=512,
        device="cuda",

        # Loss weights: broadcast (1.0,) or dict like {"n2":10.0,"n5":10.0}
        head_weights=(1.0,),

        # Make gradients more explicit
        pc_center_xyz=False,
        pc_xyz_scale=10.0,
        pc_feat_scale=2.0,

        drop_last=False,
        save_dir="check_apple",

        loss_type="rmse",
        loss_normalize="per_dim_rms",
        loss_min_scale=1e-3,
        loss_eps=1e-6,
        loss_huber_beta=1.0,
        loss_detach_scale=True,
        loss_max_scale=None,  # optionally set 100.0 if needed
    )
    main(cfg)
