# train_rh_pointcloud_chunk_wandb.py
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple, List

import torch
from torch.utils.data import DataLoader
import wandb

from learning.model.dataloader import (
    PointCloudTactileActionEpisodeDataset,
    collate_pointcloud_to_action_chunk,
)

from learning.model.model import (
    FourierFeatures,
    MultiModalPolicy,
    MultiHeadLoss,
    MultiHeadLossConfig,
    infer_head_dims_from_action,
    split_action_vector,
)

# -------------------------
# Small helpers (robust to key names / nesting)
# -------------------------
def pick_first_present(d, keys):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    raise KeyError(
        f"None of these keys found: {keys}. "
        f"Available keys: {list(d.keys()) if isinstance(d, dict) else type(d)}"
    )

def get_obs_dict(batch):
    return batch.get("obs", batch) if isinstance(batch, dict) else batch

def get_action_chunk(batch):
    # our collate returns "actions" (dict)
    return pick_first_present(batch, ["action_chunk", "action_seq", "actions", "action", "act"])

def get_point_cloud(obs):
    return pick_first_present(obs, ["point_cloud", "pc", "points", "cloud"])

def get_joint(obs, B, device):
    # Episode dataset has no joints -> dummy.
    for k in ["joint", "joints", "joint_angles", "qpos", "hand_qpos"]:
        if isinstance(obs, dict) and k in obs:
            return obs[k]
    return torch.zeros(B, 1, device=device)

def get_tactile(obs):
    if not isinstance(obs, dict):
        return None
    for k in ["tactile", "tactile_state", "taxel", "touch"]:
        if k in obs:
            return obs[k]
    return None

def get_action_mask(batch):
    if not isinstance(batch, dict):
        return None
    for k in ["action_mask", "mask", "valid_mask", "seq_mask"]:
        if k in batch:
            return batch[k]
    return None

def normalize_head_weights(head_names: List[str], cfg_weights: Tuple[float, ...]) -> Tuple[float, ...]:
    """
    Make cfg.head_weights match len(head_names).
    - If cfg_weights has length 1: broadcast
    - If too short: pad with last weight
    - If too long: truncate
    """
    if cfg_weights is None or len(cfg_weights) == 0:
        return tuple([1.0] * len(head_names))
    if len(cfg_weights) == 1 and len(head_names) > 1:
        return tuple([float(cfg_weights[0])] * len(head_names))
    if len(cfg_weights) < len(head_names):
        pad = [float(cfg_weights[-1])] * (len(head_names) - len(cfg_weights))
        return tuple(list(map(float, cfg_weights)) + pad)
    if len(cfg_weights) > len(head_names):
        return tuple(map(float, cfg_weights[: len(head_names)]))
    return tuple(map(float, cfg_weights))

def preprocess_tactile(tactile: Optional[torch.Tensor], mode: str) -> Optional[torch.Tensor]:
    """
    Collate returns tactile as:
      tactile: [B, T, C, P]  (C=5, P=96 by default)
    Most models want:
      [B, D] or [B, T, D], so we flatten C*P.

    mode:
      - "first": use tactile[:,0] -> [B, D]
      - "last":  use tactile[:,-1] -> [B, D]
      - "mean":  mean over time -> [B, D]
      - "sequence": keep [B, T, D]
    """
    if tactile is None:
        return None

    if tactile.dim() == 4:
        # [B,T,C,P] -> [B,T,C*P]
        tactile = tactile.flatten(2)
    elif tactile.dim() == 3:
        # already [B,T,D]
        pass
    elif tactile.dim() == 2:
        # already [B,D]
        pass
    else:
        raise ValueError(f"Unexpected tactile shape: {tuple(tactile.shape)}")

    if mode == "sequence":
        return tactile  # [B,T,D] (or [B,D] if it came that way)
    if tactile.dim() == 2:
        return tactile  # already [B,D]

    # tactile is [B,T,D]
    if mode == "first":
        return tactile[:, 0]
    if mode == "last":
        return tactile[:, -1]
    if mode == "mean":
        return tactile.mean(dim=1)
    raise ValueError("tactile_mode must be one of {'first','last','mean','sequence'}")


# -------------------------
# Config
# -------------------------
@dataclass
class TrainConfig:
    # data
    obs_root: str = "path/to/obs_root"
    tactile_root: str = "path/to/tactile_root"     # ✅ NEW
    act_root: str = "path/to/act_root"

    # episode file names (your desired format)
    pc_name: str = "point.ply"
    tactile_name: str = "tactile.logs"

    # tactile shape (default matches your logging: 5 channels x 96 points)
    tactile_channels: int = 5
    tactile_points: int = 96
    tactile_mode: str = "first"   # ✅ recommended for compatibility: "first" / "sequence"

    # action heads to train
    groups: Tuple[str, ...] = ("rh",)   # ✅ right-hand only (set to ("n2","n5","lh","rh") if you want all)

    num_points: int = 4096
    features: Tuple[str, ...] = ("xyz",)
    batch_size: int = 32
    num_workers: int = 8
    shuffle: bool = True

    # chunking
    chunk_len: int = 64

    # optimization
    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip_norm: float = 1.0
    max_steps: int = 20000
    log_every: int = 20
    ckpt_every: int = 1000

    # loss
    loss_type: str = "mse"
    head_weights: Tuple[float, ...] = (1.0,)  # ✅ match groups; broadcast/pad/truncate handled

    # wandb
    wandb_project: str = "rh-pointcloud-chunk"
    wandb_run_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_group: Optional[str] = None
    resume: bool = True
    run_id_file: str = "./wandb_run_id.txt"

    # misc
    device: str = "cuda"
    seed: int = 0
    save_dir: str = "./checkpoints"


def seed_everything(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Build model + infer heads (from one batch)
# -------------------------
@torch.no_grad()
def infer_heads_from_batch(action_chunk: Any):
    """
    Returns (head_names, head_dims_step, T_in_batch).
    """
    if isinstance(action_chunk, dict):
        sample_action = {}
        for k, v in action_chunk.items():
            if v.dim() == 3:
                sample_action[k] = v[0, 0]
            elif v.dim() == 2:
                sample_action[k] = v[0]
            else:
                raise ValueError(f"Unexpected action dict tensor shape for key={k}: {v.shape}")
        head_names, head_dims_step = infer_head_dims_from_action(sample_action)
        any_v = next(iter(action_chunk.values()))
        T = any_v.shape[1] if any_v.dim() == 3 else 1
        return head_names, head_dims_step, T

    # tensor action
    if action_chunk.dim() == 3:
        T = action_chunk.shape[1]
        sample_step = action_chunk[0, 0]
    elif action_chunk.dim() == 2:
        T = 1
        sample_step = action_chunk[0]
    else:
        raise ValueError(f"Unexpected action_chunk shape: {action_chunk.shape}")

    head_names, head_dims_step = infer_head_dims_from_action(sample_step)
    return head_names, head_dims_step, T


def build_model_from_batch(cfg: TrainConfig, batch: Dict[str, Any], device: torch.device):
    obs = get_obs_dict(batch)

    pc = get_point_cloud(obs).to(device)
    B = pc.shape[0]

    joint = get_joint(obs, B=B, device=device).to(device)

    tactile = get_tactile(obs)
    tactile = tactile.to(device) if tactile is not None else None
    tactile = preprocess_tactile(tactile, mode=cfg.tactile_mode)
    tactile_dim = 0 if tactile is None else tactile.shape[-1]

    action_chunk = get_action_chunk(batch)
    action_chunk = {k: v.to(device) for k, v in action_chunk.items()} if isinstance(action_chunk, dict) else action_chunk.to(device)

    head_names, head_dims_step, T = infer_heads_from_batch(action_chunk)

    # ✅ no more “must be 4 heads”
    if T not in (1, cfg.chunk_len):
        raise ValueError(f"Expected T==chunk_len or 1, got T={T}")

    point_dim = pc.shape[-1]
    joint_dim = joint.shape[-1]

    # Positional encodings
    joint_pe = FourierFeatures(joint_dim, num_bands=6, max_freq=20.0)
    point_pe = FourierFeatures(3, num_bands=6, max_freq=10.0)
    tactile_pe = FourierFeatures(tactile_dim, num_bands=4, max_freq=10.0) if tactile_dim > 0 else None

    head_dims_chunk = [d * cfg.chunk_len for d in head_dims_step]

    model = MultiModalPolicy(
        joint_dim=joint_dim,
        point_dim=point_dim,
        tactile_dim=tactile_dim,
        joint_pe=joint_pe,
        point_xyz_pe=point_pe,
        tactile_pe=tactile_pe,
        head_names=head_names,
        head_dims=head_dims_chunk,
    ).to(device)

    weights = normalize_head_weights(head_names, cfg.head_weights)
    loss_cfg = MultiHeadLossConfig(loss_type=cfg.loss_type, weights=weights)
    loss_fn = MultiHeadLoss(head_names, loss_cfg).to(device)

    meta = {
        "head_names": head_names,
        "head_dims_step": head_dims_step,
        "joint_dim": joint_dim,
        "point_dim": point_dim,
        "tactile_dim": tactile_dim,
        "tactile_mode": cfg.tactile_mode,
        "groups": cfg.groups,
        "head_weights_used": weights,
    }
    return model, loss_fn, meta


# -------------------------
# Loss computation (supports optional action_mask)
# -------------------------
def compute_losses(
    preds: Dict[str, torch.Tensor],            # head -> (B,T,d)
    targets: Dict[str, torch.Tensor],          # head -> (B,T,d)
    head_names: list,
    head_weights: Tuple[float, ...],
    action_mask: Optional[torch.Tensor] = None
):
    B, T = next(iter(preds.values())).shape[:2]
    device = next(iter(preds.values())).device

    if action_mask is None:
        total = torch.zeros((), device=device)
        by_head = {}
        for hn, w in zip(head_names, head_weights):
            p = preds[hn].reshape(B * T, -1)
            y = targets[hn].reshape(B * T, -1)
            l = torch.mean((p - y) ** 2)
            by_head[hn] = l
            total = total + (w * l)
        return total, by_head

    m = action_mask
    if m.dtype != torch.bool:
        m = m > 0.5
    m_flat = m.reshape(B * T)

    total = torch.zeros((), device=device)
    by_head = {}
    for hn, w in zip(head_names, head_weights):
        p = preds[hn].reshape(B * T, -1)
        y = targets[hn].reshape(B * T, -1)
        if m_flat.any():
            diff = p[m_flat] - y[m_flat]
            l = (diff * diff).mean()
        else:
            l = torch.zeros((), device=device)
        by_head[hn] = l
        total = total + (w * l)
    return total, by_head


# -------------------------
# Checkpoint I/O
# -------------------------
def save_ckpt(path: str, model, opt, step: int, meta: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"step": step, "model": model.state_dict(), "opt": opt.state_dict(), "meta": meta}, path)

def try_load_latest_ckpt(save_dir: str, model, opt) -> int:
    if not os.path.isdir(save_dir):
        return 0
    ckpts = [p for p in os.listdir(save_dir) if p.endswith(".pt")]
    if not ckpts:
        return 0
    ckpts.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    latest = os.path.join(save_dir, ckpts[-1])
    data = torch.load(latest, map_location="cpu")
    model.load_state_dict(data["model"])
    opt.load_state_dict(data["opt"])
    return int(data.get("step", 0))


# -------------------------
# Main training loop
# -------------------------
def main(cfg: TrainConfig):
    seed_everything(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")

    # ✅ NEW dataset (episode-level)
    ds = PointCloudTactileActionEpisodeDataset(
        obs_root=cfg.obs_root,
        tactile_root=cfg.tactile_root,
        act_root=cfg.act_root,
        pc_name=cfg.pc_name,
        tactile_name=cfg.tactile_name,
        groups=cfg.groups,
        tactile_channels=cfg.tactile_channels,
        tactile_points=cfg.tactile_points,
        num_points=cfg.num_points,
        features=cfg.features,
    )

    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,  # ✅ shuffle EPISODES (correct)
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        collate_fn=lambda b: collate_pointcloud_to_action_chunk(b, chunk_len=cfg.chunk_len),
    )

    # build from a real batch
    first_batch = next(iter(loader))
    model, loss_fn_unused, meta = build_model_from_batch(cfg, first_batch, device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    start_step = try_load_latest_ckpt(cfg.save_dir, model, opt)

    run_id = None
    if cfg.resume and os.path.exists(cfg.run_id_file):
        try:
            run_id = open(cfg.run_id_file, "r", encoding="utf-8").read().strip() or None
        except Exception:
            run_id = None

    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.wandb_run_name,
        group=cfg.wandb_group,
        config={**asdict(cfg), **meta},
        id=run_id,
        resume="allow" if cfg.resume else None,
    )

    try:
        with open(cfg.run_id_file, "w", encoding="utf-8") as f:
            f.write(wandb.run.id)
    except Exception:
        pass

    head_names = meta["head_names"]
    head_dims_step = meta["head_dims_step"]
    head_weights_used = meta["head_weights_used"]

    model.train()
    t0 = time.time()
    data_iter = iter(loader)

    for step in range(start_step, cfg.max_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        obs = get_obs_dict(batch)

        pc = get_point_cloud(obs).to(device, non_blocking=True)
        B = pc.shape[0]

        joint = get_joint(obs, B=B, device=device).to(device, non_blocking=True)

        tactile = get_tactile(obs)
        tactile = tactile.to(device, non_blocking=True) if tactile is not None else None
        tactile = preprocess_tactile(tactile, mode=cfg.tactile_mode)

        action_chunk = get_action_chunk(batch)
        action_chunk = {k: v.to(device, non_blocking=True) for k, v in action_chunk.items()} if isinstance(action_chunk, dict) else action_chunk.to(device, non_blocking=True)

        action_mask = get_action_mask(batch)
        action_mask = action_mask.to(device, non_blocking=True) if action_mask is not None else None

        # forward
        preds_flat = model(joint=joint, point_cloud=pc, tactile=tactile)

        # reshape preds to (B, T, d_step)
        preds = {}
        for hn, d_step in zip(head_names, head_dims_step):
            preds[hn] = preds_flat[hn].view(B, cfg.chunk_len, d_step)

        # targets
        if isinstance(action_chunk, dict):
            targets = {hn: action_chunk[hn] for hn in head_names}
            for hn, d_step in zip(head_names, head_dims_step):
                if targets[hn].dim() == 2:
                    targets[hn] = targets[hn].unsqueeze(1).expand(B, cfg.chunk_len, d_step)
        else:
            if action_chunk.dim() == 2:
                action_chunk = action_chunk.unsqueeze(1).expand(B, cfg.chunk_len, action_chunk.shape[-1])
            BT = B * cfg.chunk_len
            action_flat = action_chunk.reshape(BT, -1)
            targets_flat = split_action_vector(action_flat, head_dims_step, head_names)
            targets = {hn: targets_flat[hn].view(B, cfg.chunk_len, d_step) for hn, d_step in zip(head_names, head_dims_step)}

        total_loss, loss_by_head = compute_losses(
            preds=preds,
            targets=targets,
            head_names=head_names,
            head_weights=head_weights_used,
            action_mask=action_mask,
        )

        opt.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm) if cfg.grad_clip_norm > 0 else 0.0
        opt.step()

        if (step % cfg.log_every) == 0:
            dt = time.time() - t0
            steps_per_sec = (step - start_step + 1) / max(dt, 1e-9)

            log_dict = {
                "train/step": step,
                "train/loss_total": float(total_loss.detach().cpu()),
                "train/grad_norm": float(grad_norm) if not isinstance(grad_norm, float) else grad_norm,
                "perf/steps_per_sec": steps_per_sec,
                "data/batch_size": B,
                "data/num_points": int(pc.shape[1]),
                "data/has_tactile": 0 if tactile is None else 1,
                "data/tactile_mode": cfg.tactile_mode,
            }
            for hn in head_names:
                log_dict[f"train/loss_{hn}"] = float(loss_by_head[hn].detach().cpu())

            if action_mask is not None:
                valid = (action_mask > 0.5) if action_mask.dtype != torch.bool else action_mask
                log_dict["data/mask_valid_frac"] = float(valid.float().mean().detach().cpu())

            wandb.log(log_dict, step=step)

        if (step > 0) and (step % cfg.ckpt_every == 0):
            ckpt_path = os.path.join(cfg.save_dir, f"ckpt_step_{step}.pt")
            save_ckpt(ckpt_path, model, opt, step=step, meta=meta)

            art = wandb.Artifact(name=f"checkpoint-{wandb.run.id}", type="model")
            art.add_file(ckpt_path)
            wandb.log_artifact(art)

    final_path = os.path.join(cfg.save_dir, f"ckpt_step_{cfg.max_steps}.pt")
    save_ckpt(final_path, model, opt, step=cfg.max_steps, meta=meta)
    wandb.finish()


if __name__ == "__main__":
    cfg = TrainConfig(
        obs_root="path/to/obs_root",
        tactile_root="path/to/tactile_root",
        act_root="path/to/act_root",
        wandb_project="rh-pointcloud-chunk",
        wandb_run_name=None,
        groups=("n2","n5","lh","rh"),          # or ("n2","n5","lh","rh")
        tactile_mode="first",    # safest default
    )
    main(cfg)
