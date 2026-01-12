"""
train_triplane_vla.py

Supervised behavior cloning for:
  MultiStageTactilePointCloudTransformer
with:
  - point cloud + tactile + object id inputs
  - ground truth robot actions + per-step stage ids

This script assumes your raw data is episodic:
  Each episode contains sequences:
    point_xyz_seq[t], tactile_xyz_seq[t], actions_seq[t], stage_id_seq[t]
  plus a single object_id for the whole episode (common).

We wrap it into "stage-aligned chunks" for training:
  At a valid start time t0 where stage is constant for at least K_stage steps,
  we build a training sample:
    obs at t0 + stage_id(t0) + object_id + action_chunk (length Kmax padded)

Important:
- Your model's decoder uses fixed chunk size per stage (stage_chunk_sizes).
- The model skeleton assumes a batch forwarded together has a uniform stage;
  so we group the batch by stage before calling model.forward_macro_step().

Usage (example):
  python train_triplane_vla.py \
    --data_root /path/to/data \
    --epochs 50 --batch_size 16 --lr 3e-4 \
    --num_objects 20 \
    --stage0_chunk 2 --stage1_chunk 8 --max_chunk_size 16

Replace EpisodeDataset.__getitem__ with your real loader.

"""

from __future__ import annotations

import os
import glob
import json
import time
import argparse
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ---- import your model ----
from model_triplane_vla import (
    ModelConfig,
    TriPlaneConfig,
    TokenizerConfig,
    EncoderConfig,
    DecoderConfig,
    MultiStageTactilePointCloudTransformer,
)


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def make_object_onehot(object_id: torch.Tensor, num_objects: int) -> torch.Tensor:
    # object_id: [B]
    return F.one_hot(object_id, num_classes=num_objects).float()


@torch.no_grad()
def compute_aabb_from_points(point_xyz: torch.Tensor, pad_ratio: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    point_xyz: [B,N,3]
    returns:
      aabb_min, aabb_max: [B,3]
    """
    mn = point_xyz.amin(dim=1)  # [B,3]
    mx = point_xyz.amax(dim=1)  # [B,3]
    center = 0.5 * (mn + mx)
    half = 0.5 * (mx - mn)
    half = half * (1.0 + pad_ratio) + 1e-6
    return center - half, center + half


def pad_actions_to_Kmax(actions: torch.Tensor, Kmax: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    actions: [L, da]
    returns:
      padded: [Kmax, da]
      mask:   [Kmax] boolean (True for valid)
    """
    L, da = actions.shape
    padded = torch.zeros((Kmax, da), dtype=actions.dtype)
    mask = torch.zeros((Kmax,), dtype=torch.bool)
    take = min(L, Kmax)
    if take > 0:
        padded[:take] = actions[:take]
        mask[:take] = True
    return padded, mask


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    pred/target: [B,K,da]
    mask:        [B,K] boolean (True valid)
    """
    # avoid empty mask
    denom = mask.sum().clamp(min=1).float()
    err = (pred - target) ** 2  # [B,K,da]
    err = err.mean(dim=-1)      # [B,K]
    return (err[mask].sum() / denom)


# -----------------------------
# Raw episodic dataset (YOU implement)
# -----------------------------

class EpisodeDataset(Dataset):
    """
    You must implement this to load your episodes.

    Each __getitem__ should return a dict containing:

    Required:
      - "object_id": int
      - "actions":   FloatTensor [T, action_dim]
      - "stage_id":  LongTensor  [T]  (stage id for each action step)

    Observations:
      Either time-varying or static (both supported by wrapper):
        - "point_xyz":  FloatTensor [T,N,3] OR [N,3]
        - "point_feats":FloatTensor [T,N,dp] OR [N,dp] OR None (optional)
        - "tactile_xyz":FloatTensor [T,M,3] OR [M,3] OR None
        - "tactile_feats":FloatTensor [T,M,dt] OR [M,dt] OR None

    Notes:
    - If you don't have point_feats, set dp=0 in config and return None.
    - If you don't have tactile, return None for tactile_xyz/tactile_feats.
    - Stage ids should be in [0, num_stages-1]. For your binary stage, that's {0,1}.
    """

    def __init__(self, data_root: str, split: str) -> None:
        super().__init__()
        self.data_root = data_root
        self.split = split

        # Example: episodes stored as .pt under data_root/split/*.pt
        self.files = sorted(glob.glob(os.path.join(data_root, split, "*.pt")))
        if len(self.files) == 0:
            raise RuntimeError(f"No episode files found at: {os.path.join(data_root, split, '*.pt')}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.files[idx]
        ep = torch.load(path, map_location="cpu")

        # EXPECTED keys in ep (you can rename here):
        # ep["object_id"] : int
        # ep["actions"]   : [T,da]
        # ep["stage_id"]  : [T]
        # ep["point_xyz"] : [T,N,3] or [N,3]
        # ep.get("point_feats")
        # ep.get("tactile_xyz"), ep.get("tactile_feats")

        return ep


# -----------------------------
# Wrapper: converts episodes -> stage-aligned chunk samples
# -----------------------------

class StageChunkDataset(Dataset):
    """
    Converts episodic (actions, stage_id per step) into per-sample stage chunks.

    Produces items:
      - point_xyz:    [N,3]
      - point_feats:  [N,dp] (or [N,0] if dp=0)
      - tactile_xyz:  [M,3] or None
      - tactile_feats:[M,dt] or None
      - object_id:    int
      - stage:        int
      - actions_gt:   [Kmax, action_dim] padded
      - actions_mask: [Kmax] boolean valid
    """

    def __init__(
        self,
        episodes: EpisodeDataset,
        action_dim: int,
        Kmax: int,
        stage_chunk_sizes: Tuple[int, int],
        dp: int,
        dt: int,
    ) -> None:
        super().__init__()
        self.episodes = episodes
        self.action_dim = action_dim
        self.Kmax = Kmax
        self.stage_chunk_sizes = stage_chunk_sizes
        self.dp = dp
        self.dt = dt

        # Build an index of valid (episode_idx, t0) where stage stays constant for >= K_stage steps.
        self.index: List[Tuple[int, int]] = []
        for eidx in range(len(self.episodes)):
            ep = self.episodes[eidx]
            stage_id = torch.as_tensor(ep["stage_id"], dtype=torch.long)  # [T]
            T = int(stage_id.shape[0])

            # run-length encoding
            t = 0
            while t < T:
                s = int(stage_id[t].item())
                t2 = t + 1
                while t2 < T and int(stage_id[t2].item()) == s:
                    t2 += 1
                seg_len = t2 - t
                if s in (0, 1):
                    K_stage = self.stage_chunk_sizes[s]
                    # any start inside [t, t2-K_stage] is valid
                    last_start = t2 - K_stage
                    for t0 in range(t, max(t, last_start) + 1):
                        self.index.append((eidx, t0))
                # advance
                t = t2

        if len(self.index) == 0:
            raise RuntimeError("No valid stage-chunks found. Check stage ids and chunk sizes.")

    def __len__(self) -> int:
        return len(self.index)

    def _get_obs_at(self, x: Optional[torch.Tensor], t0: int) -> Optional[torch.Tensor]:
        if x is None:
            return None
        if x.dim() == 2:
            return x  # [N,3] or [N,dp] static
        if x.dim() == 3:
            return x[t0]  # [N,3] time-varying
        raise ValueError(f"Unsupported obs tensor shape: {tuple(x.shape)}")

    def __getitem__(self, i: int) -> Dict[str, Any]:
        eidx, t0 = self.index[i]
        ep = self.episodes[eidx]

        object_id = int(ep["object_id"])
        actions = torch.as_tensor(ep["actions"], dtype=torch.float32)  # [T,da]
        stage_id = torch.as_tensor(ep["stage_id"], dtype=torch.long)   # [T]
        stage = int(stage_id[t0].item())

        assert actions.shape[-1] == self.action_dim, "action_dim mismatch"

        # Observation at t0
        point_xyz = self._get_obs_at(torch.as_tensor(ep["point_xyz"], dtype=torch.float32), t0)  # [N,3]

        # point feats optional
        point_feats_raw = ep.get("point_feats", None)
        if self.dp == 0:
            point_feats = torch.zeros((point_xyz.shape[0], 0), dtype=torch.float32)
        else:
            if point_feats_raw is None:
                raise ValueError("dp>0 but episode has no point_feats. Set dp=0 or provide point_feats.")
            point_feats = self._get_obs_at(torch.as_tensor(point_feats_raw, dtype=torch.float32), t0)
            assert point_feats.shape[-1] == self.dp

        # tactile optional
        tactile_xyz_raw = ep.get("tactile_xyz", None)
        tactile_feats_raw = ep.get("tactile_feats", None)
        tactile_xyz = None
        tactile_feats = None
        if tactile_xyz_raw is not None and tactile_feats_raw is not None:
            tactile_xyz = self._get_obs_at(torch.as_tensor(tactile_xyz_raw, dtype=torch.float32), t0)
            tactile_feats = self._get_obs_at(torch.as_tensor(tactile_feats_raw, dtype=torch.float32), t0)
            assert tactile_feats.shape[-1] == self.dt

        # GT action chunk (start at t0)
        gt_slice = actions[t0:]  # [T-t0,da]
        actions_gt, actions_mask = pad_actions_to_Kmax(gt_slice, self.Kmax)

        return dict(
            point_xyz=point_xyz,               # [N,3]
            point_feats=point_feats,           # [N,dp]
            tactile_xyz=tactile_xyz,           # [M,3] or None
            tactile_feats=tactile_feats,       # [M,dt] or None
            object_id=object_id,               # int
            stage=stage,                       # int
            actions_gt=actions_gt,             # [Kmax,da]
            actions_mask=actions_mask,         # [Kmax]
        )


# -----------------------------
# Collate
# -----------------------------

def collate_stage_chunks(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Pads tactile (M) to max in batch.
    Assumes point cloud N is fixed (recommended). If variable, you should pre-sample to fixed N in EpisodeDataset.
    """
    B = len(batch)

    # stack points (require fixed N)
    N0 = batch[0]["point_xyz"].shape[0]
    for b in batch:
        if b["point_xyz"].shape[0] != N0:
            raise ValueError("Variable N detected. Please downsample point cloud to fixed N in your dataset.")

    point_xyz = torch.stack([b["point_xyz"] for b in batch], dim=0)         # [B,N,3]
    point_feats = torch.stack([b["point_feats"] for b in batch], dim=0)     # [B,N,dp]

    # stages / object
    stage = torch.tensor([b["stage"] for b in batch], dtype=torch.long)     # [B]
    object_id = torch.tensor([b["object_id"] for b in batch], dtype=torch.long)  # [B]

    # actions
    actions_gt = torch.stack([b["actions_gt"] for b in batch], dim=0)       # [B,Kmax,da]
    actions_mask = torch.stack([b["actions_mask"] for b in batch], dim=0)   # [B,Kmax]

    # tactile pad
    # represent None as empty [0,3]/[0,dt]
    tactile_xyz_list = []
    tactile_feats_list = []
    for b in batch:
        if b["tactile_xyz"] is None or b["tactile_feats"] is None:
            tactile_xyz_list.append(torch.zeros((0, 3), dtype=torch.float32))
            tactile_feats_list.append(torch.zeros((0, 0), dtype=torch.float32))  # dt unknown here
        else:
            tactile_xyz_list.append(b["tactile_xyz"])
            tactile_feats_list.append(b["tactile_feats"])

    # infer dt from first non-empty
    dt = None
    for tf in tactile_feats_list:
        if tf.numel() > 0:
            dt = tf.shape[-1]
            break
    if dt is None:
        dt = 0

    Mmax = max([t.shape[0] for t in tactile_xyz_list]) if B > 0 else 0
    tactile_count = torch.tensor([t.shape[0] for t in tactile_xyz_list], dtype=torch.long)  # [B]

    if Mmax > 0:
        tactile_xyz = torch.zeros((B, Mmax, 3), dtype=torch.float32)
        tactile_feats = torch.zeros((B, Mmax, dt), dtype=torch.float32)
        for i in range(B):
            m = tactile_xyz_list[i].shape[0]
            if m > 0:
                tactile_xyz[i, :m] = tactile_xyz_list[i]
                tactile_feats[i, :m] = tactile_feats_list[i]
    else:
        tactile_xyz = torch.zeros((B, 0, 3), dtype=torch.float32)
        tactile_feats = torch.zeros((B, 0, dt), dtype=torch.float32)

    return dict(
        point_xyz=point_xyz,
        point_feats=point_feats,
        tactile_xyz=tactile_xyz,
        tactile_feats=tactile_feats,
        tactile_count=tactile_count,
        stage=stage,
        object_id=object_id,
        actions_gt=actions_gt,
        actions_mask=actions_mask,
    )


# -----------------------------
# Train / Eval
# -----------------------------

def run_one_epoch(
    model: MultiStageTactilePointCloudTransformer,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    num_objects: int,
    max_chunk_size: int,
    grad_clip: float,
    use_amp: bool,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    total = 0
    sum_loss = 0.0

    for batch in loader:
        # move
        point_xyz = batch["point_xyz"].to(device)
        point_feats = batch["point_feats"].to(device)
        tactile_xyz = batch["tactile_xyz"].to(device)
        tactile_feats = batch["tactile_feats"].to(device)
        tactile_count = batch["tactile_count"].to(device)
        stage = batch["stage"].to(device)
        object_id = batch["object_id"].to(device)
        actions_gt = batch["actions_gt"].to(device)
        actions_mask = batch["actions_mask"].to(device)

        B = point_xyz.shape[0]
        total += B

        object_onehot = make_object_onehot(object_id, num_objects).to(device)

        # auto AABB from point cloud
        aabb_min, aabb_max = compute_aabb_from_points(point_xyz)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        # Important:
        # Your model skeleton requires uniform stage within the forwarded batch,
        # so we group by stage (and also by has_tactile to avoid passing fake tactile for empty).
        unique_stages = torch.unique(stage).tolist()

        with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
            loss_batch = 0.0
            groups = 0

            for s in unique_stages:
                stage_mask = (stage == s)
                if stage_mask.sum() == 0:
                    continue

                # split into tactile-present / tactile-empty
                has_tac = tactile_count > 0

                for tac_flag in (False, True):
                    idx = stage_mask & (has_tac == tac_flag)
                    if idx.sum() == 0:
                        continue

                    # gather group tensors
                    px = point_xyz[idx]
                    pf = point_feats[idx]
                    st = stage[idx]
                    oo = object_onehot[idx]
                    amin = aabb_min[idx]
                    amax = aabb_max[idx]
                    agt = actions_gt[idx]       # [Bg,Kmax,da]
                    amask = actions_mask[idx]   # [Bg,Kmax]

                    if tac_flag:
                        tx = tactile_xyz[idx]
                        tf = tactile_feats[idx]
                    else:
                        tx = None
                        tf = None

                    Bg = px.shape[0]
                    planes = model.init_planes(Bg, device=device, dtype=px.dtype)

                    out = model.forward_macro_step(
                        point_xyz=px,
                        point_feats=pf,
                        tactile_xyz=tx,
                        tactile_feats=tf,
                        stage=st,
                        object_onehot=oo,
                        aabb_min=amin,
                        aabb_max=amax,
                        planes=planes,
                        actions_gt=agt,
                        teacher_forcing=True,
                    )

                    pred = out["pred_actions"]     # [Bg,K,da]
                    K = int(out["K"].item())

                    # loss on first K, masked by validity (in case you padded)
                    predK = pred[:, :K, :]
                    gtK = agt[:, :K, :]
                    maskK = amask[:, :K]

                    loss_g = masked_mse(predK, gtK, maskK)
                    loss_batch = loss_batch + loss_g
                    groups += 1

            if groups > 0:
                loss_batch = loss_batch / float(groups)

        if is_train:
            scaler.scale(loss_batch).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

        sum_loss += float(loss_batch.detach().item()) * B

    return {"loss": sum_loss / max(1, total)}


def save_ckpt(path: str, model: nn.Module, opt: torch.optim.Optimizer, epoch: int, cfg: ModelConfig, args: argparse.Namespace):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "cfg": asdict(cfg),
            "args": vars(args),
        },
        path,
    )


# -----------------------------
# Config
# -----------------------------

def build_cfg(args: argparse.Namespace) -> ModelConfig:
    cfg = ModelConfig(
        plane=TriPlaneConfig(res=args.plane_res, channels=args.plane_channels),
        tokenizer=TokenizerConfig(
            point_feat_dim=args.point_feat_dim,
            tactile_feat_dim=args.tactile_feat_dim,
            d_model=args.d_model,
            include_tactile_tokens=args.include_tactile_tokens,
            k_tactile=args.k_tactile,
            tactile_temp=args.tactile_temp,
        ),
        encoder=EncoderConfig(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.enc_layers,
            dim_feedforward=args.ffn_dim,
            dropout=args.dropout,
        ),
        decoder=DecoderConfig(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.dec_layers,
            dim_feedforward=args.ffn_dim,
            dropout=args.dropout,
            action_dim=args.action_dim,
            max_chunk_size=args.max_chunk_size,
            stage_chunk_sizes=(args.stage0_chunk, args.stage1_chunk),
            use_adaptive_chunking=args.adaptive_chunking,
            candidate_chunk_sizes=tuple(args.candidate_chunks),
        ),
        num_stages=args.num_stages,
        num_objects=args.num_objects,
        writer_cond_dim=None,  # auto = 3*d_model
    )
    return cfg


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)

    # training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=0)

    # output
    p.add_argument("--out_dir", type=str, default="runs/triplane_vla")
    p.add_argument("--save_every", type=int, default=1)

    # model dims
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--enc_layers", type=int, default=4)
    p.add_argument("--dec_layers", type=int, default=4)
    p.add_argument("--ffn_dim", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)

    # tri-plane
    p.add_argument("--plane_res", type=int, default=64)
    p.add_argument("--plane_channels", type=int, default=32)

    # input dims
    p.add_argument("--point_feat_dim", type=int, default=0)   # set 0 if only xyz
    p.add_argument("--tactile_feat_dim", type=int, default=0) # set 0 if tactile has only xyz (rare)
    p.add_argument("--action_dim", type=int, default=7)

    # tactile tokenization
    p.add_argument("--include_tactile_tokens", action="store_true")
    p.add_argument("--k_tactile", type=int, default=8)
    p.add_argument("--tactile_temp", type=float, default=0.05)

    # stages / objects
    p.add_argument("--num_stages", type=int, default=2)
    p.add_argument("--num_objects", type=int, default=10)
    p.add_argument("--stage0_chunk", type=int, default=2)
    p.add_argument("--stage1_chunk", type=int, default=8)

    # chunking
    p.add_argument("--max_chunk_size", type=int, default=16)
    p.add_argument("--adaptive_chunking", action="store_true")
    p.add_argument("--candidate_chunks", type=int, nargs="+", default=[1, 2, 4, 8, 16])

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = build_cfg(args)

    # Save cfg/args for reproducibility
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump({"cfg": asdict(cfg), "args": vars(args)}, f, indent=2)

    model = MultiStageTactilePointCloudTransformer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Build datasets
    ep_train = EpisodeDataset(args.data_root, split="train")
    ep_val = EpisodeDataset(args.data_root, split="val")

    train_ds = StageChunkDataset(
        episodes=ep_train,
        action_dim=args.action_dim,
        Kmax=args.max_chunk_size,
        stage_chunk_sizes=(args.stage0_chunk, args.stage1_chunk),
        dp=args.point_feat_dim,
        dt=args.tactile_feat_dim,
    )
    val_ds = StageChunkDataset(
        episodes=ep_val,
        action_dim=args.action_dim,
        Kmax=args.max_chunk_size,
        stage_chunk_sizes=(args.stage0_chunk, args.stage1_chunk),
        dp=args.point_feat_dim,
        dt=args.tactile_feat_dim,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_stage_chunks,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_stage_chunks,
        drop_last=False,
    )

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_stats = run_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=opt,
            num_objects=args.num_objects,
            max_chunk_size=args.max_chunk_size,
            grad_clip=args.grad_clip,
            use_amp=args.amp,
        )
        val_stats = run_one_epoch(
            model=model,
            loader=val_loader,
            device=device,
            optimizer=None,
            num_objects=args.num_objects,
            max_chunk_size=args.max_chunk_size,
            grad_clip=0.0,
            use_amp=False,
        )

        dt = time.time() - t0
        log = {
            "epoch": epoch,
            "time_sec": round(dt, 2),
            "train_loss": round(train_stats["loss"], 6),
            "val_loss": round(val_stats["loss"], 6),
        }
        print(json.dumps(log))

        if epoch % args.save_every == 0:
            save_ckpt(
                os.path.join(args.out_dir, f"ckpt_epoch_{epoch}.pt"),
                model=model,
                opt=opt,
                epoch=epoch,
                cfg=cfg,
                args=args,
            )

        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            save_ckpt(
                os.path.join(args.out_dir, "ckpt_best.pt"),
                model=model,
                opt=opt,
                epoch=epoch,
                cfg=cfg,
                args=args,
            )

    print("Done. Best val loss:", best_val)


if __name__ == "__main__":
    main()
