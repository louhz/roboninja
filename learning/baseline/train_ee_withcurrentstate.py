"""
train_triplane_vla_dual_policy_postprocess_robotstate.py

Dual-policy supervised BC with differentiable post-process chunk resampling, PLUS robot proprioception:
- robot_state (or arm_state + hand_state) is read at t0 and passed into forward_macro_step(..., robot_state=...)
- base model (model_triplane_vla.py) must include robot_state support (see updated file above)

Episode format (.pt):
Required:
  - object_id: int
  - stage_id: LongTensor [T]
  - point_xyz: [T,N,3] or [N,3]
  - (optional) point_feats: [T,N,dp] or [N,dp]
  - (optional) tactile_xyz/tactile_feats similarly

Robot state (one of):
  Option A: robot_state: [T,D_robot] or [D_robot]
  Option B: arm_state: [T,D_arm] or [D_arm] AND hand_state: [T,D_hand] or [D_hand]
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
    return F.one_hot(object_id, num_classes=num_objects).float()

@torch.no_grad()
def compute_aabb_from_points(point_xyz: torch.Tensor, pad_ratio: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
    mn = point_xyz.amin(dim=1)
    mx = point_xyz.amax(dim=1)
    center = 0.5 * (mn + mx)
    half = 0.5 * (mx - mn)
    half = half * (1.0 + pad_ratio) + 1e-6
    return center - half, center + half

def pad_actions_to_Kmax(actions: torch.Tensor, Kmax: int) -> Tuple[torch.Tensor, torch.Tensor]:
    L, da = actions.shape
    padded = torch.zeros((Kmax, da), dtype=actions.dtype)
    mask = torch.zeros((Kmax,), dtype=torch.bool)
    take = min(L, Kmax)
    if take > 0:
        padded[:take] = actions[:take]
        mask[:take] = True
    return padded, mask

def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp(min=1).float()
    err = (pred - target) ** 2
    err = err.mean(dim=-1)  # [B,K]
    return err[mask].sum() / denom


# -----------------------------
# Differentiable post-processing: interpolation / extrapolation
# -----------------------------

class DifferentiableChunkResampler(nn.Module):
    def __init__(self, K_base: int, K_out_max: int, mode: str = "auto") -> None:
        super().__init__()
        assert mode in ("auto", "stretch", "extrapolate")
        self.K_base = int(K_base)
        self.K_out_max = int(K_out_max)
        self.mode = mode

    def forward(self, x: torch.Tensor, target_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.dim() == 3 and x.shape[1] == self.K_base
        B, K, da = x.shape
        device = x.device
        dtype = x.dtype
        K_out = self.K_out_max

        tgt = target_len.to(device=device).long().clamp(min=1, max=K_out)  # [B]

        j_long = torch.arange(K_out, device=device, dtype=torch.long)[None, :]  # [1,K_out]
        j = j_long.to(dtype=dtype).expand(B, -1)  # [B,K_out]

        mask_target = j_long.expand(B, -1) < tgt[:, None]  # [B,K_out] bool

        if self.mode == "extrapolate":
            t = j
        else:
            denom = (tgt - 1).clamp(min=1).to(dtype=dtype)
            scale = (float(K - 1) / denom)                  # [B]
            t_stretch = j * scale[:, None]                  # [B,K_out]
            if self.mode == "stretch":
                t = t_stretch
            else:
                use_stretch = (tgt <= K)                    # [B]
                t = torch.where(use_stretch[:, None], t_stretch, j)

        inp = x.permute(0, 2, 1).unsqueeze(2)  # [B,da,1,K]

        t_clamped = t.clamp(min=0.0, max=float(K - 1))
        x_norm = (t_clamped / float(K - 1)) * 2.0 - 1.0  # [B,K_out]
        y_norm = torch.zeros_like(x_norm)

        grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(1)  # [B,1,K_out,2]

        y_interp = F.grid_sample(
            inp, grid, mode="bilinear", padding_mode="border", align_corners=True
        )  # [B,da,1,K_out]
        y = y_interp.squeeze(2).permute(0, 2, 1).contiguous()  # [B,K_out,da]

        if K >= 2:
            last = x[:, K - 1, :]
            prev = x[:, K - 2, :]
            slope_last = last - prev

            first = x[:, 0, :]
            nxt = x[:, 1, :]
            slope_first = nxt - first
        else:
            last = x[:, 0, :]
            slope_last = torch.zeros_like(last)
            first = x[:, 0, :]
            slope_first = torch.zeros_like(first)

        over = (t > float(K - 1))
        if over.any():
            dt_over = (t - float(K - 1)).clamp(min=0.0)
            y_over = last[:, None, :] + dt_over.unsqueeze(-1) * slope_last[:, None, :]
            y = torch.where(over.unsqueeze(-1), y_over, y)

        under = (t < 0.0)
        if under.any():
            dt_under = (t - 0.0).clamp(max=0.0)
            y_under = first[:, None, :] + dt_under.unsqueeze(-1) * slope_first[:, None, :]
            y = torch.where(under.unsqueeze(-1), y_under, y)

        return y, mask_target


# -----------------------------
# Dataset
# -----------------------------

class EpisodeDataset(Dataset):
    def __init__(self, data_root: str, split: str) -> None:
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.files = sorted(glob.glob(os.path.join(data_root, split, "*.pt")))
        if len(self.files) == 0:
            raise RuntimeError(f"No episode files found at: {os.path.join(data_root, split, '*.pt')}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return torch.load(self.files[idx], map_location="cpu")


def _get_obs_at(x: Optional[torch.Tensor], t0: int) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if x.dim() == 2:
        return x
    if x.dim() == 3:
        return x[t0]
    raise ValueError(f"Unsupported obs tensor shape: {tuple(x.shape)}")


def _get_vec_at(x: Optional[torch.Tensor], t0: int) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if x.dim() == 1:
        return x
    if x.dim() == 2:
        return x[t0]
    raise ValueError(f"Unsupported vector tensor shape: {tuple(x.shape)}")


def _get_action_pair(
    ep: Dict[str, Any],
    action_dim_ee: int,
    action_dim_dex: int,
    actions_key: str,
    actions_ee_key: str,
    actions_dex_key: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if actions_ee_key in ep and actions_dex_key in ep:
        a_ee = torch.as_tensor(ep[actions_ee_key], dtype=torch.float32)
        a_dx = torch.as_tensor(ep[actions_dex_key], dtype=torch.float32)
        if a_ee.shape[-1] != action_dim_ee:
            raise ValueError(f"{actions_ee_key} last dim {a_ee.shape[-1]} != action_dim_ee={action_dim_ee}")
        if a_dx.shape[-1] != action_dim_dex:
            raise ValueError(f"{actions_dex_key} last dim {a_dx.shape[-1]} != action_dim_dex={action_dim_dex}")
        return a_ee, a_dx

    if actions_key in ep:
        a = torch.as_tensor(ep[actions_key], dtype=torch.float32)
        if a.shape[-1] == (action_dim_ee + action_dim_dex):
            a_ee = a[..., :action_dim_ee]
            a_dx = a[..., action_dim_ee:action_dim_ee + action_dim_dex]
            return a_ee, a_dx

        raise ValueError(
            f"Episode has '{actions_key}' with last dim {a.shape[-1]}, "
            f"but expected action_dim_ee+action_dim_dex = {action_dim_ee+action_dim_dex}."
        )

    raise ValueError(f"Episode must contain either ('{actions_ee_key}','{actions_dex_key}') or '{actions_key}'.")


class StageSegmentChunkDataset(Dataset):
    """
    Produces training samples that DO NOT cross stage boundaries.
    Adds robot_state at t0 (current arm/hand state) as a separate input tensor.
    """
    def __init__(
        self,
        episodes: EpisodeDataset,
        action_dim_ee: int,
        action_dim_dex: int,
        K_gt_max: int,
        dp: int,
        dt: int,
        min_steps_in_stage: int,
        actions_key: str,
        actions_ee_key: str,
        actions_dex_key: str,
        # robot state
        robot_state_dim: int,
        arm_state_dim: int,
        hand_state_dim: int,
        robot_state_key: str,
        arm_state_key: str,
        hand_state_key: str,
    ) -> None:
        super().__init__()
        self.episodes = episodes
        self.action_dim_ee = int(action_dim_ee)
        self.action_dim_dex = int(action_dim_dex)
        self.K_gt_max = int(K_gt_max)
        self.dp = int(dp)
        self.dt = int(dt)
        self.min_steps_in_stage = int(min_steps_in_stage)
        self.actions_key = actions_key
        self.actions_ee_key = actions_ee_key
        self.actions_dex_key = actions_dex_key

        self.robot_state_dim = int(robot_state_dim)
        self.arm_state_dim = int(arm_state_dim)
        self.hand_state_dim = int(hand_state_dim)
        self.robot_state_key = str(robot_state_key)
        self.arm_state_key = str(arm_state_key)
        self.hand_state_key = str(hand_state_key)

        self.index: List[Tuple[int, int, int]] = []
        for eidx in range(len(self.episodes)):
            ep = self.episodes[eidx]
            stage_id = torch.as_tensor(ep["stage_id"], dtype=torch.long)
            T = int(stage_id.shape[0])

            t = 0
            while t < T:
                s = int(stage_id[t].item())
                t2 = t + 1
                while t2 < T and int(stage_id[t2].item()) == s:
                    t2 += 1

                seg_len = t2 - t
                if seg_len >= self.min_steps_in_stage:
                    last_start = t2 - self.min_steps_in_stage
                    for t0 in range(t, last_start + 1):
                        self.index.append((eidx, t0, t2))
                t = t2

        if len(self.index) == 0:
            raise RuntimeError("No valid stage-segment samples found. Try reducing --min_steps_in_stage.")

    def __len__(self) -> int:
        return len(self.index)

    def _read_robot_state(self, ep: Dict[str, Any], t0: int) -> torch.Tensor:
        if self.robot_state_dim <= 0:
            return torch.zeros((0,), dtype=torch.float32)

        # Option B: pre-concatenated
        if self.robot_state_key in ep:
            rs = torch.as_tensor(ep[self.robot_state_key], dtype=torch.float32)
            rs = _get_vec_at(rs, t0)
            assert rs is not None and rs.dim() == 1
            if rs.shape[-1] != self.robot_state_dim:
                raise ValueError(f"{self.robot_state_key} dim {rs.shape[-1]} != robot_state_dim {self.robot_state_dim}")
            return rs

        # Option A: arm + hand
        parts: List[torch.Tensor] = []
        if self.arm_state_dim > 0:
            if self.arm_state_key not in ep:
                raise ValueError(f"arm_state_dim>0 but episode missing '{self.arm_state_key}'")
            arm = torch.as_tensor(ep[self.arm_state_key], dtype=torch.float32)
            arm = _get_vec_at(arm, t0)
            assert arm is not None and arm.dim() == 1
            if arm.shape[-1] != self.arm_state_dim:
                raise ValueError(f"{self.arm_state_key} dim {arm.shape[-1]} != arm_state_dim {self.arm_state_dim}")
            parts.append(arm)

        if self.hand_state_dim > 0:
            if self.hand_state_key not in ep:
                raise ValueError(f"hand_state_dim>0 but episode missing '{self.hand_state_key}'")
            hand = torch.as_tensor(ep[self.hand_state_key], dtype=torch.float32)
            hand = _get_vec_at(hand, t0)
            assert hand is not None and hand.dim() == 1
            if hand.shape[-1] != self.hand_state_dim:
                raise ValueError(f"{self.hand_state_key} dim {hand.shape[-1]} != hand_state_dim {self.hand_state_dim}")
            parts.append(hand)

        if len(parts) == 0:
            raise ValueError(
                f"robot_state_dim={self.robot_state_dim} but no '{self.robot_state_key}' "
                f"and arm_state_dim/hand_state_dim not configured."
            )

        rs = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        if rs.shape[-1] != self.robot_state_dim:
            raise ValueError(f"concat arm+hand dim {rs.shape[-1]} != robot_state_dim {self.robot_state_dim}")
        return rs

    def __getitem__(self, i: int) -> Dict[str, Any]:
        eidx, t0, t_end = self.index[i]
        ep = self.episodes[eidx]

        object_id = int(ep["object_id"])
        stage_id = torch.as_tensor(ep["stage_id"], dtype=torch.long)
        stage = int(stage_id[t0].item())

        actions_ee, actions_dex = _get_action_pair(
            ep,
            action_dim_ee=self.action_dim_ee,
            action_dim_dex=self.action_dim_dex,
            actions_key=self.actions_key,
            actions_ee_key=self.actions_ee_key,
            actions_dex_key=self.actions_dex_key,
        )

        point_xyz = _get_obs_at(torch.as_tensor(ep["point_xyz"], dtype=torch.float32), t0)

        point_feats_raw = ep.get("point_feats", None)
        if self.dp == 0:
            point_feats = torch.zeros((point_xyz.shape[0], 0), dtype=torch.float32)
        else:
            if point_feats_raw is None:
                raise ValueError("point_feat_dim>0 but episode has no point_feats.")
            point_feats = _get_obs_at(torch.as_tensor(point_feats_raw, dtype=torch.float32), t0)
            assert point_feats.shape[-1] == self.dp

        tactile_xyz_raw = ep.get("tactile_xyz", None)
        tactile_feats_raw = ep.get("tactile_feats", None)
        tactile_xyz = None
        tactile_feats = None
        if tactile_xyz_raw is not None and tactile_feats_raw is not None:
            tactile_xyz = _get_obs_at(torch.as_tensor(tactile_xyz_raw, dtype=torch.float32), t0)
            tactile_feats = _get_obs_at(torch.as_tensor(tactile_feats_raw, dtype=torch.float32), t0)
            if self.dt > 0:
                assert tactile_feats.shape[-1] == self.dt

        # robot state at t0
        robot_state = self._read_robot_state(ep, t0)

        ee_slice = actions_ee[t0:t_end]
        dx_slice = actions_dex[t0:t_end]

        actions_gt_ee, actions_mask_ee = pad_actions_to_Kmax(ee_slice, self.K_gt_max)
        actions_gt_dex, actions_mask_dex = pad_actions_to_Kmax(dx_slice, self.K_gt_max)

        return dict(
            point_xyz=point_xyz,
            point_feats=point_feats,
            tactile_xyz=tactile_xyz,
            tactile_feats=tactile_feats,
            robot_state=robot_state,
            object_id=object_id,
            stage=stage,
            actions_gt_ee=actions_gt_ee,
            actions_mask_ee=actions_mask_ee,
            actions_gt_dex=actions_gt_dex,
            actions_mask_dex=actions_mask_dex,
        )


def collate_stage_segments_dual(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    B = len(batch)

    N0 = batch[0]["point_xyz"].shape[0]
    for b in batch:
        if b["point_xyz"].shape[0] != N0:
            raise ValueError("Variable N detected. Downsample point clouds to fixed N in EpisodeDataset.")

    point_xyz = torch.stack([b["point_xyz"] for b in batch], dim=0)
    point_feats = torch.stack([b["point_feats"] for b in batch], dim=0)

    stage = torch.tensor([b["stage"] for b in batch], dtype=torch.long)
    object_id = torch.tensor([b["object_id"] for b in batch], dtype=torch.long)

    actions_gt_ee = torch.stack([b["actions_gt_ee"] for b in batch], dim=0)
    actions_mask_ee = torch.stack([b["actions_mask_ee"] for b in batch], dim=0)
    actions_gt_dex = torch.stack([b["actions_gt_dex"] for b in batch], dim=0)
    actions_mask_dex = torch.stack([b["actions_mask_dex"] for b in batch], dim=0)

    # robot state
    robot_state = torch.stack([b["robot_state"] for b in batch], dim=0)  # [B,D_robot] or [B,0]

    # tactile padding
    tactile_xyz_list = []
    tactile_feats_list = []
    for b in batch:
        if b["tactile_xyz"] is None or b["tactile_feats"] is None:
            tactile_xyz_list.append(torch.zeros((0, 3), dtype=torch.float32))
            tactile_feats_list.append(torch.zeros((0, 0), dtype=torch.float32))
        else:
            tactile_xyz_list.append(b["tactile_xyz"])
            tactile_feats_list.append(b["tactile_feats"])

    dt = 0
    for tf in tactile_feats_list:
        if tf.numel() > 0:
            dt = tf.shape[-1]
            break

    Mmax = max([t.shape[0] for t in tactile_xyz_list]) if B > 0 else 0
    tactile_count = torch.tensor([t.shape[0] for t in tactile_xyz_list], dtype=torch.long)

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
        robot_state=robot_state,
        stage=stage,
        object_id=object_id,
        actions_gt_ee=actions_gt_ee,
        actions_mask_ee=actions_mask_ee,
        actions_gt_dex=actions_gt_dex,
        actions_mask_dex=actions_mask_dex,
    )


# -----------------------------
# Object-id -> target chunk length
# -----------------------------

def load_object_chunk_schedule(
    num_objects: int,
    default_len: int,
    json_path: Optional[str],
    list_vals: Optional[List[int]],
) -> torch.Tensor:
    schedule = torch.full((num_objects,), int(default_len), dtype=torch.long)
    if list_vals is not None and len(list_vals) > 0:
        if len(list_vals) != num_objects:
            raise ValueError(f"--object_chunk_sizes must have length num_objects={num_objects}")
        return torch.tensor(list_vals, dtype=torch.long)

    if json_path is None:
        return schedule

    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        if len(data) != num_objects:
            raise ValueError(f"object_chunk_json list must have length num_objects={num_objects}")
        return torch.tensor([int(x) for x in data], dtype=torch.long)

    if isinstance(data, dict):
        for k, v in data.items():
            oid = int(k)
            if 0 <= oid < num_objects:
                schedule[oid] = int(v)
        return schedule

    raise ValueError("object_chunk_json must be a list or dict")


# -----------------------------
# Config builder
# -----------------------------

def build_cfg(args: argparse.Namespace, action_dim: int, K_base: int, robot_state_dim: int) -> ModelConfig:
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
            action_dim=action_dim,
            max_chunk_size=K_base,
            stage_chunk_sizes=(K_base, K_base),
            use_adaptive_chunking=False,
            candidate_chunk_sizes=tuple([K_base]),
        ),
        num_stages=args.num_stages,
        num_objects=args.num_objects,
        robot_state_dim=robot_state_dim,
        use_robot_token=(not args.disable_robot_token),
        fuse_robot_into_ctx=(not args.disable_robot_ctx),
        writer_cond_dim=None,
    )
    return cfg


# -----------------------------
# Loss per-policy
# -----------------------------

def _policy_group_loss(
    model: MultiStageTactilePointCloudTransformer,
    resampler: DifferentiableChunkResampler,
    *,
    idx: torch.Tensor,                 # [B] bool
    point_xyz: torch.Tensor,
    point_feats: torch.Tensor,
    tactile_xyz: torch.Tensor,
    tactile_feats: torch.Tensor,
    tactile_count: torch.Tensor,
    robot_state: torch.Tensor,         # [B,D_robot]
    stage: torch.Tensor,
    object_onehot: torch.Tensor,
    aabb_min: torch.Tensor,
    aabb_max: torch.Tensor,
    actions_gt_full: torch.Tensor,     # [B,K_gt_max,da]
    actions_mask_full: torch.Tensor,   # [B,K_gt_max]
    target_len: torch.Tensor,          # [B] long
    K_base: int,
    teacher_forcing: bool,
) -> torch.Tensor:
    if idx.sum().item() == 0:
        return torch.zeros((), device=point_xyz.device)

    has_tac = tactile_count > 0
    loss_accum = 0.0
    groups = 0

    for tac_flag in (False, True):
        sub = idx & (has_tac == tac_flag)
        if sub.sum().item() == 0:
            continue

        px = point_xyz[sub]
        pf = point_feats[sub]
        st = stage[sub]
        oo = object_onehot[sub]
        amin = aabb_min[sub]
        amax = aabb_max[sub]
        rs = robot_state[sub]
        gt_full = actions_gt_full[sub]
        mask_full = actions_mask_full[sub]
        tgt_len = target_len[sub]

        gt_in = gt_full[:, :K_base, :]

        if tac_flag:
            tx = tactile_xyz[sub]
            tf = tactile_feats[sub]
        else:
            tx = None
            tf = None

        Bg = px.shape[0]
        planes = model.init_planes(Bg, device=px.device, dtype=px.dtype)

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
            robot_state=rs,
            actions_gt=gt_in,
            teacher_forcing=teacher_forcing,
        )

        pred_base = out["pred_actions"]           # [Bg,K_base,da]
        pred_full, mask_target = resampler(pred_base, tgt_len)

        mask = mask_full & mask_target
        loss_g = masked_mse(pred_full, gt_full, mask)

        loss_accum = loss_accum + loss_g
        groups += 1

    if groups > 0:
        loss_accum = loss_accum / float(groups)
    return loss_accum


# -----------------------------
# Train / Eval epoch
# -----------------------------

def run_one_epoch_dual(
    model_ee: MultiStageTactilePointCloudTransformer,
    model_dex: MultiStageTactilePointCloudTransformer,
    resampler_ee: DifferentiableChunkResampler,
    resampler_dex: DifferentiableChunkResampler,
    loader: DataLoader,
    device: torch.device,
    opt_ee: Optional[torch.optim.Optimizer],
    opt_dex: Optional[torch.optim.Optimizer],
    object_chunk_schedule: torch.Tensor,
    *,
    num_objects: int,
    K_base: int,
    ee_stage_id: int,
    dex_stage_id: int,
    grad_clip: float,
    use_amp: bool,
    teacher_forcing: bool,
) -> Dict[str, float]:
    is_train = (opt_ee is not None) or (opt_dex is not None)
    model_ee.train(is_train)
    model_dex.train(is_train)

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    total = 0
    sum_loss = 0.0
    sum_loss_ee = 0.0
    sum_loss_dex = 0.0

    object_chunk_schedule = object_chunk_schedule.to(device=device)

    for batch in loader:
        point_xyz = batch["point_xyz"].to(device)
        point_feats = batch["point_feats"].to(device)
        tactile_xyz = batch["tactile_xyz"].to(device)
        tactile_feats = batch["tactile_feats"].to(device)
        tactile_count = batch["tactile_count"].to(device)
        robot_state = batch["robot_state"].to(device)

        stage = batch["stage"].to(device)
        object_id = batch["object_id"].to(device)

        actions_gt_ee = batch["actions_gt_ee"].to(device)
        actions_mask_ee = batch["actions_mask_ee"].to(device)
        actions_gt_dex = batch["actions_gt_dex"].to(device)
        actions_mask_dex = batch["actions_mask_dex"].to(device)

        B = point_xyz.shape[0]
        total += B

        object_onehot = make_object_onehot(object_id, num_objects).to(device)
        aabb_min, aabb_max = compute_aabb_from_points(point_xyz)
        target_len = object_chunk_schedule[object_id].long()

        if is_train:
            if opt_ee is not None:
                opt_ee.zero_grad(set_to_none=True)
            if opt_dex is not None:
                opt_dex.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
            idx_ee = (stage == int(ee_stage_id))
            idx_dx = (stage == int(dex_stage_id))

            loss_ee = torch.zeros((), device=device)
            loss_dx = torch.zeros((), device=device)

            if idx_ee.any():
                loss_ee = _policy_group_loss(
                    model_ee, resampler_ee,
                    idx=idx_ee,
                    point_xyz=point_xyz,
                    point_feats=point_feats,
                    tactile_xyz=tactile_xyz,
                    tactile_feats=tactile_feats,
                    tactile_count=tactile_count,
                    robot_state=robot_state,
                    stage=stage,
                    object_onehot=object_onehot,
                    aabb_min=aabb_min,
                    aabb_max=aabb_max,
                    actions_gt_full=actions_gt_ee,
                    actions_mask_full=actions_mask_ee,
                    target_len=target_len,
                    K_base=K_base,
                    teacher_forcing=teacher_forcing,
                )

            if idx_dx.any():
                loss_dx = _policy_group_loss(
                    model_dex, resampler_dex,
                    idx=idx_dx,
                    point_xyz=point_xyz,
                    point_feats=point_feats,
                    tactile_xyz=tactile_xyz,
                    tactile_feats=tactile_feats,
                    tactile_count=tactile_count,
                    robot_state=robot_state,
                    stage=stage,
                    object_onehot=object_onehot,
                    aabb_min=aabb_min,
                    aabb_max=aabb_max,
                    actions_gt_full=actions_gt_dex,
                    actions_mask_full=actions_mask_dex,
                    target_len=target_len,
                    K_base=K_base,
                    teacher_forcing=teacher_forcing,
                )

            loss = loss_ee + loss_dx

        if is_train and loss.requires_grad:
            scaler.scale(loss).backward()

            if grad_clip > 0:
                if opt_ee is not None:
                    scaler.unscale_(opt_ee)
                if opt_dex is not None:
                    scaler.unscale_(opt_dex)
                if opt_ee is not None:
                    nn.utils.clip_grad_norm_(model_ee.parameters(), grad_clip)
                if opt_dex is not None:
                    nn.utils.clip_grad_norm_(model_dex.parameters(), grad_clip)

            if opt_ee is not None:
                scaler.step(opt_ee)
            if opt_dex is not None:
                scaler.step(opt_dex)
            scaler.update()

        sum_loss += float(loss.detach().item()) * B
        sum_loss_ee += float(loss_ee.detach().item()) * B
        sum_loss_dex += float(loss_dx.detach().item()) * B

    denom = max(1, total)
    return {
        "loss": sum_loss / denom,
        "loss_ee": sum_loss_ee / denom,
        "loss_dex": sum_loss_dex / denom,
    }


# -----------------------------
# Checkpoint
# -----------------------------

def save_ckpt_dual(
    path: str,
    model_ee: nn.Module,
    model_dex: nn.Module,
    opt_ee: Optional[torch.optim.Optimizer],
    opt_dex: Optional[torch.optim.Optimizer],
    epoch: int,
    cfg_ee: ModelConfig,
    cfg_dex: ModelConfig,
    args: argparse.Namespace,
    object_chunk_schedule: torch.Tensor,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_ee": model_ee.state_dict(),
            "model_dex": model_dex.state_dict(),
            "opt_ee": (opt_ee.state_dict() if opt_ee is not None else None),
            "opt_dex": (opt_dex.state_dict() if opt_dex is not None else None),
            "cfg_ee": asdict(cfg_ee),
            "cfg_dex": asdict(cfg_dex),
            "args": vars(args),
            "object_chunk_schedule": object_chunk_schedule.cpu(),
        },
        path,
    )


# -----------------------------
# Args
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
    p.add_argument("--teacher_forcing", action="store_true")

    # output
    p.add_argument("--out_dir", type=str, default="runs/triplane_vla_dual_post_robotstate")
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
    p.add_argument("--point_feat_dim", type=int, default=0)
    p.add_argument("--tactile_feat_dim", type=int, default=0)

    # tactile tokenization
    p.add_argument("--include_tactile_tokens", action="store_true")
    p.add_argument("--k_tactile", type=int, default=8)
    p.add_argument("--tactile_temp", type=float, default=0.05)

    # stages / objects
    p.add_argument("--num_stages", type=int, default=2)
    p.add_argument("--num_objects", type=int, default=10)

    # policy routing
    p.add_argument("--ee_stage_id", type=int, default=0)
    p.add_argument("--dex_stage_id", type=int, default=1)

    # actions
    p.add_argument("--action_dim", type=int, default=7)
    p.add_argument("--action_dim_ee", type=int, default=None)
    p.add_argument("--action_dim_dex", type=int, default=None)

    # how to read actions from episode
    p.add_argument("--actions_key", type=str, default="actions")
    p.add_argument("--actions_ee_key", type=str, default="actions_ee")
    p.add_argument("--actions_dex_key", type=str, default="actions_dex")

    # chunking
    p.add_argument("--base_chunk_size", type=int, default=32)
    p.add_argument("--min_steps_in_stage", type=int, default=1)

    # object schedule
    p.add_argument("--default_object_chunk", type=int, default=32)
    p.add_argument("--object_chunk_json", type=str, default=None)
    p.add_argument("--object_chunk_sizes", type=int, nargs="+", default=None)
    p.add_argument("--post_chunk_max", type=int, default=0)
    p.add_argument("--postprocess_mode", type=str, default="auto", choices=["auto", "stretch", "extrapolate"])

    # robot state
    p.add_argument("--robot_state_dim", type=int, default=0,
                   help="Total robot state dim. If 0, inferred as arm_state_dim + hand_state_dim.")
    p.add_argument("--arm_state_dim", type=int, default=0)
    p.add_argument("--hand_state_dim", type=int, default=0)
    p.add_argument("--robot_state_key", type=str, default="robot_state")
    p.add_argument("--arm_state_key", type=str, default="arm_state")
    p.add_argument("--hand_state_key", type=str, default="hand_state")
    p.add_argument("--disable_robot_token", action="store_true")
    p.add_argument("--disable_robot_ctx", action="store_true")

    return p.parse_args()


# -----------------------------
# Main
# -----------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    # robot_state_dim resolve
    inferred_robot_dim = int(args.arm_state_dim + args.hand_state_dim)
    if args.robot_state_dim > 0 and inferred_robot_dim > 0 and args.robot_state_dim != inferred_robot_dim:
        raise ValueError(
            f"--robot_state_dim ({args.robot_state_dim}) != --arm_state_dim+--hand_state_dim ({inferred_robot_dim}). "
            f"Either set only --robot_state_dim, or make them match."
        )
    robot_state_dim = int(args.robot_state_dim) if args.robot_state_dim > 0 else inferred_robot_dim

    # resolve per-policy dims
    action_dim_ee = args.action_dim if args.action_dim_ee is None else args.action_dim_ee
    action_dim_dex = args.action_dim if args.action_dim_dex is None else args.action_dim_dex

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    schedule = load_object_chunk_schedule(
        num_objects=args.num_objects,
        default_len=args.default_object_chunk,
        json_path=args.object_chunk_json,
        list_vals=args.object_chunk_sizes,
    )

    K_base = int(args.base_chunk_size)
    K_gt_max = int(args.post_chunk_max) if args.post_chunk_max > 0 else int(max(K_base, int(schedule.max().item())))

    cfg_ee = build_cfg(args, action_dim=int(action_dim_ee), K_base=K_base, robot_state_dim=robot_state_dim)
    cfg_dex = build_cfg(args, action_dim=int(action_dim_dex), K_base=K_base, robot_state_dim=robot_state_dim)

    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(
            {
                "cfg_ee": asdict(cfg_ee),
                "cfg_dex": asdict(cfg_dex),
                "args": vars(args),
                "object_chunk_schedule": schedule.tolist(),
                "K_gt_max": K_gt_max,
                "robot_state_dim": robot_state_dim,
            },
            f,
            indent=2,
        )

    model_ee = MultiStageTactilePointCloudTransformer(cfg_ee).to(device)
    model_dex = MultiStageTactilePointCloudTransformer(cfg_dex).to(device)

    opt_ee = torch.optim.AdamW(model_ee.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_dex = torch.optim.AdamW(model_dex.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    resampler_ee = DifferentiableChunkResampler(K_base=K_base, K_out_max=K_gt_max, mode=args.postprocess_mode).to(device)
    resampler_dex = DifferentiableChunkResampler(K_base=K_base, K_out_max=K_gt_max, mode=args.postprocess_mode).to(device)

    ep_train = EpisodeDataset(args.data_root, split="train")
    ep_val = EpisodeDataset(args.data_root, split="val")

    train_ds = StageSegmentChunkDataset(
        episodes=ep_train,
        action_dim_ee=action_dim_ee,
        action_dim_dex=action_dim_dex,
        K_gt_max=K_gt_max,
        dp=args.point_feat_dim,
        dt=args.tactile_feat_dim,
        min_steps_in_stage=args.min_steps_in_stage,
        actions_key=args.actions_key,
        actions_ee_key=args.actions_ee_key,
        actions_dex_key=args.actions_dex_key,
        robot_state_dim=robot_state_dim,
        arm_state_dim=args.arm_state_dim,
        hand_state_dim=args.hand_state_dim,
        robot_state_key=args.robot_state_key,
        arm_state_key=args.arm_state_key,
        hand_state_key=args.hand_state_key,
    )
    val_ds = StageSegmentChunkDataset(
        episodes=ep_val,
        action_dim_ee=action_dim_ee,
        action_dim_dex=action_dim_dex,
        K_gt_max=K_gt_max,
        dp=args.point_feat_dim,
        dt=args.tactile_feat_dim,
        min_steps_in_stage=args.min_steps_in_stage,
        actions_key=args.actions_key,
        actions_ee_key=args.actions_ee_key,
        actions_dex_key=args.actions_dex_key,
        robot_state_dim=robot_state_dim,
        arm_state_dim=args.arm_state_dim,
        hand_state_dim=args.hand_state_dim,
        robot_state_key=args.robot_state_key,
        arm_state_key=args.arm_state_key,
        hand_state_key=args.hand_state_key,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_stage_segments_dual,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_stage_segments_dual,
        drop_last=False,
    )

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_stats = run_one_epoch_dual(
            model_ee=model_ee,
            model_dex=model_dex,
            resampler_ee=resampler_ee,
            resampler_dex=resampler_dex,
            loader=train_loader,
            device=device,
            opt_ee=opt_ee,
            opt_dex=opt_dex,
            object_chunk_schedule=schedule,
            num_objects=args.num_objects,
            K_base=K_base,
            ee_stage_id=args.ee_stage_id,
            dex_stage_id=args.dex_stage_id,
            grad_clip=args.grad_clip,
            use_amp=args.amp,
            teacher_forcing=args.teacher_forcing,
        )

        val_stats = run_one_epoch_dual(
            model_ee=model_ee,
            model_dex=model_dex,
            resampler_ee=resampler_ee,
            resampler_dex=resampler_dex,
            loader=val_loader,
            device=device,
            opt_ee=None,
            opt_dex=None,
            object_chunk_schedule=schedule,
            num_objects=args.num_objects,
            K_base=K_base,
            ee_stage_id=args.ee_stage_id,
            dex_stage_id=args.dex_stage_id,
            grad_clip=0.0,
            use_amp=False,
            teacher_forcing=args.teacher_forcing,
        )

        dt = time.time() - t0
        log = {
            "epoch": epoch,
            "time_sec": round(dt, 2),
            "train_loss": round(train_stats["loss"], 6),
            "train_loss_ee": round(train_stats["loss_ee"], 6),
            "train_loss_dex": round(train_stats["loss_dex"], 6),
            "val_loss": round(val_stats["loss"], 6),
            "val_loss_ee": round(val_stats["loss_ee"], 6),
            "val_loss_dex": round(val_stats["loss_dex"], 6),
        }
        print(json.dumps(log))

        if epoch % args.save_every == 0:
            save_ckpt_dual(
                os.path.join(args.out_dir, f"ckpt_epoch_{epoch}.pt"),
                model_ee=model_ee,
                model_dex=model_dex,
                opt_ee=opt_ee,
                opt_dex=opt_dex,
                epoch=epoch,
                cfg_ee=cfg_ee,
                cfg_dex=cfg_dex,
                args=args,
                object_chunk_schedule=schedule,
            )

        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            save_ckpt_dual(
                os.path.join(args.out_dir, "ckpt_best.pt"),
                model_ee=model_ee,
                model_dex=model_dex,
                opt_ee=opt_ee,
                opt_dex=opt_dex,
                epoch=epoch,
                cfg_ee=cfg_ee,
                cfg_dex=cfg_dex,
                args=args,
                object_chunk_schedule=schedule,
            )

    print("Done. Best val loss:", best_val)


if __name__ == "__main__":
    main()




# command
# python train_triplane_vla_dual_policy_postprocess_robotstate.py \
#   --data_root /path/to/data \
#   --num_objects 10 \
#   --arm_state_dim 14 \
#   --hand_state_dim 16 \
#   --teacher_forcing \
#   --amp