"""learning.model.model

A lightweight multi-modal policy / action predictor for robotics data:
- Joint angles (vector) with Fourier positional encoding
- Point cloud (NxC) with Fourier positional encoding on xyz and a PointNet-style encoder
- Tactile (vector) with an MLP encoder
- 4 output heads (multi-task / factorized action prediction)
- Separate loss per head + weighted total loss

This module is intentionally self-contained and works with a wide variety of dataset formats.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor


class FourierFeatures(nn.Module):
    """Fourier feature mapping (a.k.a. positional encoding / random Fourier features without randomness).

    Maps x in R^D -> [x, sin(2π f_k x), cos(2π f_k x)] for k=1..K (broadcasted over D).

    Notes:
      - If your inputs are not normalized, consider tuning max_freq.
      - For joint angles in radians, max_freq ~ 10-30 often works.
      - For xyz in meters, you typically want to normalize coordinates first or use smaller max_freq.
    """

    def __init__(
        self,
        input_dim: int,
        num_bands: int = 6,
        max_freq: float = 10.0,
        include_input: bool = True,
        log_space: bool = True,
        base: float = 2.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be > 0, got {input_dim}")
        if num_bands <= 0:
            raise ValueError(f"num_bands must be > 0, got {num_bands}")
        if max_freq <= 0:
            raise ValueError(f"max_freq must be > 0, got {max_freq}")

        self.input_dim = int(input_dim)
        self.num_bands = int(num_bands)
        self.max_freq = float(max_freq)
        self.include_input = bool(include_input)
        self.log_space = bool(log_space)
        self.base = float(base)

        # Register frequencies as a buffer so they move with .to(device) and are saved in state_dict
        if log_space:
            # frequencies: base^linspace(0, log_base(max_freq), K)
            max_power = math.log(max_freq, base)
            freqs = base ** torch.linspace(0.0, max_power, num_bands)
        else:
            freqs = torch.linspace(1.0, max_freq, num_bands)

        self.register_buffer("freqs", freqs, persistent=True)

    @property
    def out_dim(self) -> int:
        return (self.input_dim if self.include_input else 0) + 2 * self.input_dim * self.num_bands

    def forward(self, x: Tensor) -> Tensor:
        """x: (..., input_dim)"""
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected last dim {self.input_dim}, got {x.shape[-1]}")
        # (..., D) -> (..., 1, D)
        x_exp = x.unsqueeze(-2)
        # (K,) -> (K, 1)
        freqs = self.freqs.view(-1, 1)
        # (..., K, D)
        # Use 2π to make frequencies more interpretable, but it isn't required.
        ang = (2.0 * math.pi) * x_exp * freqs
        sin = torch.sin(ang)
        cos = torch.cos(ang)
        # (..., K, D) -> (..., K*D)
        sin = sin.flatten(-2)
        cos = cos.flatten(-2)
        if self.include_input:
            return torch.cat([x, sin, cos], dim=-1)
        return torch.cat([sin, cos], dim=-1)


def _mlp(
    in_dim: int,
    hidden_dims: Sequence[int],
    out_dim: int,
    *,
    activation: nn.Module = nn.ReLU(inplace=True),
    dropout: float = 0.0,
    layer_norm: bool = True,
) -> nn.Sequential:
    """A simple MLP builder."""
    dims = [in_dim, *map(int, hidden_dims), int(out_dim)]
    layers: List[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i != len(dims) - 2:
            if layer_norm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(activation)
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


class PointNetEncoder(nn.Module):
    """A lightweight PointNet-style encoder with optional xyz positional encoding.

    Input: points: (B, N, C) where C >= 3 and points[..., :3] are xyz.
    Output: global feature: (B, out_dim)

    If C>3 (e.g., rgb, intensity), those channels are concatenated after xyz encoding.
    """

    def __init__(
        self,
        point_dim: int,
        out_dim: int,
        *,
        xyz_pe: Optional[FourierFeatures] = None,
        per_point_hidden: Sequence[int] = (128, 256),
        pooling: str = "max",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if point_dim < 3:
            raise ValueError("point_dim must be >= 3 (xyz)")
        self.point_dim = int(point_dim)
        self.out_dim = int(out_dim)
        self.xyz_pe = xyz_pe
        self.pooling = pooling

        in_dim = (xyz_pe.out_dim if xyz_pe is not None else 3) + (point_dim - 3)
        self.per_point = _mlp(in_dim, per_point_hidden, out_dim, dropout=dropout, layer_norm=True)

    def forward(self, points: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """points: (B,N,C); mask: (B,N) bool where True means valid."""
        if points.dim() != 3:
            raise ValueError(f"points must be (B,N,C), got {points.shape}")
        B, N, C = points.shape
        if C != self.point_dim:
            raise ValueError(f"Expected point_dim {self.point_dim}, got {C}")

        xyz = points[..., :3]
        extra = points[..., 3:] if C > 3 else None

        if self.xyz_pe is not None:
            xyz_enc = self.xyz_pe(xyz)  # (B,N,pe_dim)
        else:
            xyz_enc = xyz

        if extra is not None and extra.numel() > 0:
            x = torch.cat([xyz_enc, extra], dim=-1)
        else:
            x = xyz_enc

        # Apply MLP point-wise. Flatten B*N for speed.
        x = x.reshape(B * N, -1)
        x = self.per_point(x)
        x = x.reshape(B, N, -1)

        if mask is not None:
            # mask invalid points before pooling
            if mask.shape != (B, N):
                raise ValueError(f"mask must be (B,N) but got {mask.shape}")
            # set invalid positions to very negative for max pooling
            if self.pooling == "max":
                x = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            else:
                x = x.masked_fill(~mask.unsqueeze(-1), 0.0)

        if self.pooling == "max":
            global_feat = torch.max(x, dim=1).values
            # If all points were invalid, max becomes -inf; replace with zeros.
            global_feat = torch.where(torch.isfinite(global_feat), global_feat, torch.zeros_like(global_feat))
        elif self.pooling == "mean":
            if mask is None:
                global_feat = torch.mean(x, dim=1)
            else:
                denom = mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
                global_feat = x.sum(dim=1) / denom
        else:
            raise ValueError(f"Unknown pooling '{self.pooling}' (use 'max' or 'mean')")
        return global_feat


class VectorEncoder(nn.Module):
    """Encodes a vector with optional Fourier positional encoding then an MLP."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        pe: Optional[FourierFeatures] = None,
        hidden_dims: Sequence[int] = (256, 256),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError("in_dim must be > 0")
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.pe = pe
        enc_in = pe.out_dim if pe is not None else in_dim
        self.mlp = _mlp(enc_in, hidden_dims, out_dim, dropout=dropout, layer_norm=True)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 2:
            raise ValueError(f"Expected (B,D), got {x.shape}")
        if x.shape[-1] != self.in_dim:
            raise ValueError(f"Expected last dim {self.in_dim}, got {x.shape[-1]}")
        if self.pe is not None:
            x = self.pe(x)
        return self.mlp(x)


class MultiHeadOutput(nn.Module):
    """Four separate heads on top of a shared trunk."""

    def __init__(
        self,
        in_dim: int,
        head_dims: Sequence[int],
        *,
        head_names: Optional[Sequence[str]] = None,
        hidden_dims: Sequence[int] = (256, 256),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(head_dims) != 4:
            raise ValueError(f"Expected 4 heads, got {len(head_dims)}")
        if head_names is None:
            head_names = [f"head{i}" for i in range(4)]
        if len(head_names) != 4:
            raise ValueError(f"Expected 4 head names, got {len(head_names)}")

        self.head_names = list(head_names)
        self.head_dims = [int(d) for d in head_dims]

        self.heads = nn.ModuleDict(
            {
                name: _mlp(in_dim, hidden_dims, out_dim, dropout=dropout, layer_norm=True)
                for name, out_dim in zip(self.head_names, self.head_dims)
            }
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        return {name: head(x) for name, head in self.heads.items()}
class MultiModalPolicy(nn.Module):
    """(Optionally) Point cloud only -> shared latent -> 4 action heads.

    If ignore_joint=True, the model does NOT use the joint input at all.
    """

    def __init__(
        self,
        *,
        joint_dim: int,
        point_dim: int,
        latent_dim: int = 512,
        joint_pe: Optional[FourierFeatures] = None,
        point_xyz_pe: Optional[FourierFeatures] = None,
        joint_feat_dim: int = 256,
        point_feat_dim: int = 256,
        trunk_hidden: Sequence[int] = (512, 512),
        trunk_dropout: float = 0.1,
        head_dims: Sequence[int] = (8, 8, 8, 8),
        head_names: Optional[Sequence[str]] = None,
        head_hidden: Sequence[int] = (256, 256),
        head_dropout: float = 0.0,
        ignore_joint: bool = False,   # <--- NEW
    ) -> None:
        super().__init__()
        if point_dim < 3:
            raise ValueError("point_dim must be >= 3")

        self.ignore_joint = bool(ignore_joint)

        # NOTE: only require joint_dim if we actually use joints
        if not self.ignore_joint and joint_dim <= 0:
            raise ValueError("joint_dim must be > 0 unless ignore_joint=True")

        self.joint_dim = int(joint_dim)
        self.point_dim = int(point_dim)

        # ---- encoders ----
        if not self.ignore_joint:
            self.joint_enc = VectorEncoder(
                joint_dim,
                joint_feat_dim,
                pe=joint_pe,
                hidden_dims=(256, 256),
                dropout=trunk_dropout,
            )
            trunk_in = joint_feat_dim + point_feat_dim
        else:
            self.joint_enc = None
            trunk_in = point_feat_dim

        self.pc_enc = PointNetEncoder(
            point_dim=point_dim,
            out_dim=point_feat_dim,
            xyz_pe=point_xyz_pe,
            per_point_hidden=(128, 256),
            pooling="max",
            dropout=trunk_dropout,
        )

        # ---- trunk + heads ----
        self.trunk = _mlp(trunk_in, trunk_hidden, latent_dim, dropout=trunk_dropout, layer_norm=True)

        self.heads = MultiHeadOutput(
            in_dim=latent_dim,
            head_dims=head_dims,
            head_names=head_names,
            hidden_dims=head_hidden,
            dropout=head_dropout,
        )

    def forward(
        self,
        *,
        joint: Optional[Tensor],      # allow None when ignore_joint=True
        point_cloud: Tensor,
        point_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        pc_feat = self.pc_enc(point_cloud, mask=point_mask)

        if self.ignore_joint:
            fused = pc_feat
        else:
            if joint is None:
                raise ValueError("joint must be provided when ignore_joint=False")
            joint_feat = self.joint_enc(joint)
            fused = torch.cat([joint_feat, pc_feat], dim=-1)

        latent = self.trunk(fused)
        return self.heads(latent)

@dataclass
class MultiHeadLossConfig:
    loss_type: str = "mse"  # 'mse' or 'smooth_l1' or 'l1'
    weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)


class MultiHeadLoss(nn.Module):
    """Computes separate losses for 4 heads and a weighted sum."""

    def __init__(self, head_names: Sequence[str], cfg: MultiHeadLossConfig = MultiHeadLossConfig()) -> None:
        super().__init__()
        if len(head_names) != 4:
            raise ValueError("MultiHeadLoss expects exactly 4 head names")
        self.head_names = list(head_names)
        self.cfg = cfg

        lt = cfg.loss_type.lower()
        if lt == "mse":
            self._loss_fn = nn.MSELoss(reduction="mean")
        elif lt in ("smooth_l1", "huber"):
            self._loss_fn = nn.SmoothL1Loss(reduction="mean")
        elif lt == "l1":
            self._loss_fn = nn.L1Loss(reduction="mean")
        else:
            raise ValueError(f"Unknown loss_type '{cfg.loss_type}'")

        if len(cfg.weights) != 4:
            raise ValueError("weights must have length 4")
        self.register_buffer("weights", torch.tensor(cfg.weights, dtype=torch.float32), persistent=False)

    @torch.no_grad()
    def _ensure_device(self, preds: Mapping[str, Tensor]) -> None:
        # keep weights on same device as preds
        dev = next(iter(preds.values())).device
        if self.weights.device != dev:
            self.weights = self.weights.to(dev)

    def forward(
        self,
        preds: Mapping[str, Tensor],
        targets: Mapping[str, Tensor],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Returns: total_loss, loss_by_head"""
        for name in self.head_names:
            if name not in preds:
                raise KeyError(f"preds missing head '{name}'")
            if name not in targets:
                raise KeyError(f"targets missing head '{name}'")

        self._ensure_device(preds)

        loss_by_head: Dict[str, Tensor] = {}
        total = torch.zeros((), device=next(iter(preds.values())).device)
        for i, name in enumerate(self.head_names):
            l = self._loss_fn(preds[name], targets[name])
            loss_by_head[name] = l
            total = total + self.weights[i] * l
        return total, loss_by_head


def split_action_vector(
    action: Tensor,
    head_dims: Sequence[int],
    head_names: Sequence[str],
) -> Dict[str, Tensor]:
    """Splits an action vector (B, sum(head_dims)) into a dict of 4 targets."""
    if action.dim() != 2:
        raise ValueError(f"action must be (B,D), got {action.shape}")
    if len(head_dims) != 4 or len(head_names) != 4:
        raise ValueError("Expected 4 head dims and 4 head names")

    total = sum(int(d) for d in head_dims)
    if action.shape[-1] != total:
        raise ValueError(f"Action dim {action.shape[-1]} does not match sum(head_dims)={total}")

    outs: Dict[str, Tensor] = {}
    idx = 0
    for name, d in zip(head_names, head_dims):
        outs[name] = action[:, idx : idx + d]
        idx += d
    return outs


def infer_head_dims_from_action(action: Union[Tensor, Mapping[str, Tensor]]) -> Tuple[List[str], List[int]]:
    """Heuristic to infer 4 heads from a sample action.

    - If action is a dict with >=4 tensors, uses first 4 keys (sorted for determinism).
    - If action is a tensor (D,), splits into 4 nearly-equal chunks.

    Returns: (head_names, head_dims)
    """
    if isinstance(action, Mapping):
        keys = sorted(list(action.keys()))
        if len(keys) < 4:
            raise ValueError(f"Action dict has {len(keys)} keys; need at least 4")
        head_names = keys[:4]
        head_dims = [int(action[k].numel()) for k in head_names]
        return head_names, head_dims

    if not torch.is_tensor(action):
        raise TypeError("action must be a Tensor or Mapping[str,Tensor]")
    D = int(action.numel())
    # split into 4 chunks as evenly as possible
    base = D // 4
    rem = D % 4
    dims = [base + (1 if i < rem else 0) for i in range(4)]
    # avoid zeros if D<4
    if any(d == 0 for d in dims):
        raise ValueError(f"Cannot infer 4 heads from action dim {D}. Provide head_dims explicitly.")
    names = [f"head{i}" for i in range(4)]
    return names, dims
