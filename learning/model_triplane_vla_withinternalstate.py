from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities
# -----------------------------

def normalize_xyz_to_minus1_1(
    xyz: torch.Tensor,
    aabb_min: torch.Tensor,
    aabb_max: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Normalize xyz from world coordinates into [-1, 1] using an AABB.

    xyz:      [B, N, 3] or [B, M, 3]
    aabb_min: [B, 3] or [3]
    aabb_max: [B, 3] or [3]
    """
    if aabb_min.dim() == 1:
        aabb_min = aabb_min.view(1, 1, 3)
    else:
        aabb_min = aabb_min.view(-1, 1, 3)

    if aabb_max.dim() == 1:
        aabb_max = aabb_max.view(1, 1, 3)
    else:
        aabb_max = aabb_max.view(-1, 1, 3)

    xyz01 = (xyz - aabb_min) / (aabb_max - aabb_min + eps)  # [0,1]
    return xyz01 * 2.0 - 1.0  # [-1,1]


def subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
    """Causal mask for autoregressive decoding. Returns [sz, sz] with -inf in upper triangle."""
    mask = torch.full((sz, sz), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int],
        out_dim: int,
        act: nn.Module = nn.GELU(),
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        dims = [in_dim] + hidden_dims + [out_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FourierPosEnc3D(nn.Module):
    """3D Fourier features positional encoding (sin/cos) for xyz in [-1,1]."""
    def __init__(self, num_bands: int = 6, max_freq: float = 10.0) -> None:
        super().__init__()
        self.num_bands = num_bands
        freqs = torch.linspace(1.0, max_freq, steps=num_bands)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        xyz: [B, N, 3]
        returns: [B, N, 3 * 2 * num_bands]
        """
        x = xyz.unsqueeze(-1) * self.freqs.view(1, 1, 1, -1) * torch.pi
        sin = torch.sin(x)
        cos = torch.cos(x)
        pe = torch.cat([sin, cos], dim=-1)  # [B,N,3,2*num_bands]
        return pe.flatten(start_dim=2)      # [B,N, 3*2*num_bands]


# -----------------------------
# Tri-plane sampling + update
# -----------------------------

@dataclass
class TriPlaneConfig:
    res: int = 64
    channels: int = 32


class TriPlaneSampler(nn.Module):
    """
    Samples tri-plane features at 3D points using bilinear grid_sample on XY/XZ/YZ planes.

    Planes:
      plane_xy, plane_xz, plane_yz: [B, C, H, W]
    Points:
      xyz_norm: [B, N, 3] in [-1,1]
    Returns:
      [B, N, 3C] concatenated features (xy,xz,yz)
    """
    def __init__(self, align_corners: bool = True) -> None:
        super().__init__()
        self.align_corners = align_corners

    def sample(
        self,
        plane_xy: torch.Tensor,
        plane_xz: torch.Tensor,
        plane_yz: torch.Tensor,
        xyz_norm: torch.Tensor
    ) -> torch.Tensor:
        B, N, _ = xyz_norm.shape
        grid_xy = xyz_norm[..., [0, 1]].unsqueeze(2)  # [B,N,1,2]
        grid_xz = xyz_norm[..., [0, 2]].unsqueeze(2)  # [B,N,1,2]
        grid_yz = xyz_norm[..., [1, 2]].unsqueeze(2)  # [B,N,1,2]

        feat_xy = F.grid_sample(plane_xy, grid_xy, mode="bilinear", padding_mode="zeros",
                                align_corners=self.align_corners)  # [B,C,N,1]
        feat_xz = F.grid_sample(plane_xz, grid_xz, mode="bilinear", padding_mode="zeros",
                                align_corners=self.align_corners)
        feat_yz = F.grid_sample(plane_yz, grid_yz, mode="bilinear", padding_mode="zeros",
                                align_corners=self.align_corners)

        feat_xy = feat_xy.squeeze(-1).transpose(1, 2)  # [B,N,C]
        feat_xz = feat_xz.squeeze(-1).transpose(1, 2)
        feat_yz = feat_yz.squeeze(-1).transpose(1, 2)

        return torch.cat([feat_xy, feat_xz, feat_yz], dim=-1)  # [B,N,3C]


class TriPlaneWriter(nn.Module):
    """
    Differentiable plane update module:
    - small conv stack per plane
    - FiLM modulation from conditioning vector cond
    - residual update

    cond: [B, cond_dim]
    """
    def __init__(self, plane_cfg: TriPlaneConfig, cond_dim: int, conv_hidden: int = 64) -> None:
        super().__init__()
        C = plane_cfg.channels

        def make_plane_conv() -> nn.Module:
            return nn.Sequential(
                nn.Conv2d(C, conv_hidden, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(conv_hidden, C, kernel_size=3, padding=1),
            )

        self.conv_xy = make_plane_conv()
        self.conv_xz = make_plane_conv()
        self.conv_yz = make_plane_conv()

        self.film = MLP(cond_dim, [cond_dim], out_dim=3 * 2 * C, dropout=0.0)
        self.delta_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        plane_xy: torch.Tensor,
        plane_xz: torch.Tensor,
        plane_yz: torch.Tensor,
        cond: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, _, _ = plane_xy.shape

        gb = self.film(cond).view(B, 3, 2, C)  # [B,3,2,C]
        gamma = gb[:, :, 0, :].unsqueeze(-1).unsqueeze(-1)  # [B,3,C,1,1]
        beta  = gb[:, :, 1, :].unsqueeze(-1).unsqueeze(-1)  # [B,3,C,1,1]

        dxy = self.conv_xy(plane_xy)
        dxz = self.conv_xz(plane_xz)
        dyz = self.conv_yz(plane_yz)

        dxy = gamma[:, 0] * dxy + beta[:, 0]
        dxz = gamma[:, 1] * dxz + beta[:, 1]
        dyz = gamma[:, 2] * dyz + beta[:, 2]

        scale = torch.clamp(self.delta_scale, 0.0, 1.0)
        plane_xy = plane_xy + scale * dxy
        plane_xz = plane_xz + scale * dxz
        plane_yz = plane_yz + scale * dyz
        return plane_xy, plane_xz, plane_yz


def init_triplanes(
    batch_size: int,
    plane_cfg: TriPlaneConfig,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    H = W = plane_cfg.res
    C = plane_cfg.channels
    plane_xy = torch.zeros((batch_size, C, H, W), device=device, dtype=dtype)
    plane_xz = torch.zeros((batch_size, C, H, W), device=device, dtype=dtype)
    plane_yz = torch.zeros((batch_size, C, H, W), device=device, dtype=dtype)
    return plane_xy, plane_xz, plane_yz


# -----------------------------
# Tokenizer: point cloud + tactile
# -----------------------------

@dataclass
class TokenizerConfig:
    point_feat_dim: int
    tactile_feat_dim: int
    d_model: int = 256
    pe_bands: int = 6
    pe_max_freq: float = 10.0

    k_tactile: int = 8
    tactile_temp: float = 0.05

    include_tactile_tokens: bool = True


class PointTactileTokenizer(nn.Module):
    """
    Builds encoder tokens:
      [ global_tok | point_tokens | (optional tactile_tokens) ]

    - point tokens include raw point feats + PE(xyz) + tri-plane sampled features
    - tactile tokens include tactile feats + PE(xyz)
    - tactile registration: kNN weighted sum of tactile token embeddings added into point tokens

    NOTE:
    `ctx_emb` is a d_model vector injected into all tokens (stage+object(+robot)).
    """
    def __init__(self, cfg: TokenizerConfig, plane_cfg: TriPlaneConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.plane_cfg = plane_cfg
        self.pe3 = FourierPosEnc3D(cfg.pe_bands, cfg.pe_max_freq)

        in_point = cfg.point_feat_dim + (3 * 2 * cfg.pe_bands) + (3 * plane_cfg.channels)
        self.point_mlp = MLP(in_point, [cfg.d_model, cfg.d_model], cfg.d_model, dropout=0.0)

        in_tac = cfg.tactile_feat_dim + (3 * 2 * cfg.pe_bands)
        self.tactile_mlp = MLP(in_tac, [cfg.d_model, cfg.d_model], cfg.d_model, dropout=0.0)

        self.global_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))

    @torch.no_grad()
    def _knn_indices(
        self, point_xyz: torch.Tensor, tactile_xyz: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = torch.cdist(point_xyz, tactile_xyz)  # [B,N,M]
        knn_dist, knn_idx = dist.topk(k=k, dim=-1, largest=False)
        return knn_dist, knn_idx

    def forward(
        self,
        point_xyz_norm: torch.Tensor,     # [B,N,3]
        point_feats: torch.Tensor,        # [B,N,dp]
        tactile_xyz_norm: Optional[torch.Tensor],  # [B,M,3] or None
        tactile_feats: Optional[torch.Tensor],     # [B,M,dt] or None
        triplane_feats_at_points: torch.Tensor,    # [B,N,3*C_plane]
        ctx_emb: torch.Tensor                      # [B,d_model]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, N, _ = point_xyz_norm.shape
        assert point_feats.shape[-1] == self.cfg.point_feat_dim, \
            f"point_feats last dim {point_feats.shape[-1]} != cfg.point_feat_dim {self.cfg.point_feat_dim}"

        pe_p = self.pe3(point_xyz_norm)
        point_in = torch.cat([point_feats, pe_p, triplane_feats_at_points], dim=-1)
        point_tok = self.point_mlp(point_in)  # [B,N,d]

        tactile_tok: Optional[torch.Tensor] = None
        tactile_reg = torch.zeros_like(point_tok)

        if tactile_xyz_norm is not None and tactile_feats is not None:
            B2, M, _ = tactile_xyz_norm.shape
            assert B2 == B
            assert tactile_feats.shape[-1] == self.cfg.tactile_feat_dim

            pe_t = self.pe3(tactile_xyz_norm)
            tactile_in = torch.cat([tactile_feats, pe_t], dim=-1)
            tactile_tok = self.tactile_mlp(tactile_in)  # [B,M,d]

            if M > 0 and self.cfg.k_tactile > 0:
                k = min(self.cfg.k_tactile, M)
                knn_dist, knn_idx = self._knn_indices(point_xyz_norm, tactile_xyz_norm, k=k)  # [B,N,k]
                batch_idx = torch.arange(B, device=point_xyz_norm.device)[:, None, None].expand(B, N, k)
                knn_emb = tactile_tok[batch_idx, knn_idx]  # [B,N,k,d]
                w = torch.softmax(-knn_dist / max(self.cfg.tactile_temp, 1e-6), dim=-1).unsqueeze(-1)
                tactile_reg = (w * knn_emb).sum(dim=2)  # [B,N,d]

        # inject tactile into points
        point_tok = point_tok + tactile_reg

        # inject ctx into all tokens
        point_tok = point_tok + ctx_emb.unsqueeze(1)

        global_tok = self.global_token.expand(B, 1, -1) + ctx_emb.unsqueeze(1)

        tokens_list = [global_tok, point_tok]
        if self.cfg.include_tactile_tokens and tactile_tok is not None:
            tactile_tok = tactile_tok + ctx_emb.unsqueeze(1)
            tokens_list.append(tactile_tok)

        tokens = torch.cat(tokens_list, dim=1)

        aux = {
            "point_tokens": point_tok,
            "tactile_tokens": tactile_tok if tactile_tok is not None else torch.empty(0, device=point_xyz_norm.device),
            "tactile_reg": tactile_reg,
            "global_token": global_tok,
        }
        return tokens, aux


# -----------------------------
# Action decoder with chunking
# -----------------------------

@dataclass
class DecoderConfig:
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1

    action_dim: int = 7
    max_chunk_size: int = 16

    stage_chunk_sizes: Tuple[int, int] = (2, 8)

    use_adaptive_chunking: bool = False
    candidate_chunk_sizes: Tuple[int, ...] = (1, 2, 4, 8, 16)


class ActionChunkDecoder(nn.Module):
    """Classic Transformer decoder that predicts an action chunk of length K."""
    def __init__(self, cfg: DecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=cfg.num_layers)

        self.action_in = nn.Linear(cfg.action_dim, cfg.d_model)
        self.action_out = nn.Linear(cfg.d_model, cfg.action_dim)

        self.pos_inchunk = nn.Embedding(cfg.max_chunk_size, cfg.d_model)
        self.start_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))

        if cfg.use_adaptive_chunking:
            self.chunk_head = MLP(cfg.d_model, [cfg.d_model], out_dim=len(cfg.candidate_chunk_sizes))
        else:
            self.chunk_head = None

    def select_chunk_size(self, memory: torch.Tensor) -> int:
        assert self.cfg.use_adaptive_chunking and self.chunk_head is not None
        pooled = memory[:, 0]  # typically ctx token
        logits = self.chunk_head(pooled)
        idx = logits.argmax(dim=-1)
        if not torch.all(idx == idx[0]):
            raise ValueError("Adaptive chunk size differs within batch; split batch or implement per-sample decoding.")
        return int(self.cfg.candidate_chunk_sizes[int(idx[0].item())])

    def fixed_chunk_size_from_stage(self, stage: torch.Tensor) -> int:
        stage = stage.view(-1)
        if not torch.all(stage == stage[0]):
            raise ValueError("Batch contains mixed stages; split batch by stage or implement masking.")
        s = int(stage[0].item())
        K = self.cfg.stage_chunk_sizes[s]
        if K > self.cfg.max_chunk_size:
            raise ValueError(f"chunk size {K} exceeds max_chunk_size {self.cfg.max_chunk_size}")
        return K

    def decode_teacher_forcing(
        self,
        memory: torch.Tensor,         # [B,S,d]
        ctx_emb: torch.Tensor,        # [B,d]
        actions_gt: torch.Tensor,     # [B,Kmax,action_dim]
        K: int
    ) -> torch.Tensor:
        B = memory.shape[0]
        device = memory.device

        assert actions_gt.shape[1] >= K
        actions_gt = actions_gt[:, :K, :]

        dec_in_actions = torch.zeros((B, K, self.cfg.action_dim), device=device, dtype=actions_gt.dtype)
        if K > 1:
            dec_in_actions[:, 1:, :] = actions_gt[:, :-1, :]

        dec_tok = self.action_in(dec_in_actions)
        dec_tok = dec_tok + self.pos_inchunk(torch.arange(K, device=device)).view(1, K, -1)
        dec_tok = dec_tok + ctx_emb.unsqueeze(1)
        dec_tok[:, :1, :] = dec_tok[:, :1, :] + self.start_token

        tgt_mask = subsequent_mask(K, device=device)
        out = self.decoder(tgt=dec_tok, memory=memory, tgt_mask=tgt_mask)
        return self.action_out(out)

    @torch.no_grad()
    def decode_autoregressive(
        self,
        memory: torch.Tensor,
        ctx_emb: torch.Tensor,
        K: int
    ) -> torch.Tensor:
        B = memory.shape[0]
        device = memory.device

        actions = []
        prev_actions = torch.zeros((B, 0, self.cfg.action_dim), device=device)

        for t in range(K):
            dec_in_actions = torch.zeros((B, t + 1, self.cfg.action_dim), device=device)
            if t > 0:
                dec_in_actions[:, 1:, :] = prev_actions

            dec_tok = self.action_in(dec_in_actions)
            dec_tok = dec_tok + self.pos_inchunk(torch.arange(t + 1, device=device)).view(1, t + 1, -1)
            dec_tok = dec_tok + ctx_emb.unsqueeze(1)
            dec_tok[:, :1, :] = dec_tok[:, :1, :] + self.start_token

            tgt_mask = subsequent_mask(t + 1, device=device)
            out = self.decoder(tgt=dec_tok, memory=memory, tgt_mask=tgt_mask)
            a_t = self.action_out(out[:, -1, :])
            actions.append(a_t)
            prev_actions = torch.cat([prev_actions, a_t.unsqueeze(1)], dim=1)

        return torch.stack(actions, dim=1)


# -----------------------------
# Encoder wrapper
# -----------------------------

@dataclass
class EncoderConfig:
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1


class PointTactileEncoder(nn.Module):
    """Classic Transformer Encoder."""
    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder(tokens)


# -----------------------------
# "Rendering" + success head (stubs)
# -----------------------------

class TriPlaneRendererStub(nn.Module):
    """Placeholder for your differentiable rendering pipeline."""
    def forward(self, plane_xy: torch.Tensor, plane_xz: torch.Tensor, plane_yz: torch.Tensor) -> torch.Tensor:
        pxy = plane_xy.mean(dim=(2, 3))
        pxz = plane_xz.mean(dim=(2, 3))
        pyz = plane_yz.mean(dim=(2, 3))
        return torch.cat([pxy, pxz, pyz], dim=-1)  # [B,3C]


class SuccessHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


# -----------------------------
# Full model (UPDATED: robot_state support)
# -----------------------------

@dataclass
class ModelConfig:
    plane: TriPlaneConfig = field(default_factory=TriPlaneConfig)

    tokenizer: TokenizerConfig = field(default_factory=lambda: TokenizerConfig(
        point_feat_dim=0, tactile_feat_dim=0, d_model=256, include_tactile_tokens=True
    ))

    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(d_model=256))
    decoder: DecoderConfig = field(default_factory=lambda: DecoderConfig(d_model=256, action_dim=7))

    num_stages: int = 2
    num_objects: int = 1

    # NEW: robot proprioception input
    robot_state_dim: int = 0
    use_robot_token: bool = True     # adds a dedicated robot token to encoder input
    fuse_robot_into_ctx: bool = True # adds robot embedding into ctx embedding

    # If None: auto
    writer_cond_dim: Optional[int] = None


class MultiStageTactilePointCloudTransformer(nn.Module):
    """
    One macro-step forward:
      - sample tri-plane at points
      - build tokens with tactile registration + (stage+object(+robot)) context embedding
      - (optional) prepend robot token
      - encode tokens
      - decode action chunk
      - update tri-plane (cond includes object (+robot) embeddings)
    """
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.tokenizer.d_model

        # Stage embedding
        self.stage_emb = nn.Embedding(cfg.num_stages, d)

        # Object one-hot -> linear embedding
        self.object_proj = nn.Linear(cfg.num_objects, d, bias=True)

        # Robot state -> embedding (if used)
        if cfg.robot_state_dim > 0:
            self.robot_proj = nn.Sequential(
                nn.Linear(cfg.robot_state_dim, d),
                nn.GELU(),
                nn.LayerNorm(d),
            )
        else:
            self.robot_proj = None

        # Auto writer cond dim
        if cfg.writer_cond_dim is None:
            # global_out(d) + pooled_action(d) + object_emb(d) + (optional robot_emb(d))
            cfg.writer_cond_dim = (3 + (1 if cfg.robot_state_dim > 0 else 0)) * d

        self.sampler = TriPlaneSampler(align_corners=True)
        self.writer = TriPlaneWriter(cfg.plane, cond_dim=cfg.writer_cond_dim)

        self.tokenizer = PointTactileTokenizer(cfg.tokenizer, cfg.plane)
        self.encoder = PointTactileEncoder(cfg.encoder)
        self.decoder = ActionChunkDecoder(cfg.decoder)

        self.action_pool = nn.Sequential(nn.Linear(cfg.decoder.action_dim, d), nn.GELU())

        self.renderer = TriPlaneRendererStub()
        self.success_head = SuccessHead(in_dim=3 * cfg.plane.channels)

        # Explicit ctx token projection
        self.ctx_token_proj = nn.Linear(d, d)

        # Dedicated robot token projection (token space)
        self.robot_token_proj = nn.Linear(d, d) if (cfg.robot_state_dim > 0 and cfg.use_robot_token) else None

    def init_planes(
        self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return init_triplanes(batch_size, self.cfg.plane, device=device, dtype=dtype)

    def forward_macro_step(
        self,
        point_xyz: torch.Tensor,              # [B,N,3]
        point_feats: torch.Tensor,            # [B,N,dp]
        tactile_xyz: Optional[torch.Tensor],  # [B,M,3] or None
        tactile_feats: Optional[torch.Tensor],# [B,M,dt] or None
        stage: torch.Tensor,                  # [B]
        object_onehot: Optional[torch.Tensor],# [B,num_objects] one-hot (or soft)
        aabb_min: torch.Tensor,               # [B,3] or [3]
        aabb_max: torch.Tensor,               # [B,3] or [3]
        planes: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        robot_state: Optional[torch.Tensor] = None,  # [B,robot_state_dim] (NEW)
        actions_gt: Optional[torch.Tensor] = None,    # [B,Kmax,da]
        teacher_forcing: bool = True
    ) -> Dict[str, torch.Tensor]:
        plane_xy, plane_xz, plane_yz = planes
        B, N, _ = point_xyz.shape
        device = point_xyz.device
        dtype = point_xyz.dtype

        # normalize xyz
        point_xyz_norm = normalize_xyz_to_minus1_1(point_xyz, aabb_min=aabb_min, aabb_max=aabb_max)
        tactile_xyz_norm = None
        if tactile_xyz is not None:
            tactile_xyz_norm = normalize_xyz_to_minus1_1(tactile_xyz, aabb_min=aabb_min, aabb_max=aabb_max)

        # stage emb
        stage = stage.long().view(-1)
        e_stage = self.stage_emb(stage)  # [B,d]

        # object emb (one-hot -> linear)
        if object_onehot is None:
            object_onehot = torch.zeros((B, self.cfg.num_objects), device=device, dtype=dtype)
        else:
            assert object_onehot.shape == (B, self.cfg.num_objects), \
                f"object_onehot must be [B,num_objects]=[{B},{self.cfg.num_objects}], got {tuple(object_onehot.shape)}"
            object_onehot = object_onehot.to(device=device, dtype=dtype)
        e_obj = self.object_proj(object_onehot)  # [B,d]

        # robot emb
        if self.cfg.robot_state_dim > 0:
            if robot_state is None:
                robot_state = torch.zeros((B, self.cfg.robot_state_dim), device=device, dtype=dtype)
            else:
                assert robot_state.shape == (B, self.cfg.robot_state_dim), \
                    f"robot_state must be [B,robot_state_dim]=[{B},{self.cfg.robot_state_dim}], got {tuple(robot_state.shape)}"
                robot_state = robot_state.to(device=device, dtype=dtype)
            assert self.robot_proj is not None
            e_robot = self.robot_proj(robot_state)  # [B,d]
        else:
            e_robot = torch.zeros((B, self.cfg.tokenizer.d_model), device=device, dtype=dtype)

        # combined context embedding
        e_ctx = e_stage + e_obj
        if self.cfg.robot_state_dim > 0 and self.cfg.fuse_robot_into_ctx:
            e_ctx = e_ctx + e_robot

        # sample triplane at points
        tri_at_pts = self.sampler.sample(plane_xy, plane_xz, plane_yz, point_xyz_norm)  # [B,N,3C]

        # tokenize (global/points/tactile)
        tokens_wo_ctxprefix, _aux_tok = self.tokenizer(
            point_xyz_norm=point_xyz_norm,
            point_feats=point_feats,
            tactile_xyz_norm=tactile_xyz_norm,
            tactile_feats=tactile_feats,
            triplane_feats_at_points=tri_at_pts,
            ctx_emb=e_ctx
        )  # [B,S0,d] where tokens_wo_ctxprefix[:,0] is global_tok

        # prepend explicit ctx token
        ctx_tok = self.ctx_token_proj(e_ctx).unsqueeze(1)  # [B,1,d]
        prefix = [ctx_tok]

        # optional robot token (recommended)
        if self.cfg.robot_state_dim > 0 and self.cfg.use_robot_token:
            assert self.robot_token_proj is not None
            robot_tok = self.robot_token_proj(e_robot).unsqueeze(1)  # [B,1,d]
            prefix.append(robot_tok)

        tokens = torch.cat(prefix + [tokens_wo_ctxprefix], dim=1)  # [B,S,d]

        # encode
        memory = self.encoder(tokens)  # [B,S,d]

        # decide K
        if self.cfg.decoder.use_adaptive_chunking:
            K = self.decoder.select_chunk_size(memory)
        else:
            K = self.decoder.fixed_chunk_size_from_stage(stage)

        # decode
        if teacher_forcing:
            if actions_gt is None:
                raise ValueError("teacher_forcing=True requires actions_gt")
            pred_actions = self.decoder.decode_teacher_forcing(
                memory=memory, ctx_emb=e_ctx, actions_gt=actions_gt, K=K
            )
        else:
            pred_actions = self.decoder.decode_autoregressive(
                memory=memory, ctx_emb=e_ctx, K=K
            )

        # tri-plane update conditioning
        # token layout:
        #   idx 0: ctx_tok
        #   idx 1: robot_tok (if enabled)
        #   idx (1 or 2): global_tok from tokenizer
        global_index = 1 + (1 if (self.cfg.robot_state_dim > 0 and self.cfg.use_robot_token) else 0)
        global_out = memory[:, global_index, :]  # [B,d]
        pooled_a = self.action_pool(pred_actions).mean(dim=1)  # [B,d]

        if self.cfg.robot_state_dim > 0:
            cond = torch.cat([global_out, pooled_a, e_obj, e_robot], dim=-1)  # [B,4d]
        else:
            cond = torch.cat([global_out, pooled_a, e_obj], dim=-1)           # [B,3d]

        plane_xy2, plane_xz2, plane_yz2 = self.writer(plane_xy, plane_xz, plane_yz, cond)

        # render + success
        render_feat = self.renderer(plane_xy2, plane_xz2, plane_yz2)  # [B,3C]
        success_prob = self.success_head(render_feat)                 # [B]

        return {
            "pred_actions": pred_actions,
            "K": torch.tensor(K, device=device),
            "plane_xy_next": plane_xy2,
            "plane_xz_next": plane_xz2,
            "plane_yz_next": plane_yz2,
            "memory": memory,
            "render_feat": render_feat,
            "success_prob": success_prob,
            "e_stage": e_stage,
            "e_obj": e_obj,
            "e_robot": e_robot,
            "e_ctx": e_ctx,
        }


# -----------------------------
# Loss helpers
# -----------------------------

def action_mse_loss(pred_actions: torch.Tensor, gt_actions: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_actions, gt_actions)

def render_l1_loss(pred_render: torch.Tensor, gt_render: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred_render, gt_render)

def success_bce_loss(pred_success_prob: torch.Tensor, success_label: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy(pred_success_prob, success_label.float())
