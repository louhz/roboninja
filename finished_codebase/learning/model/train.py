"""Training entrypoint.

Example:
  python -m learning.model.train \
    --obs_root path/to/obs_root \
    --act_root path/to/act_root \
    --pc_glob '*.ply' \
    --num_points 2048 \
    --batch_size 8 \
    --epochs 50

This expects your project already has:
  from learning.model.dataloader import ArmHandPointCloudActionDataset, RightHandPointCloudActionDataset
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader

from .batch import collate_frames, to_device
from .loss import (
    FourierFeatures,
    MultiHeadLoss,
    MultiHeadLossConfig,
    MultiModalPolicy,
    infer_head_dims_from_action,
    split_action_vector,
)


def _infer_modal_dims(sample_batch: Dict[str, Any]) -> Tuple[int, int, int]:
    joint_dim = int(sample_batch["joint"].shape[-1])
    point_dim = int(sample_batch["point_cloud"].shape[-1])
    tactile = sample_batch.get("tactile", None)
    tactile_dim = int(tactile.shape[-1]) if torch.is_tensor(tactile) else 0
    return joint_dim, point_dim, tactile_dim


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--obs_root", type=str, required=True)
    p.add_argument("--act_root", type=str, required=True)
    p.add_argument("--pc_glob", type=str, default="*.ply")
    p.add_argument("--num_points", type=int, default=2048)
    p.add_argument("--features", type=str, default="xyz", help="comma-separated, e.g. xyz or xyz,rgb")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--loss", type=str, default="mse", choices=["mse", "smooth_l1", "l1"])
    p.add_argument("--head_dims", type=str, default="", help="Optional: 4 comma-separated ints")
    p.add_argument("--head_names", type=str, default="", help="Optional: 4 comma-separated strings")
    p.add_argument("--log_every", type=int, default=50)
    args = p.parse_args()

    # Import datasets from your existing codebase
    from learning.model.dataloader import ArmHandPointCloudActionDataset  # type: ignore

    features = tuple([f.strip() for f in args.features.split(",") if f.strip()])

    ds = ArmHandPointCloudActionDataset(
        obs_root=args.obs_root,
        act_root=args.act_root,
        pc_glob=args.pc_glob,
        num_points=args.num_points,
        features=features,
        align_mode="nearest",
        missing_policy="initial",
    )

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_frames, num_workers=0)

    # Peek one batch to infer dims
    batch0 = next(iter(dl))
    joint_dim, point_dim, tactile_dim = _infer_modal_dims(batch0)

    # Infer head dims from a sample action (or user override)
    action0 = batch0["action"]
    if args.head_dims.strip():
        head_dims = [int(x) for x in args.head_dims.split(",")]
        if len(head_dims) != 4:
            raise ValueError("--head_dims must have 4 comma-separated ints")
        head_names = [f"head{i}" for i in range(4)]
    else:
        # action0 may be dict or tensor; if dict, we infer names and dims
        head_names, head_dims = infer_head_dims_from_action(action0 if not torch.is_tensor(action0) else action0[0])

    if args.head_names.strip():
        head_names = [s.strip() for s in args.head_names.split(",")]
        if len(head_names) != 4:
            raise ValueError("--head_names must have 4 comma-separated strings")

    # Build positional encodings
    joint_pe = FourierFeatures(joint_dim, num_bands=6, max_freq=20.0, include_input=True)
    point_pe = FourierFeatures(3, num_bands=6, max_freq=10.0, include_input=True)
    tactile_pe = FourierFeatures(tactile_dim, num_bands=4, max_freq=10.0, include_input=True) if tactile_dim > 0 else None

    model = MultiModalPolicy(
        joint_dim=joint_dim,
        point_dim=point_dim,
        tactile_dim=tactile_dim,
        joint_pe=joint_pe,
        point_xyz_pe=point_pe,
        tactile_pe=tactile_pe,
        head_dims=head_dims,
        head_names=head_names,
    ).to(args.device)

    loss_mod = MultiHeadLoss(head_names, MultiHeadLossConfig(loss_type=args.loss)).to(args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    model.train()
    step = 0
    for epoch in range(args.epochs):
        for batch in dl:
            batch = to_device(batch, torch.device(args.device))
            preds = model(
                joint=batch["joint"],
                point_cloud=batch["point_cloud"],
                tactile=batch.get("tactile", None),
            )

            # Targets: dict of 4 tensors
            action = batch["action"]
            if isinstance(action, dict):
                # If action dict has more than 4 keys, we only use head_names
                targets = {hn: action[hn] for hn in head_names}
            else:
                targets = split_action_vector(action, head_dims, head_names)

            total_loss, loss_by_head = loss_mod(preds, targets)

            opt.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % args.log_every == 0:
                lh = ", ".join([f"{k}={v.item():.4f}" for k, v in loss_by_head.items()])
                print(f"epoch {epoch} step {step} total={total_loss.item():.4f} | {lh}")
            step += 1

    # Save checkpoint
    out_dir = os.path.join(os.getcwd(), "checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "multimodal_policy.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "head_names": head_names,
            "head_dims": head_dims,
            "joint_dim": joint_dim,
            "point_dim": point_dim,
            "tactile_dim": tactile_dim,
        },
        ckpt_path,
    )
    print("Saved checkpoint to", ckpt_path)


if __name__ == "__main__":
    main()
