"""Quick dataset smoke test + batch collation.

Run:
  python demo_dataloader.py

Edit obs_root / act_root to point at your data.
"""

from torch.utils.data import DataLoader

from learning.model.dataloader import ArmHandPointCloudActionDataset, RightHandPointCloudActionDataset
from learning.model.batch import collate_frames


if __name__ == "__main__":
    ds_all = ArmHandPointCloudActionDataset(
        obs_root="path/to/obs_root",
        act_root="path/to/act_root",
        pc_glob="*.ply",  # or "*.pcd"
        num_points=2048,  # make point clouds stackable
        features=("xyz",),  # or ("xyz","rgb") if you want colors too
        align_mode="nearest",
        missing_policy="initial",
    )

    ds_rh = RightHandPointCloudActionDataset(
        obs_root="path/to/obs_root",
        act_root="path/to/act_root",
        pc_glob="*.ply",
        num_points=2048,
        features=("xyz",),
    )

    dl = DataLoader(ds_all, batch_size=8, shuffle=False, collate_fn=collate_frames)
    batch = next(iter(dl))
    print("Batch keys:", batch.keys())
    print("joint:", batch["joint"].shape)
    print("point_cloud:", batch["point_cloud"].shape)
    print("tactile:", None if batch["tactile"] is None else batch["tactile"].shape)
    if isinstance(batch["action"], dict):
        print("action dict keys:", list(batch["action"].keys()))
        for k, v in batch["action"].items():
            print(" ", k, v.shape)
    else:
        print("action:", batch["action"].shape)
