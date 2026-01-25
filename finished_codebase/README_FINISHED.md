# Finished codebase: multi-modal policy with 4 heads

This drop-in adds:

- Fourier positional encoding for:
  - joint angle vectors
  - point cloud xyz
  - tactile vectors (optional)
- PointNet-style point cloud encoder
- Joint + tactile MLP encoders
- Shared trunk
- **4 heads** (multi-task outputs)
- **4 separate losses** + weighted total loss

## Files

- `learning/model/model.py` – model + positional encoding + multi-head loss
- `learning/model/batch.py` – robust sample extraction + `collate_frames`
- `learning/model/train.py` – training script using your existing datasets
- `demo_dataloader.py` – smoke test / batch shape print

## Expected dataset sample formats

The collate tries to support:

1) `(obs_dict, action)`  
2) `{'obs': obs_dict, 'act': action}` or `{'obs': obs_dict, 'action': action}`  
3) Flat dict that includes `action` plus observation fields.

The observation dict should include *at least*:
- a joint vector under one of: `joint`, `joints`, `joint_angles`, `qpos`, ...
- a point cloud under one of: `point_cloud`, `pc`, `points`, ...
- tactile is optional: `tactile`, `tactile_state`, ...

Action can be:
- a tensor (D,) or (B,D) that will be split into 4 heads, or
- a dict of tensors with 4 keys (or more; you can pick the 4 you want via `--head_names`).

## Run

```bash
python demo_dataloader.py
python -m learning.model.train --obs_root path/to/obs_root --act_root path/to/act_root
```

If your action is not divisible into 4 meaningful chunks, pass explicit head dims:

```bash
python -m learning.model.train ... --head_dims 7,7,1,3
```
