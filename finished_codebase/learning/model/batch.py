"""learning.model.batch

Helpers to collate dataset samples into a single batch.

Your datasets may return many formats (tuple, dict, nested dict). These helpers try to
standardize into a dict with:
  - 'joint': (B, J)
  - 'point_cloud': (B, N, C)
  - 'tactile': (B, T) or None
  - 'action': either (B, D) Tensor OR dict of 4 tensors (B, d_i)

If your dataset uses different keys, adjust KEY_CANDIDATES below or pass an explicit
key mapping in `extract_sample(..., keys=...)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

Tensor = torch.Tensor


# Key candidates (ordered from most-likely to less-likely)
JOINT_KEYS = ("joint", "joints", "joint_angles", "qpos", "arm_qpos", "hand_qpos", "angles")
PC_KEYS = ("point_cloud", "pc", "points", "cloud", "xyz")
TACTILE_KEYS = ("tactile", "tactile_state", "taxel", "tactile_obs", "touch")
ACTION_KEYS = ("action", "act", "actions", "y", "target")


def _to_tensor(x: Any, dtype: torch.dtype = torch.float32) -> Tensor:
    if x is None:
        raise ValueError("Cannot convert None to tensor")
    if torch.is_tensor(x):
        return x.to(dtype=dtype)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(dtype=dtype)
    # list/tuple of numbers
    return torch.tensor(x, dtype=dtype)


def _find_first(d: Mapping[str, Any], keys: Sequence[str]) -> Tuple[str, Any]:
    for k in keys:
        if k in d:
            return k, d[k]
    raise KeyError(f"None of the keys {keys} found in dict. Available keys: {list(d.keys())}")


def _is_mapping(x: Any) -> bool:
    return isinstance(x, Mapping)


def extract_sample(sample: Any) -> Dict[str, Any]:
    """Extract (joint, point_cloud, tactile, action) from one dataset sample."""
    obs = None
    act = None

    # Common patterns:
    # 1) (obs, act)
    # 2) {'obs': ..., 'act': ...}
    # 3) {'...obs fields...', 'action': ...}
    if isinstance(sample, (tuple, list)) and len(sample) == 2:
        obs, act = sample
    elif _is_mapping(sample) and ("obs" in sample and ("act" in sample or "action" in sample)):
        obs = sample["obs"]
        act = sample.get("act", sample.get("action"))
    elif _is_mapping(sample):
        # if it's a flat dict, try to pick action and treat the rest as obs
        for ak in ACTION_KEYS:
            if ak in sample:
                act = sample[ak]
                break
        obs = sample
    else:
        raise TypeError(f"Unsupported sample type: {type(sample)}")

    if not _is_mapping(obs):
        raise TypeError(f"obs must be dict-like, got {type(obs)}")

    _, joint = _find_first(obs, JOINT_KEYS)
    _, pc = _find_first(obs, PC_KEYS)
    tactile = None
    for tk in TACTILE_KEYS:
        if tk in obs:
            tactile = obs[tk]
            break

    if act is None:
        # try find in obs if not already found
        for ak in ACTION_KEYS:
            if ak in obs:
                act = obs[ak]
                break
    if act is None:
        raise KeyError("Could not find action in sample.")

    # Convert to tensors (but keep dict action if it's already dict-like)
    joint_t = _to_tensor(joint)
    pc_t = _to_tensor(pc)
    tactile_t = _to_tensor(tactile) if tactile is not None else None

    if _is_mapping(act):
        act_dict = {k: _to_tensor(v) for k, v in act.items()}
        act_out: Union[Tensor, Dict[str, Tensor]] = act_dict
    else:
        act_out = _to_tensor(act)

    return {
        "joint": joint_t,
        "point_cloud": pc_t,
        "tactile": tactile_t,
        "action": act_out,
    }


def collate_frames(frames: List[Any]) -> Dict[str, Any]:
    """Stacks a list of samples into a batch."""
    parsed = [extract_sample(s) for s in frames]

    joint = torch.stack([p["joint"] for p in parsed], dim=0)
    pc = torch.stack([p["point_cloud"] for p in parsed], dim=0)

    tactile_list = [p["tactile"] for p in parsed]
    tactile = None
    if all(t is None for t in tactile_list):
        tactile = None
    elif any(t is None for t in tactile_list):
        raise ValueError("Some samples have tactile and others do not. Make dataset consistent.")
    else:
        tactile = torch.stack(tactile_list, dim=0)

    action0 = parsed[0]["action"]
    if isinstance(action0, dict):
        # stack per key
        keys = list(action0.keys())
        action = {k: torch.stack([p["action"][k] for p in parsed], dim=0) for k in keys}
    else:
        action = torch.stack([p["action"] for p in parsed], dim=0)

    return {
        "joint": joint,
        "point_cloud": pc,
        "tactile": tactile,
        "action": action,
    }


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = {kk: vv.to(device) for kk, vv in v.items()}
        else:
            out[k] = v
    return out
