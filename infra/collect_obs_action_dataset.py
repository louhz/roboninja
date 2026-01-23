#!/usr/bin/env python3
"""collect_obs_action_dataset.py

Generate expert trajectories and save (observation, action) pairs for imitation.

This script is designed to work with your existing `raw_policy.py` expert
optimizer (uploaded) and your Hydra config (`expert.yaml`).

What it does (per mesh instance / episode):
  1) Compose an env config (based on expert.yaml)
  2) Build the Taichi env with `get_strawberry_cuthalf_env(cfg.cut_env)`
  3) Obtain an action sequence using ONE of:
       - "optimize": run the same gradient-based optimization as raw_policy.py
       - "load_pkl": load an existing optimization.pkl and take best actions
  4) Roll out the env with those actions and record observations + actions.
  5) Save as a pickle with a list of episodes.

Notes / assumptions:
  - Observations are taken from `env.get_obs()` if present, else from
    `env.get_state()`.
  - Actions are recorded per-step as the velocity actions (actions_v).
    The final row of `comp_actions` (action_p) is saved as metadata.

Typical usage:
  # From your repo root
  python collect_obs_action_dataset.py \
    --config /path/to/roboninja/config/expert.yaml \
    --mesh_root /home/louhz/Desktop/Rss/roboninja/generateddata/Generated_strawberry \
    --out dataset_strawberry_expert.pkl \
    --mode optimize

If you already ran raw_policy.py and have optimization.pkl per mesh:
  python collect_obs_action_dataset.py ... --mode load_pkl --pkl_root data/optimization
"""

from __future__ import annotations

import argparse
import copy
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from omegaconf import OmegaConf


NUMERIC_DIR_RE = re.compile(r"^\d+$")


def _is_numeric_dir(p: Path) -> bool:
    return p.is_dir() and bool(NUMERIC_DIR_RE.match(p.name))


def _sorted_numeric_dirs(root: Path) -> List[Path]:
    dirs = [p for p in root.iterdir() if _is_numeric_dir(p)]
    return sorted(dirs, key=lambda p: int(p.name))


def _to_numpy_safe(x: Any) -> Any:
    """Best-effort conversion to numpy / python primitives for serialization."""
    if x is None:
        return None
    if isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, np.ndarray):
        return x
    # Taichi fields or tensors sometimes expose to_numpy()
    to_np = getattr(x, "to_numpy", None)
    if callable(to_np):
        try:
            return to_np()
        except Exception:
            pass
    if isinstance(x, dict):
        return {k: _to_numpy_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_numpy_safe(v) for v in x)
    # Fallback: try numpy.array
    try:
        return np.array(x)
    except Exception:
        return x


def _get_obs(env: Any) -> Any:
    """Prefer env.get_obs() if available, else use env.get_state()."""
    if hasattr(env, "get_obs") and callable(getattr(env, "get_obs")):
        return env.get_obs()
    if hasattr(env, "get_state") and callable(getattr(env, "get_state")):
        return env.get_state()
    raise AttributeError("Env has neither get_obs() nor get_state()")


def _coerce_env_cfg(cfg: Any) -> Any:
    # Mirror the helper inside raw_policy.py
    if hasattr(cfg, "strawberry") and cfg.strawberry is not None:
        for k in ["pos", "scale", "euler"]:
            if k in cfg.strawberry and cfg.strawberry[k] is not None:
                v = cfg.strawberry[k]
                if isinstance(v, (list, tuple)):
                    cfg.strawberry[k] = tuple(float(x) for x in v)
    return cfg


def _loss_scalar(loss_obj: Any) -> float:
    if isinstance(loss_obj, dict):
        if "loss" in loss_obj:
            return float(loss_obj["loss"])
        for v in loss_obj.values():
            if isinstance(v, (int, float, np.floating)):
                return float(v)
        return float("nan")
    return float(loss_obj)


def _load_cfg(config_path: Path) -> Any:
    """Load a Hydra-style config.

    If hydra-core is available, we try to *compose* so `defaults:` works.
    Otherwise we fall back to a plain OmegaConf.load.
    """
    config_path = config_path.resolve()
    try:
        from hydra import compose, initialize_config_dir  # type: ignore

        with initialize_config_dir(config_dir=str(config_path.parent), job_name="collect_dataset"):
            cfg = compose(config_name=config_path.stem)
        OmegaConf.resolve(cfg)
        return cfg
    except Exception as e:
        print(f"[WARN] Hydra compose failed, falling back to OmegaConf.load: {e}")
        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        return cfg


def _import_raw_policy_module(raw_policy_path: Path):
    """Import raw_policy.py by filesystem path."""
    import importlib.util

    # Try to mimic the repo-root bootstrap raw_policy.py normally does when run
    # as __main__. When imported, those blocks don't execute, so we help PYTHONPATH.
    raw_policy_path = raw_policy_path.resolve()
    inferred_root = raw_policy_path.parent
    for _ in range(3):
        inferred_root = inferred_root.parent
    # Heuristic: if this looks like a repo root (contains roboninja/), add it.
    if (inferred_root / "roboninja").exists() and str(inferred_root) not in sys.path:
        sys.path.insert(0, str(inferred_root))
        os.chdir(inferred_root)

    spec = importlib.util.spec_from_file_location("raw_policy", str(raw_policy_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {raw_policy_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def _optimize_actions(raw_policy_mod: Any, cfg: Any) -> np.ndarray:
    """Run the same optimization as raw_policy.py, but return best comp_actions."""
    # Build env
    cut_env_cfg = _coerce_env_cfg(cfg.cut_env)
    env = raw_policy_mod.get_strawberry_cuthalf_env(cut_env_cfg)

    horizon_action = int(cfg.horizon_action)
    init_action_p = np.array(cfg.init_action_p, dtype=np.float32)
    init_action_v = np.array([cfg.init_action_v] * horizon_action, dtype=np.float32)
    init_action_v[: int(horizon_action * 0.4), 1] = -0.3
    init_actions = np.concatenate([init_action_v, init_action_p[None]], axis=0)
    current_actions = init_actions.copy()

    # Optimizer
    lr_action_p = np.ones_like(init_action_p) * float(cfg.optim.lr_action_p)
    lr_action_v = np.ones_like(init_action_v) * float(cfg.optim.lr_action_v)
    lr = np.concatenate([lr_action_v, lr_action_p[None]], axis=0)
    optim_cls = raw_policy_mod.OPTIMS[cfg.optim.type]
    optim = optim_cls(init_actions.copy(), lr, cfg.optim)

    st0 = env.get_state()
    init_state = st0["state"] if isinstance(st0, dict) and "state" in st0 else st0

    best_loss = float("inf")
    best_actions = current_actions.copy()

    for _it in range(int(cfg.n_iters)):
        fwd = raw_policy_mod.forward(
            taichi_env=env,
            init_state=init_state,
            comp_actions=current_actions,
            grad_enabled=True,
            render=False,
        )
        bwd = raw_policy_mod.backward(taichi_env=env, comp_actions=current_actions)
        loss_val = _loss_scalar(fwd["loss"])
        if loss_val < best_loss:
            best_loss = loss_val
            best_actions = current_actions.copy()

        grad = bwd["grad"]
        current_actions = optim.step(grad)
        if "bounds" in cfg.optim and cfg.optim.bounds is not None:
            lo, hi = cfg.optim.bounds
            current_actions = np.clip(current_actions, float(lo), float(hi))

    return best_actions


def _load_best_actions_from_pkl(pkl_path: Path) -> np.ndarray:
    with open(pkl_path, "rb") as f:
        save_info = pickle.load(f)
    # Each element has keys like iter, comp_actions, loss
    best = None
    best_loss = float("inf")
    for it in save_info:
        loss_val = _loss_scalar(it.get("loss"))
        if loss_val < best_loss:
            best_loss = loss_val
            best = it
    if best is None:
        raise ValueError(f"No iterations found in {pkl_path}")
    return np.array(best["comp_actions"], dtype=np.float32)


def _rollout_episode(raw_policy_mod: Any, cfg: Any, comp_actions: np.ndarray) -> Dict[str, Any]:
    cut_env_cfg = _coerce_env_cfg(cfg.cut_env)
    env = raw_policy_mod.get_strawberry_cuthalf_env(cut_env_cfg)

    actions_v = comp_actions[:-1]
    action_p = comp_actions[-1]

    # Apply action_p once (same as forward())
    if hasattr(env, "apply_agent_action_p"):
        env.apply_agent_action_p(action_p)

    obs_list: List[Any] = []
    act_list: List[Any] = []

    # Record initial obs
    obs_list.append(_to_numpy_safe(copy.deepcopy(_get_obs(env))))

    for i in range(len(actions_v)):
        a = actions_v[i]
        env.step(a)
        act_list.append(np.array(a, dtype=np.float32))
        obs_list.append(_to_numpy_safe(copy.deepcopy(_get_obs(env))))

    # Loss / metrics (optional)
    loss_obj = env.get_loss() if hasattr(env, "get_loss") else None
    episode = {
        "mesh_path": str(cfg.cut_env.strawberry.mesh_path) if hasattr(cfg.cut_env, "strawberry") else None,
        "init_action_p": np.array(action_p, dtype=np.float32),
        "actions": np.stack(act_list, axis=0) if len(act_list) > 0 else np.zeros((0,), dtype=np.float32),
        "observations": obs_list,
        "loss": _to_numpy_safe(loss_obj),
    }
    return episode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True, help="Path to expert.yaml")
    ap.add_argument(
        "--raw_policy",
        type=Path,
        default=Path("raw_policy.py"),
        help="Path to raw_policy.py (expert optimizer)",
    )
    ap.add_argument(
        "--mesh_root",
        type=Path,
        required=True,
        help="Root folder containing numbered subfolders with mesh.obj",
    )
    ap.add_argument(
        "--mesh_obj_name",
        default="mesh.obj",
        help="Obj filename inside each numbered folder (default: mesh.obj)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output pickle file path (episodes list)",
    )
    ap.add_argument(
        "--mode",
        choices=["optimize", "load_pkl"],
        default="optimize",
        help="How to obtain actions for each mesh",
    )
    ap.add_argument(
        "--pkl_root",
        type=Path,
        default=None,
        help="If mode=load_pkl: root directory containing per-mesh optimization.pkl",
    )
    ap.add_argument(
        "--max_episodes",
        type=int,
        default=-1,
        help="Limit number of episodes (-1 means all)",
    )
    args = ap.parse_args()

    cfg = _load_cfg(args.config)

    raw_policy_mod = _import_raw_policy_module(args.raw_policy)

    mesh_dirs = _sorted_numeric_dirs(args.mesh_root)
    if args.max_episodes is not None and args.max_episodes > 0:
        mesh_dirs = mesh_dirs[: args.max_episodes]

    episodes: List[Dict[str, Any]] = []

    for d in mesh_dirs:
        mesh_path = d / args.mesh_obj_name
        if not mesh_path.exists():
            print(f"[WARN] Missing mesh obj: {mesh_path}  (skip)")
            continue

        # Override mesh path in cfg for this episode
        cfg_ep = copy.deepcopy(cfg)
        if "cut_env" not in cfg_ep:
            raise KeyError("Config missing cut_env")
        if "strawberry" not in cfg_ep.cut_env:
            raise KeyError("Config missing cut_env.strawberry")
        cfg_ep.cut_env.strawberry.mesh_path = str(mesh_path)

        if args.mode == "optimize":
            comp_actions = _optimize_actions(raw_policy_mod, cfg_ep)
        else:
            if args.pkl_root is None:
                raise SystemExit("--pkl_root is required when --mode load_pkl")
            # Expected layout: pkl_root/<mesh_dir_name>/optimization.pkl OR similar
            candidate1 = args.pkl_root / d.name / "optimization.pkl"
            candidate2 = args.pkl_root / "optimization.pkl"  # fallback
            pkl_path = candidate1 if candidate1.exists() else candidate2
            if not pkl_path.exists():
                print(f"[WARN] Missing optimization.pkl for {d.name}: tried {candidate1} and {candidate2}")
                continue
            comp_actions = _load_best_actions_from_pkl(pkl_path)

        ep = _rollout_episode(raw_policy_mod, cfg_ep, comp_actions)
        ep["mesh_id"] = d.name
        episodes.append(ep)
        print(f"Collected episode {d.name}: T={len(ep['actions'])}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(episodes, f)

    print(f"Saved dataset: {args.out}  episodes={len(episodes)}")


if __name__ == "__main__":
    main()
