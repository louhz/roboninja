#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train PPO (rsl_rl) on your Taichi strawberry cut-half env.

- Loads the same Hydra env config as simulation.ipynb:
    config_dir = roboninja/config/cut_env
    cfg = compose(config_name="cut_env_strawberry")
  and applies the same strawberry overrides (mesh_path/pos/scale/etc).

- Wraps taichi_env into an rsl_rl VecEnv-style interface:
    reset() -> (obs, extras)
    step(actions) -> (obs, rew, dones, extras)

Obs/action are fruit-centric to generalize across strawberry pose/scale.
"""

import os
import sys
import copy
import time
import json
import pathlib
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch

# -----------------------------
# Repo root helper (robust cwd)
# -----------------------------
def _find_repo_root(start: pathlib.Path) -> pathlib.Path:
    for p in [start] + list(start.parents):
        if (p / "roboninja").exists() and (p / "cut_simulation").exists():
            return p
    return start.parent

THIS_FILE = pathlib.Path(__file__).resolve()
ROOT_DIR = _find_repo_root(THIS_FILE)
sys.path.append(str(ROOT_DIR))
os.chdir(str(ROOT_DIR))

# -----------------------------
# Hydra config (same as notebook)
# -----------------------------
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from roboninja.env.tc_env import get_strawberry_cuthalf_env


def load_env_cfg(overrides=None):
    if overrides is None:
        overrides = []
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    config_dir = os.path.abspath("roboninja/config/cut_env")
    initialize_config_dir(config_dir=config_dir, version_base=None)
    cfg = compose(config_name="cut_env_strawberry", overrides=overrides)

    # ---- Copy the same overrides you set in simulation.ipynb ----
    # NOTE: update mesh_path to your local path, or pass via --override
    if not getattr(cfg.strawberry, "mesh_path", None):
        cfg.strawberry.mesh_path = "/home/louhz/Desktop/Rss/roboninja/generateddata/Generated_strawberry/0001/mesh.obj"
    cfg.strawberry.pos = (0.5, 0.1, 0.45)      # (x, z, y) in your comment; keep consistent with your env
    cfg.strawberry.scale = (0.25, 0.25, 0.25)
    cfg.strawberry.voxelize_res = 256
    cfg.strawberry.shell_radius_vox = 2
    cfg.strawberry.close_iters = 2
    cfg.strawberry.normalize = True
    cfg.strawberry.trim_percentile = 0.5
    cfg.strawberry.cache_voxels = True
    cfg.strawberry.euler = (0.0, 180.0, 0.0)
    cfg.auto_boundary = True

    # bone name like notebook (bone_idx=0 default)
    bone_idx = int(getattr(cfg, "bone_idx", 0))
    cfg.bone.name = f"bone_{bone_idx}"

    return cfg


# -----------------------------
# Fruit frame adapter
# -----------------------------
try:
    from scipy.spatial.transform import Rotation as _R
except Exception:
    _R = None


def _rot_xyz_deg(euler_deg_xyz):
    e = np.asarray(euler_deg_xyz, dtype=np.float32)
    if _R is None:
        rx, ry, rz = np.deg2rad(e)
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
        return (Rz @ Ry @ Rx).astype(np.float32)
    return _R.from_euler("xyz", e, degrees=True).as_matrix().astype(np.float32)


@dataclass
class FruitFrameAdapter:
    fruit_pos_w: np.ndarray
    fruit_euler_deg: np.ndarray
    fruit_scale: np.ndarray

    def __post_init__(self):
        self.fruit_pos_w = np.asarray(self.fruit_pos_w, dtype=np.float32)
        self.fruit_scale = np.asarray(self.fruit_scale, dtype=np.float32)
        self.R_wf = _rot_xyz_deg(self.fruit_euler_deg)  # fruit->world
        self.R_fw = self.R_wf.T                         # world->fruit

    def world_to_fruit_pos(self, p_w):
        p_w = np.asarray(p_w, dtype=np.float32)
        return (self.R_fw @ (p_w - self.fruit_pos_w)) / self.fruit_scale

    def fruit_to_world_pos(self, p_f):
        p_f = np.asarray(p_f, dtype=np.float32)
        return self.fruit_pos_w + self.R_wf @ (p_f * self.fruit_scale)

    def world_to_fruit_delta(self, d_w):
        d_w = np.asarray(d_w, dtype=np.float32)
        return (self.R_fw @ d_w) / self.fruit_scale

    def fruit_to_world_delta(self, d_f):
        d_f = np.asarray(d_f, dtype=np.float32)
        return self.R_wf @ (d_f * self.fruit_scale)


# -----------------------------
# Knife state helpers (best-effort)
# -----------------------------
def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "to_numpy"):
        return x.to_numpy()
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def get_knife_pos(knife, t_idx: int):
    for attr in ("x_k", "pos_k", "x", "pos", "position"):
        if hasattr(knife, attr):
            v = getattr(knife, attr)
            try:
                return _to_numpy(v[t_idx])
            except Exception:
                vv = _to_numpy(v)
                if vv.ndim >= 2:
                    return vv[-1]
                return vv
    raise AttributeError("Cannot find knife position. Add your field name to get_knife_pos().")


def get_knife_theta(knife, t_idx: int):
    for attr in ("theta_k", "theta", "rot", "rotation"):
        if hasattr(knife, attr):
            v = getattr(knife, attr)
            try:
                return float(_to_numpy(v[t_idx]))
            except Exception:
                return float(_to_numpy(v))
    return 0.0


def build_obs(knife, adapter: FruitFrameAdapter, t_idx: int, horizon: int, prev_pos_w):
    pos_w = get_knife_pos(knife, t_idx)
    pos_f = adapter.world_to_fruit_pos(pos_w)

    if prev_pos_w is None:
        dpos_f = np.zeros(3, dtype=np.float32)
    else:
        dpos_w = (pos_w - prev_pos_w).astype(np.float32)
        dpos_f = adapter.world_to_fruit_delta(dpos_w)

    theta = get_knife_theta(knife, t_idx)
    phase = np.array([t_idx / max(1, horizon)], dtype=np.float32)

    obs = np.concatenate(
        [
            pos_f.astype(np.float32),
            dpos_f.astype(np.float32),
            np.array([np.sin(theta), np.cos(theta)], dtype=np.float32),
            phase,
        ],
        axis=0,
    ).astype(np.float32)

    return obs, pos_w


# -----------------------------
# rsl_rl VecEnv wrapper (single-step reset)
# -----------------------------
class StrawberryTaichiVecEnv:
    """
    Minimal rsl_rl VecEnv-style wrapper (IsaacLab v1 style):
      - reset() -> (obs: torch.Tensor [N, num_obs], extras: dict)
      - step(actions) -> (obs, rew, dones, extras)
    """

    def __init__(
        self,
        cfg,
        num_envs=1,
        device="cuda:0",
        horizon_action=60,
        init_action_p_f=(-0.6, 0.155, 0.5),
        action_scale_f=(1.0, 1.0, 1.0),
        reward_scale=1.0,
        collision_done_threshold=None,
    ):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.max_episode_length = int(horizon_action)

        # Our policy outputs a 3D delta in fruit frame by default
        self.num_actions = 3

        # obs: pos_f(3) + dpos_f(3) + sincos(theta)(2) + phase(1) = 9
        self.num_obs = 9
        self.num_privileged_obs = 0

        self.init_action_p_f = np.asarray(init_action_p_f, dtype=np.float32)
        self.action_scale_f = np.asarray(action_scale_f, dtype=np.float32)
        self.reward_scale = float(reward_scale)
        self.collision_done_threshold = collision_done_threshold

        # create env instances
        self.envs = []
        self.knives = []
        self.init_states = []
        self.adapters = []

        for i in range(self.num_envs):
            cfg_i = copy.deepcopy(cfg)
            cfg_i.bone.name = f"{cfg.bone.name}_{i}"
            taichi_env = get_strawberry_cuthalf_env(cfg_i)
            knife = taichi_env.agent.effectors[0]

            adapter = FruitFrameAdapter(
                fruit_pos_w=np.array(cfg_i.strawberry.pos, dtype=np.float32),
                fruit_euler_deg=np.array(cfg_i.strawberry.euler, dtype=np.float32),
                fruit_scale=np.array(cfg_i.strawberry.scale, dtype=np.float32),
            )

            init_state = taichi_env.get_state()["state"]

            self.envs.append(taichi_env)
            self.knives.append(knife)
            self.init_states.append(init_state)
            self.adapters.append(adapter)

        # buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=torch.float32)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self.done_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)

        self._prev_pos_w = [None for _ in range(self.num_envs)]

        # IMPORTANT: runner often doesn't call reset first
        self.reset()

    def seed(self, seed: int = -1):
        if seed is None or seed < 0:
            seed = int(time.time())
        np.random.seed(seed)
        torch.manual_seed(seed)
        return seed

    def _reset_one(self, i: int):
        env = self.envs[i]
        knife = self.knives[i]
        adapter = self.adapters[i]

        env.set_state(self.init_states[i], grad_enabled=False)

        # set initial "p" in WORLD (env expects world coords in your notebook)
        init_action_p_w = adapter.fruit_to_world_pos(self.init_action_p_f)
        env.apply_agent_action_p(init_action_p_w)

        self.episode_length_buf[i] = 0
        self._prev_pos_w[i] = None

        obs_np, prev = build_obs(knife, adapter, t_idx=0, horizon=self.max_episode_length, prev_pos_w=None)
        self._prev_pos_w[i] = prev
        self.obs_buf[i] = torch.from_numpy(obs_np).to(self.device)

    def reset(self):
        for i in range(self.num_envs):
            self._reset_one(i)
            self.done_buf[i] = 0
            self.rew_buf[i] = 0.0

        extras = {"observations": {"policy": self.obs_buf}}
        return self.obs_buf, extras

    def get_observations(self):
        extras = {"observations": {"policy": self.obs_buf}}
        return self.obs_buf, extras

    def step(self, actions: torch.Tensor):
        """
        actions: torch.Tensor [num_envs, num_actions] in [-1, 1]
        returns: obs, rew, dones, extras
        """
        actions = actions.to(self.device)
        actions = torch.clamp(actions, -1.0, 1.0)

        episode_infos = []
        for i in range(self.num_envs):
            # same-step reset mode: if already done, reset and skip dynamics
            if int(self.done_buf[i].item()) == 1:
                self._reset_one(i)
                self.done_buf[i] = 0
                self.rew_buf[i] = 0.0
                continue

            env = self.envs[i]
            knife = self.knives[i]
            adapter = self.adapters[i]

            act_f = actions[i].detach().cpu().numpy().astype(np.float32)
            act_f = np.clip(act_f, -1.0, 1.0) * self.action_scale_f
            act_w = adapter.fruit_to_world_delta(act_f)

            env.step(act_w)

            # update step counter
            self.episode_length_buf[i] += 1
            t_idx = int(self.episode_length_buf[i].item())

            # compute reward from env loss
            loss_dict = env.get_loss()
            total_loss = float(loss_dict.get("loss", 0.0))
            rew = -total_loss * self.reward_scale

            # optional early terminate on collision_loss
            done = False
            if not np.isfinite(total_loss):
                done = True
            if self.collision_done_threshold is not None and "collision_loss" in loss_dict:
                if float(loss_dict["collision_loss"]) > float(self.collision_done_threshold):
                    done = True
            if t_idx >= self.max_episode_length:
                done = True

            self.rew_buf[i] = float(rew)
            self.done_buf[i] = 1 if done else 0

            # next obs (best-effort index)
            try:
                obs_np, prev = build_obs(knife, adapter, t_idx=t_idx, horizon=self.max_episode_length, prev_pos_w=self._prev_pos_w[i])
            except Exception:
                # fallback: read last state
                obs_np, prev = build_obs(knife, adapter, t_idx=-1, horizon=self.max_episode_length, prev_pos_w=self._prev_pos_w[i])
            self._prev_pos_w[i] = prev
            self.obs_buf[i] = torch.from_numpy(obs_np).to(self.device)

            if done:
                episode_infos.append({"r": float(rew), "l": int(t_idx)})

        extras = {"observations": {"policy": self.obs_buf}}
        if len(episode_infos) > 0:
            # common convention used by many runners for episodic stats
            extras["episode"] = episode_infos

        return self.obs_buf, self.rew_buf, self.done_buf, extras

    def close(self):
        # If your taichi env has a close() call it here
        return


# -----------------------------
# rsl_rl training
# -----------------------------
def make_train_cfg(
    num_steps_per_env=60,
    max_iterations=2000,
    save_interval=100,
    experiment_name="ppo_strawberry",
    device="cuda:0",
):
    """
    rsl_rl expects a nested dict with "algorithm", "policy", and "runner" keys in many examples. :contentReference[oaicite:2]{index=2}
    """
    train_cfg = {
        "algorithm": {
            "class_name": "PPO",
            "seed": 1,
            "device": device,
            "num_steps_per_env": int(num_steps_per_env),
            "max_iterations": int(max_iterations),
            "save_interval": int(save_interval),

            # PPO/GAE basics
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "lam": 0.95,
            "clip_param": 0.2,
            "entropy_coef": 0.0,
            "value_loss_coef": 1.0,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "desired_kl": 0.01,
        },
        "policy": {
            "class_name": "ActorCritic",
            "activation": "elu",
            "init_noise_std": 0.5,
            "actor_hidden_dims": [256, 256],
            "critic_hidden_dims": [256, 256],
        },
        "runner": {
            "class_name": "OnPolicyRunner",
            "experiment_name": experiment_name,
        },
    }
    return train_cfg


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--horizon_action", type=int, default=60)
    parser.add_argument("--max_iterations", type=int, default=2000)
    parser.add_argument("--num_steps_per_env", type=int, default=60)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--log_dir", type=str, default="runs/ppo_strawberry")
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--collision_done_threshold", type=float, default=-1.0)  # -1 disables
    parser.add_argument("--mesh_path", type=str, default="")
    parser.add_argument("--override", action="append", default=[], help="Hydra override, e.g. strawberry.mesh_path=/abs/mesh.obj")

    args = parser.parse_args()

    overrides = list(args.override)
    if args.mesh_path:
        overrides.append(f"strawberry.mesh_path={args.mesh_path}")

    cfg = load_env_cfg(overrides=overrides)

    collision_thr = None if args.collision_done_threshold < 0 else float(args.collision_done_threshold)

    env = StrawberryTaichiVecEnv(
        cfg=cfg,
        num_envs=args.num_envs,
        device=args.device,
        horizon_action=args.horizon_action,
        init_action_p_f=(-0.6, 0.155, 0.5),
        action_scale_f=(1.0, 1.0, 1.0),
        reward_scale=args.reward_scale,
        collision_done_threshold=collision_thr,
    )

    # build train cfg for rsl_rl runner
    exp_name = f"ppo_strawberry_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    train_cfg = make_train_cfg(
        num_steps_per_env=args.num_steps_per_env,
        max_iterations=args.max_iterations,
        save_interval=args.save_interval,
        experiment_name=exp_name,
        device=args.device,
    )

    # logging directory
    log_dir = os.path.join(args.log_dir, exp_name)
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "env_cfg.json"), "w") as f:
        json.dump({"strawberry": {
            "mesh_path": str(cfg.strawberry.mesh_path),
            "pos": list(cfg.strawberry.pos),
            "scale": list(cfg.strawberry.scale),
            "euler": list(cfg.strawberry.euler),
        }}, f, indent=2)

    try:
        from rsl_rl.runners import OnPolicyRunner
    except Exception as e:
        raise RuntimeError(
            "Failed to import rsl_rl. Make sure you installed the same rsl_rl package you used before.\n"
            "Typical install: pip install -e path/to/rsl_rl"
        ) from e

    # runner signature differs slightly across versions; handle both
    try:
        runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=args.device)
    except TypeError:
        runner = OnPolicyRunner(env, train_cfg, log_dir, device=args.device)

    # many examples call: learn(num_learning_iterations=..., init_at_random_ep_len=True) :contentReference[oaicite:3]{index=3}
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    main()
