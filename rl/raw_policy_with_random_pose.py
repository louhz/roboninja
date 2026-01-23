import os
import sys
import pathlib
import time
import copy
import pickle
import math


if __name__ == "__main__":
    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)
import os
import pathlib
import sys
import time
import copy
import pickle

import hydra
import numpy as np
from omegaconf import OmegaConf, open_dict

# Optional pretty prints + wandb
from colorama import Fore, Style
import wandb

from cut_simulation.configs.macros import *  # keep if your repo uses macros/resolvers
from cut_simulation.optimizer.optim import SGD, Adam, Momentum

from roboninja.env.tc_env import get_strawberry_cuthalf_env


OmegaConf.register_new_resolver("eval", eval)

OPTIMS = {
    "Adam": Adam,
    "Momentum": Momentum,
    "SGD": SGD,
}


# -----------------------------
# Your forward/backward (same)
# -----------------------------
def forward(taichi_env, init_state, comp_actions, grad_enabled, render, **kwargs):
    taichi_env.set_state(init_state, grad_enabled=grad_enabled)

    actions_v = comp_actions[:-1]
    action_p = comp_actions[-1]
    horizon_action = len(actions_v)

    taichi_env.apply_agent_action_p(action_p)
    render_images = []

    if render:
        render_images.append(taichi_env.render())

    for i in range(horizon_action):
        taichi_env.step(actions_v[i])
        if render:
            render_images.append(taichi_env.render())

    loss = taichi_env.get_loss()
    return {"loss": loss, "render_images": render_images}


def backward(taichi_env, comp_actions, **kwargs):
    taichi_env.reset_grad()
    taichi_env.get_loss_grad()

    actions_v = comp_actions[:-1]
    horizon_action = len(actions_v)

    for i in range(horizon_action - 1, -1, -1):
        taichi_env.step_grad(actions_v[i])

    grad = taichi_env.agent.get_grad(len(actions_v))
    grad[-1][1:] = 0  # keep only init_x trainable (same as your code)

    return {"grad": grad}


def _loss_scalar(loss_obj):
    """Safely print a scalar even if env returns dict losses."""
    if isinstance(loss_obj, dict):
        if "loss" in loss_obj:
            return float(loss_obj["loss"])
        # fallback: first numeric
        for v in loss_obj.values():
            if isinstance(v, (int, float, np.floating)):
                return float(v)
        return float("nan")
    return float(loss_obj)



# -----------------------------
# Random-pose infra (NEW)
# -----------------------------
def _pack_pos(order: str, x: float, y: float, z: float):
    """
    Packs physical (x,y,z) into the env-config list order.

    Examples:
      order="xyz" -> [x,y,z]
      order="xzy" -> [x,z,y]   (this matches your expert.yaml comment)
    """
    order = str(order)
    assert set(order) == set("xyz") and len(order) == 3, f"pos_order must be a permutation of xyz, got {order}"
    m = {"x": float(x), "y": float(y), "z": float(z)}
    return [m[c] for c in order]


def _unpack_pos(order: str, packed):
    """Inverse of _pack_pos(): returns physical (x,y,z) from a packed list/tuple."""
    order = str(order)
    assert set(order) == set("xyz") and len(order) == 3, f"pos_order must be a permutation of xyz, got {order}"
    p = [float(v) for v in packed]
    m = {order[i]: p[i] for i in range(3)}
    return m["x"], m["y"], m["z"]


def _get(cfg, key, default=None):
    return cfg[key] if (cfg is not None and key in cfg) else default


def apply_episode_randomization(cfg: OmegaConf):
    """
    Mutates cfg in-place to implement:
      - random table (x,y,yaw) in world square [-0.5,0.5]^2
      - strawberry sampled on tabletop in table-local XY, rotated into world
      - strawberry Z set to tabletop height + small offset

    This is designed for "one config = one episode":
      seed = base_seed + episode_id
    """
    rand = _get(cfg, "randomize", None)
    if rand is None:
        return
    if not bool(_get(rand, "enable", False)):
        return

    episode_id = int(_get(cfg, "episode_id", 0))
    base_seed = int(_get(cfg, "base_seed", 0))
    seed = base_seed + episode_id
    rng = np.random.default_rng(seed)

    # convention: how cut_env.pos is ordered
    pos_order = str(_get(rand, "pos_order", "xzy"))  # your expert.yaml uses (x,z,y) comment

    # --- table pose (world XY + yaw) ---
    xy_min = _get(rand, "table_world_xy_min", [-0.5, -0.5])
    xy_max = _get(rand, "table_world_xy_max", [0.5, 0.5])
    tx = float(rng.uniform(float(xy_min[0]), float(xy_max[0])))
    ty = float(rng.uniform(float(xy_min[1]), float(xy_max[1])))

    yaw_range = _get(rand, "table_yaw_deg", [-180.0, 180.0])
    table_yaw_deg = float(rng.uniform(float(yaw_range[0]), float(yaw_range[1])))

    table_z = float(_get(rand, "table_z", 0.0))

    # --- tabletop size in table-local coords ---
    # user statement: tabletop spans [-0.5,-0.5]..[0.5,0.5] meters => half extents [0.5,0.5]
    hx, hy = _get(rand, "table_top_half_extents", [0.5, 0.5])
    hx, hy = float(hx), float(hy)

    margin = float(_get(rand, "strawberry_margin", 0.03))

    # sample strawberry local offset on tabletop
    lx = float(rng.uniform(-hx + margin, hx - margin))
    ly = float(rng.uniform(-hy + margin, hy - margin))

    # rotate local offset by table yaw into world XY
    yaw = math.radians(table_yaw_deg)
    ox = lx * math.cos(yaw) - ly * math.sin(yaw)
    oy = lx * math.sin(yaw) + ly * math.cos(yaw)

    # table_top_z: if not provided, try to infer from current strawberry pos (keeps backward compat)
    strawberry_z_offset = float(_get(rand, "strawberry_z_offset", 0.02))
    table_top_z = _get(rand, "table_top_z", None)
    if table_top_z is None:
        try:
            sx0, sy0, sz0 = _unpack_pos(pos_order, cfg.cut_env.strawberry.pos)
            table_top_z = float(sz0 - table_z - strawberry_z_offset)
        except Exception:
            table_top_z = 0.0
    table_top_z = float(table_top_z)

    # strawberry world pose
    sx = float(tx + ox)
    sy = float(ty + oy)
    sz = float(table_z + table_top_z + strawberry_z_offset)

    # strawberry yaw on the table
    s_yaw_range = _get(rand, "strawberry_yaw_deg", [-180.0, 180.0])
    strawberry_yaw_deg = float(rng.uniform(float(s_yaw_range[0]), float(s_yaw_range[1])))

    # euler base + yaw axis index (defaults: yaw about z => axis 2)
    table_yaw_axis = int(_get(rand, "table_yaw_axis", 2))
    strawberry_yaw_axis = int(_get(rand, "strawberry_yaw_axis", 2))

    # base euler (keep strawberry alignment; just overwrite yaw component)
    t_base = list(_get(rand, "table_base_euler", [0.0, 0.0, 0.0]))
    s_base = list(_get(rand, "strawberry_base_euler", cfg.cut_env.strawberry.euler))

    if len(t_base) != 3 or len(s_base) != 3:
        raise ValueError("table_base_euler and strawberry_base_euler must be length-3 lists")

    t_base[table_yaw_axis] = table_yaw_deg
    s_base[strawberry_yaw_axis] = strawberry_yaw_deg

    # --- write results back into cfg (safe even with struct mode) ---
    with open_dict(cfg):
        cfg.seed = seed  # for logging/repro

        # ensure cut_env.table exists
        if "table" not in cfg.cut_env or cfg.cut_env.table is None:
            cfg.cut_env.table = OmegaConf.create({})

        cfg.cut_env.table.pos = _pack_pos(pos_order, tx, ty, table_z)
        cfg.cut_env.table.euler = [float(v) for v in t_base]

        cfg.cut_env.strawberry.pos = _pack_pos(pos_order, sx, sy, sz)
        cfg.cut_env.strawberry.euler = [float(v) for v in s_base]

        # optional: record the local tabletop sample (handy for debugging)
        cfg.randomize_result = {
            "episode_id": episode_id,
            "seed": seed,
            "table": {"x": tx, "y": ty, "z": table_z, "yaw_deg": table_yaw_deg},
            "strawberry": {
                "x": sx,
                "y": sy,
                "z": sz,
                "yaw_deg": strawberry_yaw_deg,
                "local_xy": [lx, ly],
            },
        }


def _coerce_env_cfg(env_cfg):
    """
    Hydra/YAML often gives lists. Some env code expects tuples.
    Coerce pos/scale/euler to tuples if present.
    """
    for obj_key in ["strawberry", "table"]:
        if hasattr(env_cfg, obj_key) and getattr(env_cfg, obj_key) is not None:
            obj = getattr(env_cfg, obj_key)
            for k in ["pos", "scale", "euler"]:
                if k in obj and obj[k] is not None:
                    v = obj[k]
                    if isinstance(v, (list, tuple)):
                        obj[k] = tuple(float(x) for x in v)
    return env_cfg

class OptimizationWorkspace:
    def __init__(self, cfg: OmegaConf):
        OmegaConf.resolve(cfg)
        self.cfg = cfg

    def log_console(self, taichi_env, knife, iteration, loss_obj, forward_s, backward_s, comp_actions):
        horizon_action = len(comp_actions) - 1
        try:
            final_rot = knife.theta_k[horizon_action]
        except Exception:
            final_rot = None

        loss_val = _loss_scalar(loss_obj)

        msg = (
            f"{Fore.WHITE}> {self.cfg.name}-{iteration:03d}  "
            f"{Fore.RED}{forward_s:.2f}+{backward_s:.2f}s  "
        )
        if final_rot is not None:
            msg += f"{Fore.GREEN}rot={Fore.YELLOW}{final_rot:.3f}  "
        msg += f"{Fore.CYAN}loss={Fore.WHITE}{loss_val:.6f}{Style.RESET_ALL}"
        print(msg)

    def run(self):
        horizon_action = int(self.cfg.horizon_action)
        render_gap = int(self.cfg.render_gap)

        # Hydra will run inside: hydra.run.dir (from config)
        output_dir = os.getcwd()
        save_path = os.path.join(output_dir, "optimization.pkl")

        if os.path.exists(save_path):
            print(f"Found existing {save_path}, skipping.")
            return

        # --- IMPORTANT: randomize episode pose BEFORE env build ---
        apply_episode_randomization(self.cfg)

        # Save resolved episode config (so each run dir captures the exact sampled pose)
        try:
            OmegaConf.save(self.cfg, os.path.join(output_dir, "episode_config.yaml"))
        except Exception as e:
            print(f"[warn] could not save episode_config.yaml: {e}")

        # Build env from cfg.cut_env
        cut_env_cfg = _coerce_env_cfg(self.cfg.cut_env)
        taichi_env = get_strawberry_cuthalf_env(cut_env_cfg)
        knife = taichi_env.agent.effectors[0]

        # wandb
        wandb_run = None
        if bool(self.cfg.log_wandb):
            wandb_run = wandb.init(
                config=OmegaConf.to_container(self.cfg, resolve=True),
                **self.cfg.logging,
            )

        # init actions from cfg
        init_action_p = np.array(self.cfg.init_action_p, dtype=np.float32)
        init_action_v = np.array([self.cfg.init_action_v] * horizon_action, dtype=np.float32)

        # keep your “push down” heuristic (same as old script)
        init_action_v[: int(horizon_action * 0.4), 1] = -0.3

        init_actions = np.concatenate([init_action_v, init_action_p[None]], axis=0)
        current_actions = init_actions.copy()

        # optimizer LR per component
        lr_action_p = np.ones_like(init_action_p) * float(self.cfg.optim.lr_action_p)
        lr_action_v = np.ones_like(init_action_v) * float(self.cfg.optim.lr_action_v)
        lr = np.concatenate([lr_action_v, lr_action_p[None]], axis=0)

        optim = OPTIMS[self.cfg.optim.type](init_actions.copy(), lr, self.cfg.optim)

        init_state = taichi_env.get_state()["state"]
        save_info = []

        for iteration in range(int(self.cfg.n_iters)):
            render = (render_gap != -1) and (iteration == 0 or ((iteration + 1) % render_gap == 0))

            # forward
            t0 = time.time()
            fwd = forward(
                taichi_env=taichi_env,
                init_state=init_state,
                comp_actions=current_actions,
                grad_enabled=True,
                render=render,
            )
            forward_s = time.time() - t0

            # backward
            t0 = time.time()
            bwd = backward(taichi_env=taichi_env, comp_actions=current_actions)
            backward_s = time.time() - t0

            loss_obj = fwd["loss"]
            grad = bwd["grad"]

            # console log
            self.log_console(taichi_env, knife, iteration, loss_obj, forward_s, backward_s, current_actions)

            # wandb log (only send video when rendered)
            if wandb_run is not None:
                log_dict = {
                    "loss": _loss_scalar(loss_obj),
                    "time/forward_s": forward_s,
                    "time/backward_s": backward_s,
                }
                if isinstance(loss_obj, dict):
                    for k, v in loss_obj.items():
                        if isinstance(v, (int, float, np.floating)):
                            log_dict[f"loss/{k}"] = float(v)

                render_images = fwd.get("render_images", [])
                if render and len(render_images) > 0:
                    video = np.stack(render_images).transpose(0, 3, 1, 2)
                    log_dict["video"] = wandb.Video(video, fps=10)

                wandb_run.log(log_dict, step=iteration)

            # save selected info (BEFORE the step, so loss aligns with comp_actions)
            iter_ctx = {
                "iter": iteration,
                "comp_actions": current_actions,
                "loss": loss_obj,
                "grad": grad,
                "render_images": fwd.get("render_images", []),
            }
            cur_save = {}
            for key in self.cfg.save_info_keys:
                cur_save[key] = copy.deepcopy(iter_ctx[key])
            save_info.append(cur_save)

            # step
            assert not np.isnan(np.mean(grad)), f"NaN grad at iter {iteration}"
            current_actions = optim.step(grad)

            # optional bounds clamp (in case your optimizer doesn’t clamp internally)
            if "bounds" in self.cfg.optim and self.cfg.optim.bounds is not None:
                lo, hi = self.cfg.optim.bounds
                current_actions = np.clip(current_actions, float(lo), float(hi))

        with open(save_path, "wb") as f:
            pickle.dump(save_info, f)

        if wandb_run is not None:
            wandb_run.finish()

        print(f"Saved: {save_path}")


# Keep the same repo-root bootstrap pattern your original script used
if __name__ == "__main__":
    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)


@hydra.main(version_base=None, config_path="../roboninja/config", config_name="expert")
def main(cfg: OmegaConf):
    ws = OptimizationWorkspace(cfg)
    ws.run()


if __name__ == "__main__":
    main()