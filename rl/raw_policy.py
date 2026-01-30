import os
import sys
import pathlib
import time
import copy
import pickle


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
from omegaconf import OmegaConf

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



def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert an HWC image to uint8 for video writing."""
    x = np.asarray(img)
    if x.dtype == np.uint8:
        return x

    # float images: assume either [0,1] or [0,255]
    if np.issubdtype(x.dtype, np.floating):
        mx = np.nanmax(x)
        if mx <= 1.0:
            x = x * 255.0

    x = np.clip(x, 0, 255).astype(np.uint8)
    return x


def save_rollout_video(render_images, out_dir: str, basename: str, fps: int = 10) -> str:
    """
    Save rollout locally (NOT to wandb).
    Tries mp4 (needs ffmpeg via imageio), falls back to gif, then to compressed npz.
    Returns the saved path.
    """
    os.makedirs(out_dir, exist_ok=True)
    frames = [_to_uint8(im) for im in render_images]

    mp4_path = os.path.join(out_dir, f"{basename}.mp4")
    gif_path = os.path.join(out_dir, f"{basename}.gif")
    npz_path = os.path.join(out_dir, f"{basename}.npz")

    # 1) Try MP4
    try:
        import imageio.v2 as imageio
        imageio.mimsave(mp4_path, frames, fps=fps)  # requires ffmpeg backend for mp4
        return mp4_path
    except Exception:
        pass

    # 2) Try GIF
    try:
        import imageio.v2 as imageio
        imageio.mimsave(gif_path, frames, fps=fps)
        return gif_path
    except Exception:
        pass

    # 3) Fallback: save raw frames (still “video”, just not encoded)
    np.savez_compressed(npz_path, video=np.stack(frames, axis=0))
    return npz_path


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


def _coerce_env_cfg(env_cfg):
    """
    Hydra/YAML often gives lists. Some env code expects tuples.
    Coerce pos/scale/euler to tuples if present.
    """
    if hasattr(env_cfg, "strawberry") and env_cfg.strawberry is not None:
        for k in ["pos", "scale", "euler"]:
            if k in env_cfg.strawberry and env_cfg.strawberry[k] is not None:
                v = env_cfg.strawberry[k]
                if isinstance(v, (list, tuple)):
                    env_cfg.strawberry[k] = tuple(float(x) for x in v)
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


        # Local video saving (NOT wandb upload)
        save_videos = bool(OmegaConf.select(self.cfg, "video.save", default=True))
        video_fps = int(OmegaConf.select(self.cfg, "video.fps", default=10))

        # Where to write videos:
        # - If config gives a path, use it
        # - Else default to <hydra_run_dir>/videos (this is OUTSIDE wandb.run.dir)
        video_dir = OmegaConf.select(self.cfg, "video.dir", default=None)
        if video_dir is None:
            video_dir = os.path.join(output_dir, "videos")
        if os.path.exists(save_path):
            print(f"Found existing {save_path}, skipping.")
            return

        # Build env from cfg.cut_env
        cut_env_cfg = _coerce_env_cfg(self.cfg.cut_env)
        taichi_env = get_strawberry_cuthalf_env(cut_env_cfg)
        knife = taichi_env.agent.effectors[0]

        # wandb
        # wandb
        wandb_run = None
        if bool(self.cfg.log_wandb):
            # Convert cfg.logging to a real dict (so we can safely insert defaults)
            wandb_kwargs = {}
            if hasattr(self.cfg, "logging") and self.cfg.logging is not None:
                wandb_kwargs = OmegaConf.to_container(self.cfg.logging, resolve=True)

            # Prefer cfg.wandb.project, otherwise cfg.wandb_project
            wandb_project = OmegaConf.select(self.cfg, "wandb.project", default=None)
            if wandb_project is None:
                wandb_project = OmegaConf.select(self.cfg, "wandb_project", default=None)

            # Only set project if not already provided in cfg.logging
            if wandb_project is not None and "project" not in wandb_kwargs:
                wandb_kwargs["project"] = wandb_project

            wandb_run = wandb.init(
                config=OmegaConf.to_container(self.cfg, resolve=True),
                **wandb_kwargs,
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
            render_images = fwd.get("render_images", [])

            # Save locally when rendered
            video_path = None
            if render and save_videos and len(render_images) > 0:
                video_path = save_rollout_video(
                    render_images=render_images,
                    out_dir=video_dir,
                    basename=f"iter_{iteration:04d}",
                    fps=video_fps,
                )

            # wandb log (NO video upload)
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

                # Log only the path as text (won't upload the file)
                if video_path is not None:
                    log_dict["video_path"] = video_path

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
