#!/usr/bin/env python3
import os
import sys
import time
import copy
import pickle
import argparse
from pathlib import Path

import numpy as np
import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from colorama import Fore, Style
import wandb

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





def find_repo_root(start: Path) -> Path:
    """
    Walk up from `start` until we find a directory that looks like the repo root.
    Adjust the heuristics if your repo layout differs.
    """
    for p in [start] + list(start.parents):
        if (p / "roboninja").is_dir() and (p / "cut_simulation").is_dir():
            return p
    return start.parent


def _loss_scalar(loss_obj) -> float:
    """Safely coerce env loss to a scalar for comparison/logging."""
    if isinstance(loss_obj, dict):
        if "loss" in loss_obj:
            return float(loss_obj["loss"])
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


def _to_uint8(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img)
    if x.dtype == np.uint8:
        return x
    if np.issubdtype(x.dtype, np.floating):
        mx = np.nanmax(x)
        if mx <= 1.0:
            x = x * 255.0
    x = np.clip(x, 0, 255).astype(np.uint8)
    return x


def save_rollout_video(render_images, out_dir: str, basename: str, fps: int = 10) -> str:
    """
    Save rollout locally (NOT to wandb).
    Tries mp4, falls back to gif, then to compressed npz.
    Returns the saved path.
    """
    os.makedirs(out_dir, exist_ok=True)
    frames = [_to_uint8(im) for im in render_images]

    mp4_path = os.path.join(out_dir, f"{basename}.mp4")
    gif_path = os.path.join(out_dir, f"{basename}.gif")
    npz_path = os.path.join(out_dir, f"{basename}.npz")

    try:
        import imageio.v2 as imageio
        imageio.mimsave(mp4_path, frames, fps=fps)
        return mp4_path
    except Exception:
        pass

    try:
        import imageio.v2 as imageio
        imageio.mimsave(gif_path, frames, fps=fps)
        return gif_path
    except Exception:
        pass

    np.savez_compressed(npz_path, video=np.stack(frames, axis=0))
    return npz_path


# -----------------------------
# Forward / backward (same logic)
# -----------------------------
def forward(taichi_env, init_state, comp_actions, grad_enabled, render):
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


def backward(taichi_env, comp_actions):
    taichi_env.reset_grad()
    taichi_env.get_loss_grad()

    actions_v = comp_actions[:-1]
    horizon_action = len(actions_v)

    for i in range(horizon_action - 1, -1, -1):
        taichi_env.step_grad(actions_v[i])

    grad = taichi_env.agent.get_grad(len(actions_v))
    grad[-1][1:] = 0  # keep only init_x trainable (same as your code)
    return {"grad": grad}


def log_console(cfg, taichi_env, knife, iteration, loss_obj, forward_s, backward_s, comp_actions):
    horizon_action = len(comp_actions) - 1
    try:
        final_rot = knife.theta_k[horizon_action]
    except Exception:
        final_rot = None

    loss_val = _loss_scalar(loss_obj)

    msg = (
        f"{Fore.WHITE}> {cfg.name}-{iteration:03d}  "
        f"{Fore.RED}{forward_s:.2f}+{backward_s:.2f}s  "
    )
    if final_rot is not None:
        msg += f"{Fore.GREEN}rot={Fore.YELLOW}{final_rot:.3f}  "
    msg += f"{Fore.CYAN}loss={Fore.WHITE}{loss_val:.6f}{Style.RESET_ALL}"
    print(msg)


# -----------------------------
# Core runner: one mesh
# -----------------------------
def run_one_mesh(cfg, run_dir: Path, mesh_id: str, mesh_path: Path,
                 skip_existing: bool = True) -> dict:
    """
    Runs optimization for a single mesh and saves:
      - optimization.pkl (checkpoint)
      - final_actions.npy
      - best_actions.npy
      - mesh_action_pair.pkl (mesh_path + actions)
      - resolved_config.yaml
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = run_dir / "optimization.pkl"
    final_actions_path = run_dir / "final_actions.npy"
    best_actions_path = run_dir / "best_actions.npy"
    pair_path = run_dir / "mesh_action_pair.pkl"
    resolved_cfg_path = run_dir / "resolved_config.yaml"

    if skip_existing and pair_path.exists() and final_actions_path.exists() and checkpoint_path.exists():
        print(f"[SKIP] {mesh_id} already has outputs in {run_dir}")
        return {
            "mesh_id": mesh_id,
            "mesh_path": str(mesh_path),
            "run_dir": str(run_dir),
            "checkpoint_path": str(checkpoint_path),
            "final_actions_path": str(final_actions_path),
            "best_actions_path": str(best_actions_path),
            "pair_path": str(pair_path),
            "skipped": True,
        }

    # Override mesh_path + video dir BEFORE resolve
    cfg = copy.deepcopy(cfg)
    cfg.cut_env.strawberry.mesh_path = str(mesh_path)

    # Make name unique per mesh (helps wandb + logs)
    base_name = str(cfg.name)
    cfg.name = f"{base_name}_{mesh_id}"

    # Ensure video.dir does NOT depend on ${hydra:runtime.output_dir}
    if "video" in cfg and cfg.video is not None:
        cfg.video.dir = str(run_dir / "videos")

    # Optionally append mesh id to wandb tags
    if hasattr(cfg, "logging") and cfg.logging is not None and "tags" in cfg.logging:
        tags = list(cfg.logging.tags) if cfg.logging.tags is not None else []
        if mesh_id not in tags:
            tags.append(mesh_id)
        cfg.logging.tags = tags

    # Resolve interpolations now
    OmegaConf.resolve(cfg)
    OmegaConf.save(config=cfg, f=str(resolved_cfg_path))

    # Prepare env
    cut_env_cfg = _coerce_env_cfg(cfg.cut_env)
    taichi_env = get_strawberry_cuthalf_env(cut_env_cfg)
    knife = taichi_env.agent.effectors[0]

    # wandb (optional)
    wandb_run = None
    if bool(cfg.log_wandb):
        wandb_kwargs = {}
        if hasattr(cfg, "logging") and cfg.logging is not None:
            wandb_kwargs = OmegaConf.to_container(cfg.logging, resolve=True)

        wandb_project = OmegaConf.select(cfg, "wandb.project", default=None)
        if wandb_project is None:
            wandb_project = OmegaConf.select(cfg, "wandb_project", default=None)
        if wandb_project is not None and "project" not in wandb_kwargs:
            wandb_kwargs["project"] = wandb_project

        wandb_run = wandb.init(
            config=OmegaConf.to_container(cfg, resolve=True),
            **wandb_kwargs,
        )

    horizon_action = int(cfg.horizon_action)
    render_gap = int(cfg.render_gap)

    save_videos = bool(OmegaConf.select(cfg, "video.save", default=True))
    video_fps = int(OmegaConf.select(cfg, "video.fps", default=10))
    video_dir = OmegaConf.select(cfg, "video.dir", default=str(run_dir / "videos"))

    # init actions from cfg
    init_action_p = np.array(cfg.init_action_p, dtype=np.float32)
    init_action_v = np.array([cfg.init_action_v] * horizon_action, dtype=np.float32)

    # same heuristic
    init_action_v[: int(horizon_action * 0.4), 1] = -0.3

    init_actions = np.concatenate([init_action_v, init_action_p[None]], axis=0)
    current_actions = init_actions.copy()

    # optimizer LR per component
    lr_action_p = np.ones_like(init_action_p) * float(cfg.optim.lr_action_p)
    lr_action_v = np.ones_like(init_action_v) * float(cfg.optim.lr_action_v)
    lr = np.concatenate([lr_action_v, lr_action_p[None]], axis=0)

    optim = OPTIMS[cfg.optim.type](init_actions.copy(), lr, cfg.optim)

    init_state = taichi_env.get_state()["state"]
    save_info = []

    best_loss = float("inf")
    best_actions = None

    for iteration in range(int(cfg.n_iters)):
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

        # best tracking
        loss_val = _loss_scalar(loss_obj)
        if loss_val < best_loss:
            best_loss = loss_val
            best_actions = current_actions.copy()

        # console log
        log_console(cfg, taichi_env, knife, iteration, loss_obj, forward_s, backward_s, current_actions)

        # local video
        render_images = fwd.get("render_images", [])
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
                "loss": loss_val,
                "time/forward_s": forward_s,
                "time/backward_s": backward_s,
            }
            if isinstance(loss_obj, dict):
                for k, v in loss_obj.items():
                    if isinstance(v, (int, float, np.floating)):
                        log_dict[f"loss/{k}"] = float(v)
            if video_path is not None:
                log_dict["video_path"] = video_path

            wandb_run.log(log_dict, step=iteration)

        # checkpoint info (pre-step, aligns with your original)
        iter_ctx = {
            "iter": iteration,
            "comp_actions": current_actions,
            "loss": loss_obj,
            "grad": grad,
            "render_images": render_images,
        }
        cur_save = {}
        for key in cfg.save_info_keys:
            cur_save[key] = copy.deepcopy(iter_ctx[key])
        save_info.append(cur_save)

        # step
        assert not np.isnan(np.mean(grad)), f"NaN grad at iter {iteration}"
        current_actions = optim.step(grad)

        # optional bounds clamp
        if "bounds" in cfg.optim and cfg.optim.bounds is not None:
            lo, hi = cfg.optim.bounds
            current_actions = np.clip(current_actions, float(lo), float(hi))

    # Save checkpoint + final/best actions
    with open(checkpoint_path, "wb") as f:
        pickle.dump(save_info, f)

    np.save(final_actions_path, current_actions)
    if best_actions is not None:
        np.save(best_actions_path, best_actions)

    # Save explicit mesh-action pair object
    pair_obj = {
        "mesh_id": mesh_id,
        "mesh_path": str(mesh_path),
        "run_dir": str(run_dir),
        "final_actions": current_actions,
        "best_actions": best_actions,
        "best_loss": best_loss,
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "checkpoint_path": str(checkpoint_path),
        "final_actions_path": str(final_actions_path),
        "best_actions_path": str(best_actions_path),
    }
    with open(pair_path, "wb") as f:
        pickle.dump(pair_obj, f)

    if wandb_run is not None:
        wandb_run.finish()

    print(f"[DONE] {mesh_id} saved to: {run_dir}")
    return {
        "mesh_id": mesh_id,
        "mesh_path": str(mesh_path),
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "final_actions_path": str(final_actions_path),
        "best_actions_path": str(best_actions_path),
        "pair_path": str(pair_path),
        "skipped": False,
    }


# -----------------------------
# Main: iterate dataset
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/louhz/Desktop/Rss/roboninja/generateddata/Generated_strawberry",
        help="Dataset root containing 0001..0300 folders.",
    )
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=300)
    parser.add_argument("--mesh_filename", type=str, default="mesh.obj")

    # Where to save per-mesh outputs:
    # output_dir = <data_root>/<id>/<out_subdir>/
    parser.add_argument("--out_subdir", type=str, default="expert_actions")

    # Hydra config compose
    parser.add_argument("--config_name", type=str, default="expert")
    parser.add_argument(
        "--config_dir",
        type=str,
        default=None,
        help="Absolute path to roboninja/config. If omitted, infer from repo root.",
    )

    parser.add_argument(
        "--n_iters",
        type=int,
        default=None,
        help="Override cfg.n_iters (optimization steps per mesh).",
    )

    # behavior
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")

    # override wandb/video quickly
    parser.add_argument("--log_wandb", action="store_true", default=False)
    parser.add_argument("--save_videos", action="store_true", default=True)
    parser.add_argument("--no_videos", dest="save_videos", action="store_false")

    args = parser.parse_args()

    # Repo root bootstrap
    repo_root = find_repo_root(Path(__file__).resolve())
    sys.path.append(str(repo_root))
    os.chdir(str(repo_root))

    # Infer config dir if not given
    config_dir = Path(args.config_dir) if args.config_dir is not None else (repo_root / "roboninja" / "config")
    config_dir = config_dir.resolve()
    assert config_dir.is_dir(), f"Config dir not found: {config_dir}"

    # Compose base config once
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        base_cfg = compose(config_name=args.config_name)

    if args.n_iters is not None:
        base_cfg.n_iters = int(args.n_iters)    

    # apply global overrides from CLI
    base_cfg.log_wandb = bool(args.log_wandb)
    if "video" in base_cfg and base_cfg.video is not None:
        base_cfg.video.save = bool(args.save_videos)

    data_root = Path(args.data_root)
    assert data_root.is_dir(), f"data_root not found: {data_root}"

    # Write a dataset-level index for convenience
    index_rows = []
    index_path = data_root / f"{args.config_name}_mesh_action_index.csv"

    for i in range(args.start, args.end + 1):
        mesh_id = f"{i:04d}"
        mesh_dir = data_root / mesh_id
        if not mesh_dir.is_dir():
            print(f"[WARN] Missing dir: {mesh_dir} (skipping)")
            continue

        mesh_path = mesh_dir / args.mesh_filename
        if not mesh_path.exists():
            # fallback: first .obj file
            objs = sorted(mesh_dir.glob("*.obj"))
            if len(objs) == 0:
                print(f"[WARN] No mesh found in {mesh_dir} (skipping)")
                continue
            mesh_path = objs[0]

        out_dir = mesh_dir / args.out_subdir

        # Run
        try:
            res = run_one_mesh(
                cfg=base_cfg,
                run_dir=out_dir,
                mesh_id=mesh_id,
                mesh_path=mesh_path,
                skip_existing=args.skip_existing,
            )
            index_rows.append(res)
        except Exception as e:
            print(f"[ERROR] mesh {mesh_id} failed: {repr(e)}")
            index_rows.append({
                "mesh_id": mesh_id,
                "mesh_path": str(mesh_path),
                "run_dir": str(out_dir),
                "error": repr(e),
            })

    # Save dataset-level index csv
    # (simple csv to avoid pandas dependency)
    if len(index_rows) > 0:
        keys = sorted({k for row in index_rows for k in row.keys()})
        with open(index_path, "w") as f:
            f.write(",".join(keys) + "\n")
            for row in index_rows:
                vals = []
                for k in keys:
                    v = row.get(k, "")
                    v = str(v).replace("\n", " ").replace(",", ";")
                    vals.append(v)
                f.write(",".join(vals) + "\n")
        print(f"[INDEX] Wrote: {index_path}")


if __name__ == "__main__":
    main()
