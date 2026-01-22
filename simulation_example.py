#!/usr/bin/env python3
"""
Converted from simulation_example.ipynb to a standalone Python script.

Notes:
- This script assumes you run it from the project root (so relative paths like
  "roboninja/config/cut_env" and "data/..." resolve correctly).
- The hard-coded mesh path in the original notebook is preserved; change it to match
  your machine if needed.
"""

from __future__ import annotations

# %% Imports
import os
from pathlib import Path
import pickle

# If you are running headless (no display), matplotlib may need a non-interactive backend.
# This is a safe default on servers; comment out if you prefer an interactive window.
if os.environ.get("DISPLAY", "") == "":
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from cut_simulation.utils.misc import *  # noqa: F403,F401 (kept as-is from the notebook)
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from roboninja.env.tc_env import get_cut_env, get_strawberry_env
from roboninja.utils.misc import animate


# %% Functions (from the notebook)
def forward(taichi_env, init_state, comp_actions, grad_enabled, render, **kwargs):
    taichi_env.set_state(init_state, grad_enabled=grad_enabled)

    actions_v = comp_actions[:-1]
    action_p = comp_actions[-1]
    horizon_action = len(actions_v)
    taichi_env.apply_agent_action_p(action_p)
    render_images = list()

    if render:
        render_images.append(taichi_env.render())
    for i in range(horizon_action):
        taichi_env.step(actions_v[i])
        if render:
            render_images.append(taichi_env.render())
    loss = taichi_env.get_loss()

    forward_output = {
        'loss': loss,
        'render_images': render_images,
    }

    return forward_output


def backward(taichi_env, comp_actions, **kwargs):
    taichi_env.reset_grad()
    taichi_env.get_loss_grad()

    actions_v = comp_actions[:-1]
    horizon_action = len(actions_v)

    for i in range(horizon_action - 1, -1, -1):
        taichi_env.step_grad(actions_v[i])
    grad = taichi_env.agent.get_grad(len(actions_v))
    grad[-1][1:] = 0

    backward_output = {
        'grad': grad,
    }

    return backward_output


# %% [markdown]
# read the strawberry
#
#
# fix the scale and relative pose for the fruit, scene, obtain the relative final pose for the knife and the strawberry, also compute the knife cutting trajectory with relative to the strawberry 
#
#
# replace the knife and strawberry and cutting traejectory in the roboninja simulation.
#
# Then simulate the moving of the particel
#
#
#
#
# Then design the custom reward for different strawberry size
#
#
#
#  compute the relationship between the knife and the end effector 
#
#  compute ik for different relative pose of the knife and the strawberry and shape of strawberry
#
#
#  batchify generate data for control
#
#
#  test it in genesis for collision avoidance
#
#
# Then add the tactile simulation on the side of this two strawberry and near knife(optional)
#
#
#  deploy to real world (test the validness of our data, for the rl   )

# %% [markdown]
# training logic:  query state(no privilage information)
#
#
# (based on the verified data)
#
# multi-stage
#
# ee+hand seperate
#
# adaptive chunking size
#
#
# we have data for the strayberry and hand 
#
#
# we need more data for the end effector pose varies
#







# fix the digital asset  

# fix the rl policy : combine with our datasetrender.py, make sure the knife is able to cut through the strawberry and it can be performed by the collision check

# fix  tactile















# %% Main logic (from notebook code cells)
def main() -> None:
    # Hydra initialization (as in the notebook)
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    # NOTE: this path is relative to the current working directory.
    config_dir = os.path.abspath("roboninja/config/cut_env")
    initialize_config_dir(config_dir=config_dir, version_base=None)

    cfg = compose(config_name="cut_env_strawberry")
    # cfg = compose(config_name='cut_env')
    bone_idx = 0

    # The following mesh path was hard-coded in the original notebook.
    # Change it to a valid path on your machine if needed.
    cfg.strawberry.mesh_path = "/home/louhz/Desktop/Rss/roboninja/generateddata/Generated_strawberry/0001/mesh.obj"
    cfg.strawberry.pos = (0.5, 0.16, 0.5)
    cfg.strawberry.scale = (0.4, 0.4, 0.4)

    cfg.strawberry.voxelize_res = 256
    cfg.strawberry.shell_radius_vox = 2
    cfg.strawberry.close_iters = 2
    cfg.strawberry.normalize = True
    cfg.strawberry.trim_percentile = 0.5
    cfg.strawberry.cache_voxels = True
    cfg.strawberry.euler = (0.0, 180.0, 0.0)
    cfg.auto_boundary = True

    cfg.bone.name = f'bone_{bone_idx}'
    taichi_env = get_strawberry_env(cfg)

    knife = taichi_env.agent.effectors[0]

    # set some constant
    horizon_action = 60

    # set init actions
    init_action_p = np.asarray([-0.8, 0.215, 0.5])
    init_action_v = np.asarray([[0.0, 0.0, 0.0]] * horizon_action)
    init_action_v[: int(horizon_action * 0.4), 1] = -0.3
    init_actions = np.concatenate([init_action_v, init_action_p[None]], axis=0)

    init_state = taichi_env.get_state()['state']
    current_actions = init_actions

    kwargs = {
        'taichi_env': taichi_env,
        'init_state': init_state,
        'comp_actions': current_actions,
        'grad_enabled': True,
        'render': True,
    }

    # ---
    # render image
    # ---

    # Render a single frame (notebook used plt.imshow(...) inline)
    render_img = taichi_env.render()
    os.makedirs('outputs', exist_ok=True)
    plt.figure()
    plt.imshow(render_img)
    plt.axis('off')
    plt.tight_layout()
    out_png = os.path.join('outputs', 'render.png')
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved render image to: {out_png}')

    # ---
    # forward pass using the initial trajectory
    # ---

    os.makedirs('videos', exist_ok=True)
    forward_output = forward(**kwargs)
    print(forward_output['loss'])
    animate(forward_output['render_images'], 'videos/initial_trajectory.mp4')

    # ---
    # compute action gradient w.r.t. loss
    # ---

    backward_output = backward(**kwargs)
    gradient = backward_output['grad']
    # `gradient` is kept for parity with the notebook; you can inspect/save it as needed.

    # ---
    # directly load the optimized trajectory
    # ---

    opt_path = f'data/expert/expert_{bone_idx}/optimization.pkl'
    if os.path.exists(opt_path):
        data = pickle.load(open(opt_path, 'rb'))
        kwargs['comp_actions'] = data[-1]['comp_actions']
        forward_output = forward(**kwargs)
        animate(forward_output['render_images'], 'videos/optimized_trajectory.mp4')
    else:
        print(f'Optimization file not found, skipping: {opt_path}')


if __name__ == '__main__':
    main()
