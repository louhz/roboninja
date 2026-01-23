# the combination of the dataset render and  simulation

# may be need to call two conda environment 


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RL-style trajectory search (CEM) in Genesis:
Goal:
  - Move a knife end-effector straight down to the ground (z=GROUND_Z)
  - Avoid self-collision (robot vs robot)
  - Export a "worked trajectory" compatible with your playback script

Outputs (ROS-like TXT, same keys your parser expects):
  - nova2.txt: 6 arm joints (DEGREES)
  - nova5.txt: 6 arm joints (DEGREES)
  - left.txt : 10 hand signals (0..255)  [kept constant by default]
  - right.txt: 10 hand signals (0..255)  [kept constant by default]

How it works:
  - We define a desired straight-down Cartesian path for the knife tip:
      p(t) = p_start + alpha(t) * (p_goal - p_start)
    where p_goal is same x,y and z=GROUND_Z.
  - We allow a small learned XY offset curve (knots -> linear interpolation).
    RL objective pushes offsets to ~0 (direct cut), but allows nonzero if needed
    to avoid self-collision.
  - For each candidate offset curve, we:
      (1) Solve IK at each waypoint (only knife arm DoFs)
      (2) Check self-collision at each waypoint (fast collision query)
      (3) Score cost and update CEM distribution
  - Finally, export the best trajectory to TXT.

NOTE:
  You MUST set:
    - MJCF_PATH
    - KNIFE_EE_LINK_NAME (your knife tip / tool link in scene.xml)
    - KNIFE_ARM_KEY ("n5" or "n2")

Optional:
  - You can enable a final dynamic playback validation with scene.step().
"""

import os
import math
import numpy as np
from pathlib import Path

import genesis as gs


import re
import numpy as np
from pathlib import Path

import re

def read_dt_from_txt_header(txt_path: str | Path) -> float | None:
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            for _ in range(30):
                line = f.readline()
                if not line:
                    break
                if not line.lstrip().startswith("#"):
                    break
                m = re.search(r"\bdt\s*=\s*([0-9.eE+\-]+)", line)
                if m:
                    return float(m.group(1))
    except Exception:
        pass
    return None

def load_comp_actions_txt(txt_path: str | Path, dt_fallback: float) -> tuple[np.ndarray, float]:
    txt_path = Path(txt_path)
    arr = np.loadtxt(txt_path, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] != 3:
        raise ValueError(f"{txt_path} must have 3 cols, got {arr.shape}")
    dt = read_dt_from_txt_header(txt_path)
    if dt is None:
        dt = float(dt_fallback)
    return arr, dt

def save_comp_actions_txt(txt_path: str | Path, comp_actions: np.ndarray, dt: float) -> None:
    txt_path = Path(txt_path)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    comp_actions = np.asarray(comp_actions, dtype=np.float64)
    header = "comp_actions: rows 0..H-1 = action_v, last row = action_p\n" + f"dt = {float(dt):.10g}"
    np.savetxt(txt_path, comp_actions, fmt="%.10f", header=header)
    print(f"[TXT] Saved comp_actions: {txt_path.resolve()} shape={comp_actions.shape} dt={dt}")

def comp_actions_to_positions(comp_actions: np.ndarray, dt: float) -> np.ndarray:
    """Integrate v actions into a waypoint position sequence P of shape (H+1,3)."""
    comp_actions = np.asarray(comp_actions, dtype=np.float64)
    v = comp_actions[:-1]
    p0 = comp_actions[-1]
    H = v.shape[0]
    P = np.zeros((H + 1, 3), dtype=np.float64)
    P[0] = p0
    for i in range(H):
        P[i + 1] = P[i] + v[i] * dt
    return P

def positions_to_comp_actions(P: np.ndarray, dt: float) -> np.ndarray:
    """Convert waypoints P (H+1,3) into comp_actions (H+1,3): v rows then last row p0."""
    P = np.asarray(P, dtype=np.float64)
    v = np.diff(P, axis=0) / dt
    p0 = P[0]
    return np.vstack([v, p0[None]])

def resample_positions(P: np.ndarray, new_len: int) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    if P.shape[0] == new_len:
        return P.copy()
    t_old = np.linspace(0.0, 1.0, P.shape[0])
    t_new = np.linspace(0.0, 1.0, new_len)
    out = np.zeros((new_len, 3), dtype=np.float64)
    for d in range(3):
        out[:, d] = np.interp(t_new, t_old, P[:, d])
    return out

# ---------------------------
# Taichi (cut_simulation) trajectory I/O
# ---------------------------
# This is the TXT saved from your Taichi script initial policy or optimized policy:
BASELINE_ACTIONS_TXT = "./outputs/knife_actions_init.txt"  # or "./outputs/knife_actions_optimized.txt"
USE_BASELINE_ACTIONS_IF_EXISTS = True

# If the TXT header doesn't contain dt, we use this
BASELINE_DT_FALLBACK = 0.01

# Safer: treat Taichi baseline as a RELATIVE motion, anchored at the robot's current knife pose in Genesis.
# This avoids needing perfect world-frame alignment between the two simulators.
USE_BASELINE_RELATIVE_TO_CURRENT_EE = True

# Export a collision-safe version back to Taichi sim:
EXPORT_TAICHI_ACTIONS = True


# Taichi expects horizon_action=60 in your script, so default to that if no baseline file is loaded.
EXPORT_TAICHI_HORIZON = 60


# ---------------------------
# User config
# ---------------------------

MJCF_PATH = "./robot_urdf_genesis/scene.xml"
OBJECT_MJCF_PATH = "./fruit_asset/fruit.xml"   # optional, can be missing

OUT_DIR = Path("./rl_out_cut_to_ground")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TAICHI_ACTIONS_OUT_TXT = OUT_DIR / "knife_actions_out.txt"
TAICHI_WAYPOINTS_OUT_TXT = OUT_DIR / "knife_waypoints_out.txt"


DT_CMD = 0.01
N_STEPS = 200  # 200 * 0.01 = 2 seconds

GROUND_Z = 0.0
TARGET_CLEARANCE = 0.0  # set to 0.005 if you want to stop slightly above ground

# Choose which arm is holding the knife: "n5" or "n2"
KNIFE_ARM_KEY = "n5"

# IMPORTANT: set this to your knife/tool end-effector link name inside scene.xml
# Examples (you must check your MJCF): "knife_tip", "tool0", "ee_link", ...
KNIFE_EE_LINK_NAME = "knife_tip"

# Desired orientation: we want tool Z-axis pointing downward.
# Genesis examples commonly use quat=[0,1,0,0] for "point down".
# We'll only constrain Z-axis (rot_mask=[False,False,True]) so yaw is free.
TARGET_QUAT_WXYZ = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
ROT_MASK = [False, False, True]  # only constrain Z-axis direction

# ---------------------------
# CEM / RL policy-search params
# ---------------------------

N_KNOTS = 5              # number of XY offset knots across the trajectory
POP_SIZE = 64            # candidates per iteration
ELITE_FRAC = 0.2         # top % used to refit distribution
N_ITERS = 30

INIT_STD_METERS = 0.03   # initial std for XY offsets (meters)
MAX_ABS_OFFSET = 0.08    # hard clamp on offsets (meters)

# Cost weights
W_OFFSET = 1.0           # penalize deviating from "direct" straight-down
W_SMOOTH = 0.05          # penalize joint-space jerk / large changes
W_IK_ERR = 500.0         # penalize IK residual error
W_SELF_COLLISION = 1e6   # huge penalty

# IK solver settings (can tune)
IK_MAX_SAMPLES = 30
IK_MAX_ITERS = 30
IK_DAMPING = 0.01
IK_POS_TOL = 5e-4
IK_ROT_TOL = 5e-3
IK_MAX_STEP = 0.5

# If you want a final playback validation in Genesis dynamics:
DO_DYNAMIC_VALIDATION = True
VALIDATION_SUBSTEPS_PER_CMD = 1  # dt_sim == dt_cmd in your playback
FAIL_ON_EXTERNAL_COLLISION = False  # set True if you also want robot-plane collisions forbidden

# ---------------------------
# Group parameters (keep same ordering as your playback script)
# ---------------------------

NOVA2_PARAMS = [
    {"joint": "nova2joint1", "kp": 300.0,  "ctrlrange": (-6.28,  6.28)},
    {"joint": "nova2joint2", "kp": 300.0,  "ctrlrange": (-3.14,  3.14)},
    {"joint": "nova2joint3", "kp": 300.0,  "ctrlrange": (-2.79,  2.79)},
    {"joint": "nova2joint4", "kp": 250.0,  "ctrlrange": (-6.28,  6.28)},
    {"joint": "nova2joint5", "kp": 200.0,  "ctrlrange": (-6.28,  6.28)},
    {"joint": "nova2joint6", "kp": 1500.0, "ctrlrange": (-6.28,  6.28)},
]

NOVA5_PARAMS = [
    {"joint": "nova5joint1", "kp": 300.0,  "ctrlrange": (-6.28,  6.28)},
    {"joint": "nova5joint2", "kp": 3000.0, "ctrlrange": (-3.14,  3.14)},
    {"joint": "nova5joint3", "kp": 3000.0, "ctrlrange": (-2.79,  2.79)},
    {"joint": "nova5joint4", "kp": 250.0,  "ctrlrange": (-6.28,  6.28)},
    {"joint": "nova5joint5", "kp": 200.0,  "ctrlrange": (-6.28,  6.28)},
    {"joint": "nova5joint6", "kp": 150.0,  "ctrlrange": (-6.28,  6.28)},
]

# Hands: your playback expects 10-channel 0-255 signals.
# We keep hands open (255) by default for safety.
DEFAULT_HAND_SIGNAL_10 = np.full((10,), 255, dtype=np.int32)

# ---------------------------
# Helpers
# ---------------------------

def params_to_arrays(group):
    names = [p["joint"] for p in group]
    kp    = np.array([p["kp"] for p in group], dtype=np.float32)
    lo    = np.array([p["ctrlrange"][0] for p in group], dtype=np.float64)
    hi    = np.array([p["ctrlrange"][1] for p in group], dtype=np.float64)
    return names, kp, lo, hi

def resolve_dofs(entity, joint_names):
    """Return local DoF indices for available names, and which names resolved/missing."""
    idx, ok, missing = [], [], []
    for n in joint_names:
        try:
            j = entity.get_joint(n)
            idx.append(j.dof_idx_local)
            ok.append(n)
        except Exception:
            missing.append(n)
    return np.array(idx, dtype=int), ok, missing

def set_group_gains(entity, dof_idx, kp, kv_scale=2.0, kp_scale=1.0):
    if dof_idx.size == 0:
        return
    kp_eff = kp_scale * kp
    kv_eff = kv_scale * np.sqrt(np.maximum(kp_eff, 1e-6))
    entity.set_dofs_kp(kp_eff.astype(np.float32), dofs_idx_local=dof_idx)
    entity.set_dofs_kv(kv_eff.astype(np.float32), dofs_idx_local=dof_idx)

def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

def to_numpy(x):
    # Genesis APIs may return torch tensors depending on backend; handle both.
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

def write_ros_txt(out_path: Path, dt: float, q: np.ndarray):
    """
    q: (N, D) array.
    Writes ROS-like blocks:
      seq:
      secs:
      nsecs:
      position: [ ... ]
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(q.shape[0]):
            t = i * dt
            secs = int(math.floor(t))
            nsecs = int(round((t - secs) * 1e9))
            f.write(f"seq: {i}\n")
            f.write(f"secs: {secs}\n")
            f.write(f"nsecs: {nsecs}\n")
            # ensure JSON-ish list
            arr = q[i].tolist()
            f.write(f"position: {arr}\n\n")

def is_self_collision_pairs(robot, pairs: np.ndarray) -> bool:
    """
    detect_collision() returns geom index pairs where at least one geom is in the entity.
    We define self-collision as BOTH geoms in [robot.geom_start, robot.geom_end).
    """
    if pairs is None or len(pairs) == 0:
        return False
    a = pairs[:, 0]
    b = pairs[:, 1]
    in_a = (a >= robot.geom_start) & (a < robot.geom_end)
    in_b = (b >= robot.geom_start) & (b < robot.geom_end)
    return bool(np.any(in_a & in_b))

def has_any_entity_collision_pairs(robot, pairs: np.ndarray) -> bool:
    """Any collision involving the robot (self or external)."""
    return pairs is not None and len(pairs) > 0

def xy_offset_curve(theta_xy: np.ndarray, alphas: np.ndarray, n_knots: int) -> np.ndarray:
    """
    theta_xy: (2*n_knots,) -> knots in meters
    alphas: (N,) in [0,1]
    returns offsets: (N,2)
    """
    theta_xy = theta_xy.reshape(n_knots, 2)
    knots = np.linspace(0.0, 1.0, n_knots)
    ox = np.interp(alphas, knots, theta_xy[:, 0])
    oy = np.interp(alphas, knots, theta_xy[:, 1])
    return np.stack([ox, oy], axis=1)

def build_straight_down_path(p_start: np.ndarray, p_goal: np.ndarray, theta_xy: np.ndarray) -> np.ndarray:
    """
    p_start, p_goal: (3,)
    theta_xy: (2*n_knots,)
    returns P: (N_STEPS,3)
    """
    alphas = np.linspace(0.0, 1.0, N_STEPS)
    P = p_start[None, :] + alphas[:, None] * (p_goal - p_start)[None, :]
    offs = xy_offset_curve(theta_xy, alphas, N_KNOTS)  # (N,2)
    P[:, 0] += offs[:, 0]
    P[:, 1] += offs[:, 1]
    return P

def eval_candidate(robot, knife_link, knife_link_local_idx: int,
                   q_init: np.ndarray,
                   dofs_idx_arm: np.ndarray,
                   p_start: np.ndarray,
                   p_goal: np.ndarray,
                   theta_xy: np.ndarray,
                   arm_limits_lo: np.ndarray,
                   arm_limits_hi: np.ndarray):
    """
    Returns:
      cost (float), q_traj (N_STEPS, n_dofs) or None if failed
    """
    # Clamp offsets so CEM can't go crazy.
    theta_xy = np.clip(theta_xy, -MAX_ABS_OFFSET, MAX_ABS_OFFSET)

    # Build path in Cartesian
    # Apply learned XY offsets on top of the baseline path
    alphas = np.linspace(0.0, 1.0, N_STEPS)
    offs = xy_offset_curve(theta_xy, alphas, N_KNOTS)  # (N,2)
    P = P_base.copy()
    P[:, 0] += offs[:, 0]
    P[:, 1] += offs[:, 1]

    # Penalty for not being "direct"
    alphas = np.linspace(0.0, 1.0, N_STEPS)
    offs = xy_offset_curve(theta_xy, alphas, N_KNOTS)
    cost_offset = W_OFFSET * float(np.mean(np.sum(offs**2, axis=1)))

    # Solve IK waypoint by waypoint
    q_prev = q_init.copy()
    q_traj = np.zeros((N_STEPS, q_init.shape[0]), dtype=np.float64)

    ik_err_accum = 0.0
    smooth_accum = 0.0

    # We will temporarily set robot pose during collision checks.
    # Save current pose to restore after evaluation.
    q_restore = to_numpy(robot.get_dofs_position())

    try:
        for i in range(N_STEPS):
            pos_tgt = P[i].astype(np.float64)

            # IK (restrict to knife arm dofs_idx_local)
            q_sol, err = robot.inverse_kinematics(
                link=knife_link,
                pos=pos_tgt,
                quat=TARGET_QUAT_WXYZ,
                init_qpos=q_prev,
                respect_joint_limit=True,
                max_samples=IK_MAX_SAMPLES,
                max_solver_iters=IK_MAX_ITERS,
                damping=IK_DAMPING,
                pos_tol=IK_POS_TOL,
                rot_tol=IK_ROT_TOL,
                pos_mask=[True, True, True],
                rot_mask=ROT_MASK,
                max_step_size=IK_MAX_STEP,
                dofs_idx_local=dofs_idx_arm,
                return_error=True,
            )
            q_sol = to_numpy(q_sol).astype(np.float64, copy=False)
            err = to_numpy(err).astype(np.float64, copy=False)

            # IK residual cost
            # err is [err_pos_x, err_pos_y, err_pos_z, err_rot_x, err_rot_y, err_rot_z]
            epos = float(np.linalg.norm(err[:3]))
            erot = float(np.linalg.norm(err[3:]))
            ik_err_accum += (epos**2 + erot**2)

            # Clamp arm joints to their limits (safety)
            q_arm = q_sol[dofs_idx_arm]
            q_arm = clamp(q_arm, arm_limits_lo, arm_limits_hi)
            q_sol[dofs_idx_arm] = q_arm

            # Smoothness penalty (joint deltas)
            if i > 0:
                dq = q_sol[dofs_idx_arm] - q_prev[dofs_idx_arm]
                smooth_accum += float(np.mean(dq * dq))

            # Collision check at this waypoint (fast, no scene.step required)
            robot.set_dofs_position(q_sol, zero_velocity=True)
            pairs = robot.detect_collision(env_idx=0)
            pairs = np.asarray(pairs) if pairs is not None else np.zeros((0, 2), dtype=np.int32)

            if is_self_collision_pairs(robot, pairs):
                # Nearest-working fallback:
                # hold the last safe configuration for the rest of the trajectory
                q_traj[i:] = q_prev  # q_prev is the last safe q
                progress = float(i) / float(max(1, N_STEPS - 1))
                collision_pen = W_SELF_COLLISION * (1.0 - progress)

                # compute partial averages to keep numbers sane
                denom = max(1, i)
                cost_ik = W_IK_ERR * float(ik_err_accum / denom)
                cost_smooth = W_SMOOTH * float(smooth_accum / max(1, (i - 1)))
                total_cost = cost_offset + cost_ik + cost_smooth + collision_pen
                return total_cost, q_traj

            if FAIL_ON_EXTERNAL_COLLISION and has_any_entity_collision_pairs(robot, pairs):
                # If you also want to forbid robot-ground contacts etc.
                return W_SELF_COLLISION + cost_offset, q_traj

            q_traj[i] = q_sol
            q_prev = q_sol

        cost_ik = W_IK_ERR * float(ik_err_accum / N_STEPS)
        cost_smooth = W_SMOOTH * float(smooth_accum / max(1, (N_STEPS - 1)))
        total_cost = cost_offset + cost_ik + cost_smooth
        return total_cost, q_traj

    finally:
        # restore robot pose
        robot.set_dofs_position(q_restore, zero_velocity=True)

def cem_optimize(robot, knife_link, knife_link_local_idx: int,
                 q_init: np.ndarray,
                 dofs_idx_arm: np.ndarray,
                 p_start: np.ndarray,
                 p_goal: np.ndarray,
                 arm_limits_lo: np.ndarray,
                 arm_limits_hi: np.ndarray,
                 seed: int = 0):
    rng = np.random.default_rng(seed)
    dim = 2 * N_KNOTS  # XY at each knot

    mean = np.zeros((dim,), dtype=np.float64)
    std  = np.full((dim,), INIT_STD_METERS, dtype=np.float64)

    best_cost = float("inf")
    best_traj = None
    best_theta = None

    n_elite = max(1, int(round(POP_SIZE * ELITE_FRAC)))

    for it in range(N_ITERS):
        thetas = mean[None, :] + std[None, :] * rng.standard_normal(size=(POP_SIZE, dim))
        thetas = np.clip(thetas, -MAX_ABS_OFFSET, MAX_ABS_OFFSET)

        costs = np.zeros((POP_SIZE,), dtype=np.float64)
        trajs = [None] * POP_SIZE

        for k in range(POP_SIZE):
            c, traj = eval_candidate(
                robot=robot,
                knife_link=knife_link,
                knife_link_local_idx=knife_link_local_idx,
                q_init=q_init,
                dofs_idx_arm=dofs_idx_arm,
                p_start=p_start,
                p_goal=p_goal,
                theta_xy=thetas[k],
                arm_limits_lo=arm_limits_lo,
                arm_limits_hi=arm_limits_hi,
            )
            costs[k] = c
            trajs[k] = traj

        order = np.argsort(costs)
        elite_idx = order[:n_elite]
        elite_thetas = thetas[elite_idx]

        # update distribution
        mean = elite_thetas.mean(axis=0)
        std  = elite_thetas.std(axis=0) + 1e-6


                # track best
        if costs[order[0]] < best_cost:
            best_cost = float(costs[order[0]])
            best_traj = trajs[order[0]]
            best_theta = thetas[order[0]].copy()

            # ---- Export a collision-safe Taichi comp_actions trajectory (nearest-working if needed) ----
        if EXPORT_TAICHI_ACTIONS:
            # Apply offsets to the export waypoint sequence P_export (length export_horizon+1)
            alphas_out = np.linspace(0.0, 1.0, P_export.shape[0])
            offs_out = xy_offset_curve(best_theta, alphas_out, N_KNOTS)  # (H+1,2)

            P_out = P_export.copy()
            P_out[:, 0] += offs_out[:, 0]
            P_out[:, 1] += offs_out[:, 1]

            # If the robot plan got frozen (nearest-working), freeze the Cartesian path too
            # (detect a “hold from some index onward” in best_traj)
            def find_freeze_index(q_traj: np.ndarray, tol: float = 1e-10) -> int | None:
                for i in range(1, q_traj.shape[0]):
                    if np.all(np.abs(q_traj[i:] - q_traj[i]) < tol):
                        return i
                return None

            freeze_idx = find_freeze_index(best_traj)
            if freeze_idx is not None:
                progress = float(freeze_idx) / float(max(1, best_traj.shape[0] - 1))
                out_freeze = int(round(progress * (P_out.shape[0] - 1)))
                P_out[out_freeze:] = P_out[out_freeze]
                print(f"[Traj] Nearest-working freeze detected: Genesis step={freeze_idx}, export step={out_freeze}")

            comp_actions_out = positions_to_comp_actions(P_out, export_dt)

            save_comp_actions_txt(TAICHI_ACTIONS_OUT_TXT, comp_actions_out, dt=export_dt)
            np.savetxt(TAICHI_WAYPOINTS_OUT_TXT, P_out, fmt="%.10f")
            print(f"[Export] Saved waypoints: {TAICHI_WAYPOINTS_OUT_TXT.resolve()}")

        print(f"[CEM] iter={it:02d} best_in_iter={costs[order[0]]: .6e}  "
              f"best_so_far={best_cost: .6e}  std_mean={float(std.mean()): .4f}")

    return best_theta, best_traj, best_cost

def dynamic_validate(scene, robot, q_traj: np.ndarray):
    """
    Optional: run PD control along trajectory with scene.step() and check self-contacts
    using get_contacts(with_entity=robot) after stepping.
    """
    print("[Validate] Dynamic playback validation...")

    # Move to first q
    robot.zero_all_dofs_velocity()
    robot.set_dofs_position(q_traj[0], zero_velocity=True)
    for _ in range(20):
        scene.step()

    # Execute
    for i in range(q_traj.shape[0]):
        robot.control_dofs_position(q_traj[i])
        for _ in range(VALIDATION_SUBSTEPS_PER_CMD):
            scene.step()

        # self-collision contacts after step
        contacts = robot.get_contacts(with_entity=robot)
        # contacts entries are empty if no contacts
        pos = contacts.get("position", None)
        if pos is not None:
            pos_np = to_numpy(pos)
            if pos_np.shape[0] > 0:
                print(f"[Validate] FAIL self-collision at step {i}, n_contacts={pos_np.shape[0]}")
                return False

    print("[Validate] OK (no self contacts in stepped sim).")
    return True

# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    # ---- Genesis init ----
    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT_CMD),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -3.5, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
            max_FPS=120,
        ),
        show_viewer=False,  # training headless
    )

    scene.add_entity(gs.morphs.Plane())

    # Ensure IK/Jacobian are enabled on robot morph if your defaults disable it.
    mjcf_morph = gs.morphs.MJCF(file=MJCF_PATH)
    # Some Genesis versions require enabling this explicitly:
    try:
        mjcf_morph.requires_jac_and_IK = True
    except Exception:
        pass

    robot = scene.add_entity(mjcf_morph)

    # optional object (fruit etc.)
    if Path(OBJECT_MJCF_PATH).exists():
        scene.add_entity(gs.morphs.MJCF(file=OBJECT_MJCF_PATH))
        print(f"[Load] Added object from {OBJECT_MJCF_PATH}")
    else:
        print(f"[Info] Object file not found (ok): {OBJECT_MJCF_PATH}")

    scene.build()

    # ---- Resolve arm DoFs ----
    groups = {"n2": NOVA2_PARAMS, "n5": NOVA5_PARAMS}

    indices = {}
    bounds = {}
    gains = {}

    for key, params in groups.items():
        names, kp, lo, hi = params_to_arrays(params)
        idx, ok, missing = resolve_dofs(robot, names)
        if missing:
            print(f"[Warn] Missing joints [{key}]: {missing}")
        if idx.size == 0:
            raise RuntimeError(f"No resolved DoFs for group {key}. Check joint names.")
        indices[key] = idx
        bounds[key] = (lo[:len(ok)], hi[:len(ok)])
        gains[key] = kp[:len(ok)].copy()

    # set PD gains similar to your playback (arms only here)
    for key in indices.keys():
        set_group_gains(robot, indices[key], gains[key], kv_scale=2.0, kp_scale=1.0)

    # ---- Select knife arm ----
    if KNIFE_ARM_KEY not in indices:
        raise RuntimeError(f"KNIFE_ARM_KEY={KNIFE_ARM_KEY} not in resolved groups {list(indices.keys())}")

    dofs_idx_arm = indices[KNIFE_ARM_KEY]
    arm_lo, arm_hi = bounds[KNIFE_ARM_KEY]

    # ---- Knife EE link ----
    # Helpful debug: print some link names if your link name is wrong.
    link_names = [lk.name for lk in robot.links]
    if KNIFE_EE_LINK_NAME not in link_names:
        print("[Error] KNIFE_EE_LINK_NAME not found in robot.links.")
        print("Available link names (first 60):")
        for n in link_names[:60]:
            print("  -", n)
        raise RuntimeError(f"Set KNIFE_EE_LINK_NAME correctly. Current: {KNIFE_EE_LINK_NAME}")

    knife_link = robot.get_link(KNIFE_EE_LINK_NAME)
    knife_link_local_idx = int(knife_link.idx - robot.link_start)

    # ---- Initial state ----
    q0 = to_numpy(robot.get_dofs_position()).astype(np.float64, copy=False)

    # Compute start EE pos via forward kinematics (no stepping needed)
    # forward_kinematics returns (links_pos, links_quat)
    links_pos, links_quat = robot.forward_kinematics(
        qpos=q0,
        links_idx_local=np.array([knife_link_local_idx], dtype=np.int32),
    )
    links_pos = to_numpy(links_pos)
    p_start = links_pos[0].copy()

    p_goal = p_start.copy()
    p_goal[2] = GROUND_Z + TARGET_CLEARANCE

    print(f"[Task] Knife start pos: {p_start}")
    print(f"[Task] Knife goal  pos: {p_goal} (GROUND_Z={GROUND_Z})")

        # ---- Build baseline Cartesian waypoints (either from Taichi TXT or straight-down) ----
    baseline_comp_actions = None
    baseline_dt = None

    if USE_BASELINE_ACTIONS_IF_EXISTS and Path(BASELINE_ACTIONS_TXT).exists():
        baseline_comp_actions, baseline_dt = load_comp_actions_txt(BASELINE_ACTIONS_TXT, dt_fallback=BASELINE_DT_FALLBACK)
        P_file = comp_actions_to_positions(baseline_comp_actions, baseline_dt)  # (H+1, 3)

        # anchor relative to current Genesis knife pose if requested
        if USE_BASELINE_RELATIVE_TO_CURRENT_EE:
            deltas = P_file - P_file[0]
            P_export = (p_start[None, :] + deltas)          # same length as file, but anchored at Genesis start
        else:
            P_export = P_file

        # For IK/collision evaluation we use N_STEPS waypoints
        P_base = resample_positions(P_export, N_STEPS)

        export_horizon = P_export.shape[0] - 1
        export_dt = baseline_dt

        print(f"[Traj] Loaded baseline comp_actions from {BASELINE_ACTIONS_TXT} "
              f"(horizon={export_horizon}, dt={export_dt})")

    else:
        # fallback: straight down in Genesis
        P_base = build_straight_down_path(p_start, p_goal, theta_xy=np.zeros((2 * N_KNOTS,), dtype=np.float64))
        export_horizon = EXPORT_TAICHI_HORIZON
        export_dt = DT_CMD
        P_export = resample_positions(P_base, export_horizon + 1)

        print("[Traj] No baseline actions TXT found; using straight-down Genesis baseline.")


    # ---- Run CEM / RL policy search ----
    best_theta, best_traj, best_cost = cem_optimize(
        robot=robot,
        knife_link=knife_link,
        knife_link_local_idx=knife_link_local_idx,
        q_init=q0,
        dofs_idx_arm=dofs_idx_arm,
        p_start=p_start,
        p_goal=p_goal,
        arm_limits_lo=arm_lo,
        arm_limits_hi=arm_hi,
        seed=0,
    )

    if best_traj is None:
        raise RuntimeError("CEM failed to produce a valid collision-free trajectory. "
                           "Try increasing MAX_ABS_OFFSET, N_ITERS, or changing IK masks/orientation.")

    print(f"[OK] Found trajectory with cost={best_cost:.6e}")
    print(f"[OK] Best theta_xy(knots) = {best_theta.reshape(N_KNOTS,2)}")

    # Optional dynamic validation (PD + step)
    if DO_DYNAMIC_VALIDATION:
        ok = dynamic_validate(scene, robot, best_traj)
        if not ok:
            print("[Warn] Dynamic validation failed (self-contact after stepping). "
                  "Try reducing DT_CMD, increasing gains, or adding more smoothness penalty.")

    # ---- Export as your playback format ----
    # Extract arm joint targets for both arms (degrees)
    q_n2 = np.tile(q0[indices["n2"]][None, :], (N_STEPS, 1))
    q_n5 = np.tile(q0[indices["n5"]][None, :], (N_STEPS, 1))

    # Replace the knife arm group from best_traj
    if KNIFE_ARM_KEY == "n2":
        q_n2 = best_traj[:, indices["n2"]]
    else:
        q_n5 = best_traj[:, indices["n5"]]

    nova2_deg = np.rad2deg(q_n2).astype(np.float64)
    nova5_deg = np.rad2deg(q_n5).astype(np.float64)

    # Hands: constant open signals (10ch)
    left_sig  = np.tile(DEFAULT_HAND_SIGNAL_10[None, :], (N_STEPS, 1))
    right_sig = np.tile(DEFAULT_HAND_SIGNAL_10[None, :], (N_STEPS, 1))

    write_ros_txt(OUT_DIR / "nova2.txt", DT_CMD, nova2_deg)
    write_ros_txt(OUT_DIR / "nova5.txt", DT_CMD, nova5_deg)
    write_ros_txt(OUT_DIR / "left.txt",  DT_CMD, left_sig.astype(np.int32))
    write_ros_txt(OUT_DIR / "right.txt", DT_CMD, right_sig.astype(np.int32))

    print("\n[Export] Wrote trajectory files:")
    print(" ", (OUT_DIR / "nova2.txt").resolve())
    print(" ", (OUT_DIR / "nova5.txt").resolve())
    print(" ", (OUT_DIR / "left.txt").resolve())
    print(" ", (OUT_DIR / "right.txt").resolve())
    print("\nRun your playback script by pointing its paths to these files.")
