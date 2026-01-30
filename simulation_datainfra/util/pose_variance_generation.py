# as you can see, we have the pair of the object pose and the end effector pose for the arm and hand()

# please finish this code

# now you need to densify the trajectory by if the object pose is moving, the arm need to move by the same offset as well. 



strawberry3=[
  1, [[-35.39, -0.545, -28.079, -19.487, 50.05, 35.079]],
  2, [[56.331, -2.573, 75.519, 29.073, -98.682, -46.062]],
  3, [[88, 233, 0, 0, 0, 0, 0, 0, 0, 255]],
  4, [[220, 0, 220, 167, 0, 0, 0, 0, 0, 240]],

  1,[[-18.52, 3.175, -64.94, 16.773, 47.957, 35.168],],
  4,[[220, 0, 220, 167, 0, 0, 0, 0, 0, 240],],
  1,[[-18.52, 6.516, -78.99, 27.481, 47.957, 35.17],],
  4,[[170, 0, 170, 167, 0, 0, 0, 0, 0, 240],],
  5,'right',
  5,'right',
  4,[[220, 0, 220, 167, 0, 0, 0, 0, 0, 240],],
  1,[[-18.52, 3.175, -64.94, 16.773, 47.957, 35.168],],
  
  2,[[56.331, -2.573, 75.519, 29.073, -98.682, -46.062],],
  5,'left',
  2,[
    [36.365, 25.274, 42.058, 31.042, -102.299, -66.045],
    [36.36, 22.507, 54.994, 20.871, -102.301, -66.055],
    [33.314, 0.776, 82.336, 14.594, -102.726, -69.146],
    ],
  5,'left',

  0,

]


EE_LINK_NAME = {
    "n2": "nova2ee_link3",   # <-- change to your nova2 end-effector link name
    "n5": "nova5ee_link3",   # <-- change to your nova5 end-effector link name
}

# the 6 degree vector is for the arm and the 10 vector is for the hand 



# the key checkpoint as you can seen is   2,[[56.331, -2.573, 75.519, 29.073, -98.682, -46.062],], 5,'left',

# [[-18.52, 6.516, -78.99, 27.481, 47.957, 35.17],],5,'right',

# this two are the vector that need to be moved with the object

# thus you need to compute for the end effector pose for this two moment by forward kinematic


# obtain the new pose of the object, move from the original one to the new one, \

# obtain the new control signal by the inverse kinematic given the new end effector pose


import numpy as np
import torch
import genesis as gs

def _yaw_to_quat_wxyz(yaw_rad: float) -> np.ndarray:
    """Yaw-only quaternion in Genesis convention (w, x, y, z)."""
    half = 0.5 * yaw_rad
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32)  # wxyz

def randomize_strawberry_on_table(
    obj_entity,
    table_xy_min=(-0.5, -0.5),
    table_xy_max=( 0.5,  0.5),
    table_center_xy=(0.0, 0.0),
    z_world=None,
    margin=0.03,
    yaw_range=(-np.pi, np.pi),
    joint_name="strawberry_free",
    settle_steps=10,
    scene=None,
    rng=None,
):
    """
    Randomize strawberry (x,y,yaw) within a table rectangle in WORLD frame.

    - If `joint_name` exists in obj_entity (e.g., fruit.xml includes table+strawberry),
      we set only that joint's qpos (7 values: xyz + wxyz quat).
    - Otherwise, we move the entity base link (works if obj_entity is just the strawberry).

    Args:
      obj_entity: the Genesis entity (your `draw1_obj`)
      table_xy_min/max: rectangle in table frame (meters)
      table_center_xy: added to sampled (x,y) (meters)
      z_world: if None, keep current z; else force this z
      margin: shrink rectangle by this amount (helps avoid edge penetrations)
      yaw_range: (min,max) yaw in radians
      joint_name: name of the strawberry free joint inside obj_entity (if present)
      settle_steps: after teleport, step physics a bit
      scene: pass your `scene` if you want settling steps applied
      rng: np.random.Generator (optional)
    """
    rng = rng if rng is not None else np.random.default_rng()

    # --- sample x,y within bounds (with margin) ---
    xmin, ymin = table_xy_min
    xmax, ymax = table_xy_max
    xmin += margin; ymin += margin
    xmax -= margin; ymax -= margin
    if xmin >= xmax or ymin >= ymax:
        raise ValueError(f"Invalid table bounds after margin: ({xmin},{ymin})..({xmax},{ymax})")

    x = rng.uniform(xmin, xmax) + float(table_center_xy[0])
    y = rng.uniform(ymin, ymax) + float(table_center_xy[1])
    yaw = rng.uniform(float(yaw_range[0]), float(yaw_range[1]))
    quat = _yaw_to_quat_wxyz(yaw)

    # --- decide Z: keep current z unless user forces z_world ---
    def _tensor_to_np(t):
        return t.detach().cpu().numpy()

    # Try: move an internal free joint called joint_name (best if fruit.xml has table+strawberry).
    moved_by_joint = False
    try:
        j = obj_entity.get_joint(joint_name)  # will throw if not found
        qs_idx_local = j.qs_idx_local  # local qpos indices for that joint
        cur = _tensor_to_np(obj_entity.get_qpos(qs_idx_local=qs_idx_local))
        if cur.shape[-1] != 7:
            raise RuntimeError(f"Joint '{joint_name}' qpos is not 7D (got {cur.shape}).")

        if z_world is None:
            z = float(cur[2])
        else:
            z = float(z_world)

        new_qpos = cur.copy()
        new_qpos[0:3] = np.array([x, y, z], dtype=np.float32)
        new_qpos[3:7] = quat.astype(np.float32)

        obj_entity.set_qpos(
            torch.as_tensor(new_qpos, dtype=gs.tc_float, device=gs.device),
            qs_idx_local=qs_idx_local,
            zero_velocity=True,
        )
        moved_by_joint = True
    except Exception:
        moved_by_joint = False

    # Fallback: move the entity base link (works if obj_entity is just the strawberry).
    if not moved_by_joint:
        if z_world is None:
            z = float(_tensor_to_np(obj_entity.get_pos())[2])
        else:
            z = float(z_world)

        obj_entity.set_pos(torch.tensor([x, y, z], dtype=gs.tc_float, device=gs.device), zero_velocity=True)
        obj_entity.set_quat(torch.as_tensor(quat, dtype=gs.tc_float, device=gs.device), zero_velocity=True)

    # Optional: let physics settle a few steps
    if scene is not None and settle_steps > 0:
        for _ in range(int(settle_steps)):
            scene.step()

    return {"x": x, "y": y, "z": z, "yaw": yaw, "quat_wxyz": quat, "used_joint": moved_by_joint}












import os
import ast
import numpy as np
from pathlib import Path
import genesis as gs

# ---------------------------
# Group parameters (names, kp, and joint limits in radians)
# ---------------------------
NOVA2_PARAMS = [
    {"joint": "nova2joint1", "kp": 300.0, "ctrlrange": (-6.28,  6.28)},
    {"joint": "nova2joint2", "kp": 300.0, "ctrlrange": (-3.14,  3.14)},
    {"joint": "nova2joint3", "kp": 300.0, "ctrlrange": (-2.79,  2.79)},
    {"joint": "nova2joint4", "kp": 250.0, "ctrlrange": (-6.28,  6.28)},
    {"joint": "nova2joint5", "kp": 200.0, "ctrlrange": (-6.28,  6.28)},
    {"joint": "nova2joint6", "kp": 1500.0, "ctrlrange": (-6.28,  6.28)},
]
NOVA5_PARAMS = [
    {"joint": "nova5joint1", "kp": 300.0, "ctrlrange": (-6.28,  6.28)},
    {"joint": "nova5joint2", "kp": 3000.0, "ctrlrange": (-3.14,  3.14)},
    {"joint": "nova5joint3", "kp": 3000.0, "ctrlrange": (-2.79,  2.79)},
    {"joint": "nova5joint4", "kp": 250.0, "ctrlrange": (-6.28,  6.28)},
    {"joint": "nova5joint5", "kp": 200.0, "ctrlrange": (-6.28,  6.28)},
    {"joint": "nova5joint6", "kp": 150.0, "ctrlrange": (-6.28,  6.28)},
]

# --- MODIFIED: Expanded to 20 DoF per hand based on scene.xml ---
RIGHT_HAND_PARAMS = [
    # Thumb (5 joints)
    {"joint": "R_thumb_cmc_roll",   "kp": 40, "ctrlrange": (0, 1.0427)},
    {"joint": "R_thumb_cmc_yaw",    "kp": 40, "ctrlrange": (0, 1.2043)},
    {"joint": "R_thumb_cmc_pitch",  "kp": 35, "ctrlrange": (0, 0.5146)},
    {"joint": "R_thumb_mcp",        "kp": 30, "ctrlrange": (0, 0.7152)},
    {"joint": "R_thumb_ip",         "kp": 25, "ctrlrange": (0, 0.7763)},
    # Index (4 joints)
    {"joint": "R_index_mcp_roll",   "kp": 30, "ctrlrange": (0, 0.2181)},
    {"joint": "R_index_mcp_pitch",  "kp": 35, "ctrlrange": (0, 1.3607)},
    {"joint": "R_index_pip",        "kp": 30, "ctrlrange": (0, 1.8317)},
    {"joint": "R_index_dip",        "kp": 25, "ctrlrange": (0, 1.8317)},
    # Middle (3 joints)
    {"joint": "R_middle_mcp_pitch", "kp": 35, "ctrlrange": (0, 1.3607)},
    {"joint": "R_middle_pip",       "kp": 30, "ctrlrange": (0, 1.8317)},
    {"joint": "R_middle_dip",       "kp": 25, "ctrlrange": (0, 0.6280)},
    # Ring (4 joints)
    {"joint": "R_ring_mcp_roll",    "kp": 30, "ctrlrange": (0, 0.2181)},
    {"joint": "R_ring_mcp_pitch",   "kp": 35, "ctrlrange": (0, 1.3607)},
    {"joint": "R_ring_pip",         "kp": 30, "ctrlrange": (0, 1.8317)},
    {"joint": "R_ring_dip",         "kp": 25, "ctrlrange": (0, 0.6280)},
    # Pinky (4 joints)
    {"joint": "R_pinky_mcp_roll",   "kp": 25, "ctrlrange": (0, 0.3489)},
    {"joint": "R_pinky_mcp_pitch",  "kp": 35, "ctrlrange": (0, 1.3607)},
    {"joint": "R_pinky_pip",        "kp": 30, "ctrlrange": (0, 1.8317)},
    {"joint": "R_pinky_dip",        "kp": 25, "ctrlrange": (0, 0.6280)},
]

LEFT_HAND_PARAMS = [
    # Thumb (5 joints)
    {"joint": "L_thumb_cmc_roll",   "kp": 40, "ctrlrange": (0, 1.0427)},
    {"joint": "L_thumb_cmc_yaw",    "kp": 40, "ctrlrange": (0, 1.2043)},
    {"joint": "L_thumb_cmc_pitch",  "kp": 35, "ctrlrange": (0, 0.5149)},
    {"joint": "L_thumb_mcp",        "kp": 30, "ctrlrange": (0, 0.7152)},
    {"joint": "L_thumb_ip",         "kp": 25, "ctrlrange": (0, 0.7763)},
    # Index (4 joints)
    {"joint": "L_index_mcp_roll",   "kp": 30, "ctrlrange": (0, 0.2181)},
    {"joint": "L_index_mcp_pitch",  "kp": 35, "ctrlrange": (0, 1.3607)},
    {"joint": "L_index_pip",        "kp": 30, "ctrlrange": (0, 1.8317)},
    {"joint": "L_index_dip",        "kp": 25, "ctrlrange": (0, 0.6280)},
    # Middle (3 joints)
    {"joint": "L_middle_mcp_pitch", "kp": 35, "ctrlrange": (0, 1.3607)},
    {"joint": "L_middle_pip",       "kp": 30, "ctrlrange": (0, 1.8317)},
    {"joint": "L_middle_dip",       "kp": 25, "ctrlrange": (0, 1.6280)},
    # Ring (4 joints)
    {"joint": "L_ring_mcp_roll",    "kp": 30, "ctrlrange": (0, 0.2181)},
    {"joint": "L_ring_mcp_pitch",   "kp": 35, "ctrlrange": (0, 1.3607)},
    {"joint": "L_ring_pip",         "kp": 30, "ctrlrange": (0, 1.8317)},
    {"joint": "L_ring_dip",         "kp": 25, "ctrlrange": (0, 0.6280)},
    # Pinky (4 joints)
    {"joint": "L_pinky_mcp_roll",   "kp": 25, "ctrlrange": (0, 0.3489)},
    {"joint": "L_pinky_mcp_pitch",  "kp": 35, "ctrlrange": (0, 1.3607)},
    {"joint": "L_pinky_pip",        "kp": 30, "ctrlrange": (0, 1.8317)},
    {"joint": "L_pinky_dip",        "kp": 25, "ctrlrange": (0, 0.6280)},
]



LEFT_HAND_POSE_SCALE = 0.95
RIGHT_HAND_POSE_SCALE = 1


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
    kv_eff = kv_scale * np.sqrt(np.maximum(kp_eff, 1e-6))  # avoid sqrt(0)
    entity.set_dofs_kp(kp_eff.astype(np.float32), dofs_idx_local=dof_idx)
    entity.set_dofs_kv(kv_eff.astype(np.float32), dofs_idx_local=dof_idx)

def load_ros_txt_positions(txt_path, key="position"):
    """
    Parse ROS-like text:
      seq: N
      secs: <int>
      nsecs: <int>
      position: [ ... ]
    Returns:
      t: (N,) seconds, relative to first sample
      q: (N, D) positions (float64)
    """
    t_list, q_list = [], []
    secs, nsecs = None, None
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s.startswith("secs:"):
                try: secs = int(s.split(":", 1)[1].strip())
                except: secs = None
            elif s.startswith("nsecs:"):
                try: nsecs = int(s.split(":", 1)[1].strip())
                except: nsecs = 0
            elif s.startswith(f"{key}:"):
                arr = ast.literal_eval(s.split(":", 1)[1].strip())
                q_list.append(np.asarray(arr, dtype=np.float64))
                if secs is not None:
                    t_list.append(float(secs) + float(nsecs or 0) * 1e-9)
                else:
                    t_list.append(len(t_list) * 1.0)
    if not q_list:
        raise RuntimeError(f"No '{key}:' entries found in {txt_path}")
    t = np.asarray(t_list, dtype=np.float64)
    q = np.vstack(q_list)
    # relative time & de-duplicate timestamps
    t -= t[0]
    uniq, idx = np.unique(t, return_index=True)
    return uniq, q[idx]

def maybe_to_rad(q, assume_degrees=True):
    return np.deg2rad(q).astype(q.dtype, copy=False) if assume_degrees else q

def process_hand_data(q_raw, hand_params, hand_prefix):
    """
    Maps 10-channel 0-255 control signals to 20-DoF joint angles in radians.
    Implements inverse linear mapping: 255 -> min_range, 0 -> max_range.
    """
    num_samples = q_raw.shape[0]
    num_joints = len(hand_params)
    q_processed = np.zeros((num_samples, num_joints), dtype=np.float64)

    name_to_idx = {p['joint']: i for i, p in enumerate(hand_params)}
    name_to_range = {p['joint']: p['ctrlrange'] for p in hand_params}

    # Input data column mapping based on user description
    # [a,b,c,d,e,f,g,h,i,j]
    # a: Thumb flexion
    # b: Thumb side-to-side (cmc_roll)
    # c: Index flexion
    # d: Middle flexion
    # e: Ring flexion
    # f: Pinky flexion
    # g: Index side-to-side (mcp_roll)
    # h: Ring side-to-side (mcp_roll)
    # i: Pinky side-to-side (mcp_roll)
    # j: Thumb rotation (cmc_yaw)
    
    input_map = {
        'thumb_flex': 0,
        'thumb_yaw': 1,
        'index_flex': 2,
        'middle_flex': 3,
        'ring_flex': 4,
        'pinky_flex': 5,
        'index_roll': 6,
        'ring_roll': 7,
        'pinky_roll': 8,
        'thumb_roll': 9,
    }

    joint_mapping = {
        'thumb_flex':  [f'{hand_prefix}_thumb_cmc_pitch', f'{hand_prefix}_thumb_mcp', f'{hand_prefix}_thumb_ip'],
        'thumb_roll':  [f'{hand_prefix}_thumb_cmc_roll'],
        'thumb_yaw':   [f'{hand_prefix}_thumb_cmc_yaw'],
        'index_flex':  [f'{hand_prefix}_index_mcp_pitch', f'{hand_prefix}_index_pip', f'{hand_prefix}_index_dip'],
        'middle_flex': [f'{hand_prefix}_middle_mcp_pitch', f'{hand_prefix}_middle_pip', f'{hand_prefix}_middle_dip'],
        'ring_flex':   [f'{hand_prefix}_ring_mcp_pitch', f'{hand_prefix}_ring_pip', f'{hand_prefix}_ring_dip'],
        'pinky_flex':  [f'{hand_prefix}_pinky_mcp_pitch', f'{hand_prefix}_pinky_pip', f'{hand_prefix}_pinky_dip'],
        'index_roll':  [f'{hand_prefix}_index_mcp_roll'],
        'ring_roll':   [f'{hand_prefix}_ring_mcp_roll'],
        'pinky_roll':  [f'{hand_prefix}_pinky_mcp_roll'],
    }

    for input_name, joint_names in joint_mapping.items():
        input_col_idx = input_map[input_name]
        s = q_raw[:, input_col_idx] # This is the 0-255 signal vector for all samples

        for joint_name in joint_names:
            if joint_name in name_to_idx:
                joint_idx = name_to_idx[joint_name]
                min_rad, max_rad = name_to_range[joint_name]
                
                # Inverse mapping: 255 -> min, 0 -> max
                q_processed[:, joint_idx] = min_rad + ((255.0 - s) / 255.0) * (max_rad - min_rad)
            else:
                print(f"[Warn] Mapped joint '{joint_name}' not in hand_params list.")

    return q_processed

def resample_to_grid(t_src, q_src, t_eval, hold_ends=True):
    """
    Interpolate q_src at t_eval (1D per-DoF).
    If hold_ends=True, extrapolation uses endpoints; else NaN.
    """
    Ngrid, D = len(t_eval), q_src.shape[1]
    q = np.zeros((Ngrid, D), dtype=np.float64)
    left_vals  = q_src[0]  if hold_ends else np.full(D, np.nan)
    right_vals = q_src[-1] if hold_ends else np.full(D, np.nan)
    for j in range(D):
        q[:, j] = np.interp(t_eval, t_src, q_src[:, j],
                            left=left_vals[j], right=right_vals[j])
    return q

def select_bounds(names_all, ok_names, lo_all, hi_all):
    """Keep limits only for DoFs that actually resolved, preserving order."""
    name2i = {n: i for i, n in enumerate(names_all)}
    sel = [name2i[n] for n in ok_names]
    return lo_all[sel], hi_all[sel]

# ---------------------------
# Paths (edit to yours)
# ---------------------------

# ---------------------------
# Tunables
# ---------------------------
ARM_INPUT_IN_DEGREES = True   # set False if your TXT logs for ARMS are already in radians
SPEEDUP = 2.0                   # >1.0 compresses the timeline (faster playback)
DT_SIMUL = 0.01                 # physics step (must match SimOptions dt)
DT_CMD   = 0.01                 # command update period (playback cadence)

# Per-group start delays (in playback seconds). Positive => start later.
GROUP_DELAYS = {
    "n2": 0.0,
    "n5": 0.0,  # e.g., set to 2.0 if you want NOVA5 to start 2 seconds later
    "lh": 0.0,
    "rh": 0.0,
}

# Which groups must finish before stopping (default: both arms)
REQUIRED_FINISH_GROUPS = ["n2", "n5"]
# If you prefer ALL active groups (including hands), set:
# REQUIRED_FINISH_GROUPS = "ALL"

# Tail seconds after playback completes (set to 0.5 if you want a brief settle)
TAIL_SECONDS = 0.0

MJCF_PATH  = "./robot_urdf_genesis/scene.xml"
# --- MODIFIED: Added path for the separate object XML ---
OBJECT_MJCF_PATH = "./fruit_asset/fruit.xml"
# --- END MODIFIED ---


nova2_path = "./data/strawberry_control_data/strawberry3/nova2.txt"
nova5_path = "./data/strawberry_control_data/strawberry3/nova5.txt"
left_path  = "./data/strawberry_control_data/strawberry3/left.txt"
right_path = "./data/strawberry_control_data/strawberry3/right.txt"
video_out  = Path("renders") / "draw_strawberry.mp4"



# ----------------------------
# Quaternion utils (Genesis convention: w,x,y,z)
# ----------------------------
def quat_normalize_wxyz(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n

def quat_inv_wxyz(q):
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = q
    n2 = np.dot(q, q)
    if n2 < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return np.array([w, -x, -y, -z], dtype=np.float64) / n2

def quat_mul_wxyz(q1, q2):
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)

def quat_apply_wxyz(q, v):
    """Rotate 3D vector v by quaternion q (wxyz)."""
    q = quat_normalize_wxyz(q)
    v = np.asarray(v, dtype=np.float64)
    w, x, y, z = q
    qvec = np.array([x, y, z], dtype=np.float64)
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2.0 * (w * uv + uuv)

def compute_delta_from_poses(obj_pos_old, obj_quat_old, obj_pos_new, obj_quat_new):
    """
    ΔT = T_new * inv(T_old)
    In quaternion form:
      qΔ = q_new * inv(q_old)
      pΔ = p_new - R(qΔ) * p_old
    """
    obj_quat_old = quat_normalize_wxyz(obj_quat_old)
    obj_quat_new = quat_normalize_wxyz(obj_quat_new)
    q_delta = quat_normalize_wxyz(quat_mul_wxyz(obj_quat_new, quat_inv_wxyz(obj_quat_old)))
    p_delta = np.asarray(obj_pos_new, dtype=np.float64) - quat_apply_wxyz(q_delta, np.asarray(obj_pos_old, dtype=np.float64))
    return p_delta, q_delta

def apply_delta_to_pose(pos, quat, p_delta, q_delta, apply_rotation=True):
    pos = np.asarray(pos, dtype=np.float64)
    quat = quat_normalize_wxyz(quat)
    pos2 = quat_apply_wxyz(q_delta, pos) + p_delta
    if apply_rotation:
        quat2 = quat_normalize_wxyz(quat_mul_wxyz(q_delta, quat))
    else:
        quat2 = quat
    return pos2, quat2

# ----------------------------
# Object pose getter (handles a free joint inside fruit.xml)
# ----------------------------
def get_object_pose_wxyz(obj_entity, joint_name="strawberry_free"):
    """
    Returns (pos(3,), quat_wxyz(4,), used_joint_bool)
    If joint_name exists and is 7D (xyz+wxyz), we use that.
    Else fall back to entity base pose.
    """
    try:
        j = obj_entity.get_joint(joint_name)
        qs_idx_local = j.qs_idx_local
        qpos7 = obj_entity.get_qpos(qs_idx_local=qs_idx_local).detach().cpu().numpy()
        if qpos7.shape[-1] == 7:
            pos = qpos7[0:3].astype(np.float64)
            quat = qpos7[3:7].astype(np.float64)
            return pos, quat, True
    except Exception:
        pass

    pos = obj_entity.get_pos().detach().cpu().numpy().astype(np.float64)
    quat = obj_entity.get_quat().detach().cpu().numpy().astype(np.float64)
    return pos, quat, False

# ----------------------------
# End-effector FK and IK retarget
# ----------------------------
def fk_link_pose(robot, link, qpos_full_np):
    """
    FK without disturbing sim state (Genesis forward_kinematics restores qpos internally). :contentReference[oaicite:2]{index=2}
    """
    qpos_t = torch.as_tensor(qpos_full_np, dtype=gs.tc_float, device=gs.device)
    links_pos, links_quat = robot.forward_kinematics(
        qpos_t,
        qs_idx_local=None,  # we pass full qpos
        links_idx_local=[link.idx_local],
    )
    pos = links_pos[0].detach().cpu().numpy().astype(np.float64)
    quat = links_quat[0].detach().cpu().numpy().astype(np.float64)  # wxyz
    return pos, quat

def retarget_arm_traj_by_object_delta(
    robot,
    ee_link,
    arm_dofs_idx_local,      # np array of 6 indices (local)
    q_arm_traj_rad,          # (T,6) in radians
    obj_pos_old, obj_quat_old,
    obj_pos_new, obj_quat_new,
    mask=None,               # optional (T,) bool, True means retarget this step
    apply_rotation=True,
    ik_pos_tol=5e-4,
    ik_rot_tol=5e-3,
    ik_max_samples=50,
    ik_max_iters=20,
    damping=0.01,
):
    """
    Dense retargeting: for each timestep:
      FK(q_arm) -> ee_pose_old
      ee_pose_new = Δ(obj_old->obj_new) * ee_pose_old
      IK(ee_pose_new) -> q_arm_new
    Uses Genesis inverse_kinematics() with dofs_idx_local to only solve that arm. :contentReference[oaicite:3]{index=3}
    """
    q_arm_traj_rad = np.asarray(q_arm_traj_rad, dtype=np.float64)
    T = q_arm_traj_rad.shape[0]
    assert q_arm_traj_rad.shape[1] == len(arm_dofs_idx_local)

    if mask is None:
        mask = np.ones((T,), dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        assert mask.shape == (T,)

    # Δ object transform
    p_delta, q_delta = compute_delta_from_poses(obj_pos_old, obj_quat_old, obj_pos_new, obj_quat_new)

    # Start from current robot qpos as baseline init (full length n_qs)
    qpos_base = robot.get_qpos().detach().cpu().numpy().astype(np.float64)  # length = robot.n_qs

    out = q_arm_traj_rad.copy()
    last_full_qpos = qpos_base.copy()

    for i in range(T):
        if not mask[i]:
            # keep original, but still update last_full_qpos so IK init stays consistent
            last_full_qpos[arm_dofs_idx_local] = out[i]
            continue

        # Build full qpos for FK init (keep other joints at last solution)
        qpos_fk = last_full_qpos.copy()
        qpos_fk[arm_dofs_idx_local] = q_arm_traj_rad[i]

        ee_pos_old, ee_quat_old = fk_link_pose(robot, ee_link, qpos_fk)
        ee_pos_new, ee_quat_new = apply_delta_to_pose(
            ee_pos_old, ee_quat_old, p_delta, q_delta, apply_rotation=apply_rotation
        )

        # IK init must be FULL qpos of length n_qs (Genesis checks this). :contentReference[oaicite:4]{index=4}
        qpos_sol, err = robot.inverse_kinematics(
            link=ee_link,
            pos=ee_pos_new,
            quat=ee_quat_new,
            init_qpos=qpos_fk,
            respect_joint_limit=True,
            max_samples=ik_max_samples,
            max_solver_iters=ik_max_iters,
            damping=damping,
            pos_tol=ik_pos_tol,
            rot_tol=ik_rot_tol,
            dofs_idx_local=arm_dofs_idx_local,
            return_error=True,
        )

        qpos_sol_np = qpos_sol.detach().cpu().numpy().astype(np.float64)
        out[i] = qpos_sol_np[arm_dofs_idx_local]
        last_full_qpos = qpos_sol_np  # warm-start next timestep (smoothness)

    return out



if __name__ == "__main__":
    video_out.parent.mkdir(parents=True, exist_ok=True)

    # ---- Genesis scene ----
    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT_SIMUL),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -3.5, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
            max_FPS=120,
        ),
        show_viewer=True,
    )
    scene.add_entity(gs.morphs.Plane())
    
    # --- MODIFIED: Load robot and object as separate entities ---
    # robot 实体只包含机械臂和手（来自修改后的 scene.xml）
    robot = scene.add_entity(gs.morphs.MJCF(file=MJCF_PATH, requires_jac_and_IK=True))

    

    SCALE = 0.1

    OFFSET = np.array([0.0, 0.5, 0.00])  
    # draw1_obj 实体是可抓取的物体（来自 object.xml）
    if Path(OBJECT_MJCF_PATH).exists():
        draw1_obj = scene.add_entity(
            gs.morphs.MJCF(file=OBJECT_MJCF_PATH, scale=SCALE)
        )
        
        print(f"[Load] Added object from {OBJECT_MJCF_PATH}")
    else:
        print(f"[Warn] Object file not found, skipping: {OBJECT_MJCF_PATH}")
    # --- END MODIFIED ---
    
    cam = scene.add_camera(
        res=(960, 640),
        pos=(2.8, -3.5, 2.3),
        lookat=(0.0, 0.0, 0.5),
        fov=35,
        GUI=False,
    )
    scene.build()
     


    # ---- Capture ORIGINAL strawberry pose (the pose your demo was recorded at) ----
    obj_pos_old, obj_quat_old, used_joint = get_object_pose_wxyz(draw1_obj, joint_name="strawberry_free")

    # ---- Randomize object pose (new episode) ----
    rand = randomize_strawberry_on_table(
        draw1_obj,
        table_xy_min=(-0.5, -0.5),
        table_xy_max=( 0.5,  0.5),
        table_center_xy=(0.0, 0.0),
        joint_name="strawberry_free",
        settle_steps=10,
        scene=scene,
    )

    # ---- Capture NEW pose ----
    obj_pos_new, obj_quat_new, _ = get_object_pose_wxyz(draw1_obj, joint_name="strawberry_free")

    print("[Obj] old pos/quat:", obj_pos_old, obj_quat_old, "used_joint=", used_joint)
    print("[Obj] new pos/quat:", obj_pos_new, obj_quat_new)

    # ---- Resolve DoFs and set PD gains ----
    # 重要：resolve_dofs 现在只在 'robot' 实体上运行，
    # 它不再知道 'draw1_obj' 的自由关节，错误得以解决。
    groups = {"n2": NOVA2_PARAMS, "n5": NOVA5_PARAMS, "lh": LEFT_HAND_PARAMS, "rh": RIGHT_HAND_PARAMS}
    indices, bounds, gains = {}, {}, {}
    for key, params in groups.items():
        names, kp, lo, hi = params_to_arrays(params)
        # 回传关节序列号，寻找到的关节名
        idx, ok, missing = resolve_dofs(robot, names) #
        if missing:
            print(f"[Warn] Missing joints [{key}]: {missing}")
        if idx.size == 0:
            print(f"[Skip] {key}: no resolved DoFs.")
            continue
        # 根据ok在names中找对应关节上下限
        lo_sel, hi_sel = select_bounds(names, ok, lo, hi)
        bounds[key]  = (lo_sel, hi_sel)             # (D,)
        indices[key] = idx                          # (D,)
        gains[key]   = kp[:len(ok)].copy()          # align kp to resolved DoFs
    
    
    # 统一调整刚度和阻尼
    for key in list(indices.keys()):
        if key in ('lh', 'rh'):
            # 5–8x hand stiffness is a good starting point; kv follows sqrt(kp)
            set_group_gains(robot, indices[key], gains[key], kv_scale=2.0, kp_scale=6.0)

            # Allow the controller to actually output that force.
            # Start with ±12 N·m per hand DOF (tune as needed; units are SI).
            hand_idx = indices[key]
            n = len(hand_idx)
            robot.set_dofs_force_range(
                lower=-np.full(n, 12.0, dtype=np.float32),
                upper= np.full(n, 12.0, dtype=np.float32),
                dofs_idx_local=hand_idx,
            )
        else:
            # keep arms as they were
            set_group_gains(robot, indices[key], gains[key], kv_scale=2.0, kp_scale=1.0)
    # ---- Load trajectories ----
    data_paths = {"n2": nova2_path, "n5": nova5_path, "lh": left_path, "rh": right_path}
    loaded = {}
    for key, path in data_paths.items():
        if key not in indices:
            continue
        p = Path(path)
        if not p.exists():
            print(f"[Skip] {key}: file not found -> {path}")
            continue
        # 返回时间步和运动序列
        t_src, q_src = load_ros_txt_positions(path)

        # --- MODIFIED: Use new processing for hands, old for arms ---
        if key == 'lh':
            q_src = process_hand_data(q_src, LEFT_HAND_PARAMS, 'L')
        elif key == 'rh':
            q_src = process_hand_data(q_src, RIGHT_HAND_PARAMS, 'R')
        else: # For arms 'n2' and 'n5'
            q_src = maybe_to_rad(q_src, assume_degrees=ARM_INPUT_IN_DEGREES)

        loaded[key] = (t_src, q_src)
        print(f"[Load] {key}: {len(t_src)} samples, {q_src.shape[1]} DoFs, duration={t_src[-1]:.3f}s from {path}")
        
    if not loaded:
        raise RuntimeError("No playable groups found (missing files or no DoFs resolved).")

    # ---- Unified time grid that ensures required groups finish ----
    S = max(SPEEDUP, 1)

    def need_playback_seconds(key: str, t_src: np.ndarray) -> float:
        """Required playback seconds so that group 'key' finishes (includes its start delay)."""
        T_k = float(t_src[-1])            # source seconds
        d_k = float(GROUP_DELAYS.get(key, 0.0))  # playback seconds
        return T_k / S + d_k

    if REQUIRED_FINISH_GROUPS == "ALL":
        keys_for_stop = list(loaded.keys())
    else:
        keys_for_stop = [k for k in (REQUIRED_FINISH_GROUPS or []) if k in loaded]
        if not keys_for_stop:
            keys_for_stop = list(loaded.keys())  # fallback

    T_playback = max(need_playback_seconds(k, loaded[k][0]) for k in keys_for_stop)

    steps_per_cmd    = max(1, int(round(DT_CMD / DT_SIMUL)))
    # 按0.01对时间进行分割
    t_grid_play      = np.arange(0.0, T_playback + 1e-9, DT_CMD, dtype=np.float64)
    t_eval_src_base  = t_grid_play * S
    TARGET_STEPS     = len(t_grid_play)
    print(f"[Sync] Ensure finish for {keys_for_stop}. SPEEDUP={SPEEDUP:.2f}x, "
          f"steps={TARGET_STEPS}, playback≈{TARGET_STEPS*DT_CMD:.2f}s")

    # ---- Prepare command arrays (apply per-group delay, align DoFs, clamp limits) ----
    q_cmds = {}
    for key, (t_src, q_src) in loaded.items():
        delay_play = GROUP_DELAYS.get(key, 0.0)             # seconds in playback time
        t_eval_src = t_eval_src_base - delay_play * S       # holds first pose before delay
        q = resample_to_grid(t_src, q_src, t_eval_src, hold_ends=True)  # (Nplay, Dsrc)

        idx = indices[key]
        D_target = idx.size
        D_src    = q.shape[1]
        if D_src != D_target:
            q_aligned = np.zeros((TARGET_STEPS, D_target), dtype=q.dtype)
            m = min(D_src, D_target)
            q_aligned[:, :m] = q[:, :m]
            q = q_aligned
            print(f"[Dim] {key}: mapped {D_src} -> {D_target} columns")

        lo, hi = bounds[key]
        lo = lo.astype(np.float64); hi = hi.astype(np.float64)

        # --- NEW: softly reduce left-hand grasp amplitude across the whole trajectory ---
        if key == 'lh':
            # scale about the lower bound so the result stays inside [lo, hi] even if lo != 0
            q = lo[None, :] + LEFT_HAND_POSE_SCALE * (q - lo[None, :])

        if key == 'rh':
            q = lo[None, :] + RIGHT_HAND_POSE_SCALE * (q - lo[None, :])


        # final safety clamp
        q = np.minimum(np.maximum(q, lo[None, :]), hi[None, :])

        q_cmds[key] = (q, idx)

    if not q_cmds:
        raise RuntimeError("Nothing to play after alignment.")
    

    # ------------------------------------------------------------
    # Densify/retarget: if object pose changed, move arms by same Δ
    # ------------------------------------------------------------
    for arm_key in ("n2", "n5"):
        if arm_key not in q_cmds:
            continue
        if arm_key not in EE_LINK_NAME:
            print(f"[Skip] No EE link name for {arm_key}")
            continue

        try:
            ee_link = robot.get_link(EE_LINK_NAME[arm_key])
        except Exception as e:
            print(f"[Skip] Cannot find EE link '{EE_LINK_NAME[arm_key]}' for {arm_key}: {e}")
            continue

        q_traj, dofs_idx = q_cmds[arm_key]  # q_traj shape (T,6)
        print(f"[Retarget] {arm_key}: retargeting {q_traj.shape[0]} steps using FK->Δobj->IK")

        q_new = retarget_arm_traj_by_object_delta(
            robot=robot,
            ee_link=ee_link,
            arm_dofs_idx_local=dofs_idx,
            q_arm_traj_rad=q_traj,
            obj_pos_old=obj_pos_old, obj_quat_old=obj_quat_old,
            obj_pos_new=obj_pos_new, obj_quat_new=obj_quat_new,
            mask=None,                 # or a boolean mask to retarget only grasp segment
            apply_rotation=True,       # keep relative EE orientation wrt object
            ik_pos_tol=5e-4,
            ik_rot_tol=5e-3,
            ik_max_samples=50,
            ik_max_iters=20,
            damping=0.01,
        )

        # safety clamp to your joint limits (even though IK respects limits)
        lo, hi = bounds[arm_key]
        q_new = np.minimum(np.maximum(q_new, lo[None, :]), hi[None, :])

        q_cmds[arm_key] = (q_new, dofs_idx)
        print(f"[Retarget] {arm_key}: done.")


    # ---- Move to the first sample and settle ----
    robot.zero_all_dofs_velocity()
    for key, (q, idx) in q_cmds.items():
        robot.set_dofs_position(q[0], idx)
    for _ in range(30):
        scene.step()

    # ---- Record playback ----
    ENABLE_DEBUG_PRINTING = True

    cam.start_recording()
    print("\n[Debug] Starting simulation loop. Target vs Actual positions will be reported.")

    for i in range(TARGET_STEPS):
        
        if ENABLE_DEBUG_PRINTING and i > 0:
            print(f"\n--- Checking state at start of step {i} (result of target {i-1}) ---")

            for key, (q_all_targets, idx) in q_cmds.items():
                target_q_rad = q_all_targets[i-1]
                actual_q_rad = robot.get_dofs_position(idx).cpu().numpy()
                error_rad = actual_q_rad - target_q_rad
                error_deg = np.rad2deg(error_rad)
                with open(f"{idx}.txt", "a", encoding="utf-8") as f:
                    f.write(f"{i},{key}," + ",".join(f"{v:.6f}" for v in error_deg) + "\n")
                error_norm_deg = np.linalg.norm(error_deg)
                per_joint_error_str = np.array2string(error_deg, precision=2, floatmode='fixed', sign=' ')

                print(f"  Group '{key}': ||Error|| = {error_norm_deg:8.4f} deg")
                if error_norm_deg > 0.1:
                    print(f"    - Target (rad): {np.array2string(target_q_rad, precision=3, floatmode='fixed')}")
                    print(f"    - Actual (rad): {np.array2string(actual_q_rad, precision=3, floatmode='fixed')}")
                    print(f"    - Error  (deg): {per_joint_error_str}")

        for key, (q, idx) in q_cmds.items():
            robot.control_dofs_position(q[i], idx)

        for _ in range(steps_per_cmd):
            scene.step()
            cam.render()

    for _ in range(int(TAIL_SECONDS / DT_SIMUL)):
        scene.step(); cam.render()

    cam.stop_recording(save_to_filename=str(video_out), fps=int(round(1.0 / DT_SIMUL)))
    print(f"\n[OK] Saved video to: {video_out.resolve()}")









