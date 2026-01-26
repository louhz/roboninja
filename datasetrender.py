# -*- coding: utf-8 -*-
"""
Synchronized multi-arm playback in Genesis with per-group start delays and faster animation,
ensuring both arms finish before stopping.

Features:
- Unified time grid across groups (perfect sync).
- Per-group start delay (playback seconds) via GROUP_DELAYS.
- SPEEDUP factor (compress timeline -> faster motion).
- Robust DoF/limit alignment (only resolved joints).
- Optional degree->radian conversion for arms.
- ***MODIFIED***: Correctly processes 0-255 linear control signals for 20-DoF hands.
- Playback length chosen so that both arms finish; configurable to "all groups".

Expected TXT format for each trajectory (ROS-like blocks):
  seq: <int>
  secs: <int>
  nsecs: <int>
  position: [ ... ]    # Arm positions in degrees; Hand positions as 0-255 signals.
"""

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


nova2_path = "./example_dataset/strawberry_real/episodes_19/control/nova2.txt"
nova5_path = "./example_dataset/strawberry_real/episodes_19/control/nova5.txt"
left_path  = "./example_dataset/strawberry_real/episodes_19/control/left.txt"
right_path = "./example_dataset/strawberry_real/episodes_19/control/right.txt"
video_out  = Path("renders") / "draw_strawberry.mp4"

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
    robot = scene.add_entity(gs.morphs.MJCF(file=MJCF_PATH))
    

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