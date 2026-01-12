#from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Dict, Tuple
import numpy as np


# -------------------------
# Math helpers
# -------------------------

def normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / (n + eps)

def sigmoid(x: float) -> float:
    # stable enough for typical ranges; clamp if you expect huge magnitudes
    return 1.0 / (1.0 + np.exp(-x))

def relu(x: float) -> float:
    return float(max(0.0, x))

def exp_safe(x: float) -> float:
    # avoid underflow/overflow in extreme cases
    x = float(np.clip(x, -80.0, 80.0))
    return float(np.exp(x))


# -------------------------
# Models / params
# -------------------------

@dataclass
class StrawberryModel:
    """
    Strawberry geometry as SDF + gradient.
    sdf(x): positive outside, 0 on surface, negative inside.
    grad_sdf(x): gradient of sdf at x (used for normal).
    curvature(x): optional mean curvature magnitude at x (or any bruise-risk proxy).
    """
    sdf: Callable[[np.ndarray], float]
    grad_sdf: Callable[[np.ndarray], np.ndarray]
    curvature: Optional[Callable[[np.ndarray], float]] = None

    # global properties (precomputed from your mesh/pointcloud)
    com: np.ndarray = np.zeros(3)              # center of mass (approx ok)
    axis_e: np.ndarray = np.array([0.0, 0.0, 1.0])  # principal axis (unit)
    h_min: float = -0.03                       # min projection along axis (on surface)
    h_max: float = +0.03                       # max projection along axis (on surface)
    mass: float = 0.02                         # kg (example: 20g strawberry)


@dataclass
class GraspPose:
    """
    Candidate grasp pose.
    R: 3x3 rotation matrix (columns are world axes of gripper frame)
    t: 3-vector position (assumed midpoint between fingers at closure)
    jaw_width: distance between inner finger pad faces (meters)
    """
    R: np.ndarray  # shape (3,3)
    t: np.ndarray  # shape (3,)
    jaw_width: float


@dataclass
class GripperModel:
    """
    For collision penalty: a small set of sample points on the gripper geometry in gripper frame.
    Increase density for better collision detection.
    """
    sample_points_local: np.ndarray  # shape (N,3)


@dataclass
class RewardParams:
    # Friction / antipodal
    mu: float = 0.7            # friction coefficient
    beta: float = 0.05         # sigmoid sharpness for antipodal term

    # Torque/COM alignment
    sigma_t: float = 0.015     # meters

    # Curvature / bruise risk
    sigma_kappa: float = 8.0   # depends on your curvature scale (tune!)

    # Preferred band along the strawberry axis
    h0: float = 0.55           # preferred normalized height in [0,1]
    sigma_h: float = 0.18
    tip_thresh: float = 0.15   # forbidden if below this normalized height
    stem_thresh: float = 0.90  # forbidden if above this normalized height

    # Pressure proxy (very rough)
    g: float = 9.81
    P_max: float = 35_000.0    # Pa (example threshold; tune with experiments)
    A0: float = 1.0e-4         # m^2 base "effective area" (e.g., 10mm x 10mm = 1e-4)
    gamma: float = 0.4         # area shrink vs curvature proxy

    # Contact finding
    pad_margin: float = 0.004  # meters: start ray a bit outside the finger pads
    ray_max_dist: float = 0.08 # meters: max tracing distance
    ray_eps: float = 1e-4
    ray_steps: int = 80

    # Weights
    w_anti: float = 2.0
    w_torque: float = 1.0
    w_band: float = 0.6
    w_curv: float = 0.6
    w_col: float = 80.0
    w_press: float = 2.0
    w_forbid: float = 10.0

    # Failure penalty if contacts can't be found
    fail_reward: float = -50.0


# -------------------------
# SDF contact + utilities
# -------------------------

def sphere_trace_to_surface(
    sdf: Callable[[np.ndarray], float],
    origin: np.ndarray,
    direction: np.ndarray,
    max_dist: float,
    eps: float,
    max_steps: int
) -> Tuple[bool, np.ndarray]:
    """
    Sphere-trace along a ray to find the first surface hit (phi ~= 0).
    Returns (hit, point).
    """
    d = normalize(direction)
    x = origin.astype(float).copy()
    traveled = 0.0

    # Ensure we start outside (phi >= 0) for "first-hit" behavior.
    # If we start inside, step outward a bit (best-effort).
    phi0 = float(sdf(x))
    if phi0 < 0.0:
        # Move outward along opposite direction until outside or until max_dist
        for _ in range(10):
            x = x - d * abs(phi0)
            phi0 = float(sdf(x))
            if phi0 >= 0.0:
                break

    prev_x = x.copy()
    prev_phi = float(sdf(x))

    for _ in range(max_steps):
        phi = float(sdf(x))

        if abs(phi) <= eps:
            return True, x

        # If phi is negative we are inside (due to imperfect SDF); bisect back to surface
        if phi < 0.0 and prev_phi > 0.0:
            lo, hi = prev_x, x
            for _ in range(20):
                mid = 0.5 * (lo + hi)
                pm = float(sdf(mid))
                if pm > 0:
                    lo = mid
                else:
                    hi = mid
                if abs(pm) <= eps:
                    return True, mid
            return True, 0.5 * (lo + hi)

        step = max(phi, eps)  # move at least eps
        step = float(min(step, max_dist - traveled))
        x_next = x + d * step

        traveled += step
        if traveled >= max_dist:
            break

        prev_x, prev_phi = x, phi
        x = x_next

    return False, x


def get_normal(model: StrawberryModel, x: np.ndarray) -> np.ndarray:
    return normalize(model.grad_sdf(x))

def get_curvature(model: StrawberryModel, x: np.ndarray) -> float:
    if model.curvature is None:
        return 0.0
    return float(model.curvature(x))


# -------------------------
# Reward terms
# -------------------------

def antipodal_score(n1: np.ndarray, n2: np.ndarray, u_hat: np.ndarray, mu: float, beta: float) -> float:
    """
    Smooth antipodal/fraction-cone score:
    s1 = (-n1 · u) and s2 = ( n2 · u) should be >= cos(arctan(mu)).
    """
    alpha = np.arctan(mu)
    cos_alpha = float(np.cos(alpha))

    s1 = float((-n1).dot(u_hat))
    s2 = float((+n2).dot(u_hat))

    return sigmoid((s1 - cos_alpha) / beta) * sigmoid((s2 - cos_alpha) / beta)


def torque_alignment_score(com: np.ndarray, p1: np.ndarray, u_hat: np.ndarray, sigma_t: float) -> float:
    """
    Distance from COM to grasp line (through p1 along u_hat).
    """
    v = (com - p1)
    proj = float(v.dot(u_hat))
    closest = p1 + proj * u_hat
    d = float(np.linalg.norm(com - closest))
    return exp_safe(-(d * d) / (sigma_t * sigma_t))


def band_score(model: StrawberryModel, m: np.ndarray, h0: float, sigma_h: float) -> Tuple[float, float]:
    """
    Preference for a band along the strawberry principal axis.
    Returns (score, h_norm).
    """
    e = normalize(model.axis_e)
    h = float((m - model.com).dot(e))
    denom = (model.h_max - model.h_min)
    if abs(denom) < 1e-9:
        h_norm = 0.5
    else:
        h_norm = (h - model.h_min) / denom
    score = exp_safe(-((h_norm - h0) ** 2) / (sigma_h ** 2))
    return score, h_norm


def curvature_score(k1: float, k2: float, sigma_kappa: float) -> float:
    return exp_safe(-((k1 * k1 + k2 * k2) / (sigma_kappa * sigma_kappa)))


def pressure_penalty(model: StrawberryModel, k1: float, k2: float, mu: float, A0: float, gamma: float, P_max: float, g: float) -> float:
    """
    Rough pressure proxy:
      F_min ≈ mg / (2 mu)
      A_eff(p) = A0 / (1 + gamma*kappa)
      P ≈ F_min / (A_eff1 + A_eff2)
    """
    if mu <= 1e-6:
        return 1e6  # effectively impossible to hold without slip

    F_min = (model.mass * g) / (2.0 * mu)

    A_eff1 = A0 / (1.0 + gamma * abs(k1))
    A_eff2 = A0 / (1.0 + gamma * abs(k2))
    A_eff = max(1e-9, (A_eff1 + A_eff2))

    P = F_min / A_eff
    return relu(P - P_max)


def collision_penalty(model: StrawberryModel, gripper: GripperModel, R: np.ndarray, t: np.ndarray) -> float:
    """
    Penalty for any gripper sample points that penetrate the object (phi < 0).
    """
    pts_w = (R @ gripper.sample_points_local.T).T + t[None, :]
    pen = 0.0
    for x in pts_w:
        pen += relu(-float(model.sdf(x)))
    return float(pen)


# -------------------------
# Main reward computation
# -------------------------

def compute_grasp_reward(
    model: StrawberryModel,
    grasp: GraspPose,
    gripper: GripperModel,
    params: RewardParams
) -> Tuple[float, Dict[str, float]]:
    """
    Returns (reward, components dict).
    """
    R = grasp.R
    t = grasp.t
    w = float(grasp.jaw_width)

    # Gripper axes in world (by convention)
    u = normalize(R[:, 0])  # closing direction
    a = normalize(R[:, 2])  # approach direction (not used directly here, but useful)

    # Contact ray origins: start outside each finger and trace inward
    # Left finger outside point is at -u * (w/2 + margin), ray goes +u
    # Right finger outside point is at +u * (w/2 + margin), ray goes -u
    o1 = t - u * (0.5 * w + params.pad_margin)
    o2 = t + u * (0.5 * w + params.pad_margin)

    hit1, p1 = sphere_trace_to_surface(model.sdf, o1, +u, params.ray_max_dist, params.ray_eps, params.ray_steps)
    hit2, p2 = sphere_trace_to_surface(model.sdf, o2, -u, params.ray_max_dist, params.ray_eps, params.ray_steps)

    if not (hit1 and hit2):
        return params.fail_reward, {
            "hit": 0.0,
            "r_total": params.fail_reward
        }

    # Compute normals / curvature at contacts
    n1 = get_normal(model, p1)
    n2 = get_normal(model, p2)
    k1 = get_curvature(model, p1)
    k2 = get_curvature(model, p2)

    # Derived grasp quantities
    u_hat = normalize(p2 - p1)  # actual contact-to-contact direction
    m = 0.5 * (p1 + p2)

    # Terms
    r_anti = antipodal_score(n1, n2, u_hat, params.mu, params.beta)
    r_torque = torque_alignment_score(model.com, p1, u_hat, params.sigma_t)
    r_band, h_norm = band_score(model, m, params.h0, params.sigma_h)
    r_curv = curvature_score(k1, k2, params.sigma_kappa)

    p_col = collision_penalty(model, gripper, R, t)
    p_press = pressure_penalty(model, k1, k2, params.mu, params.A0, params.gamma, params.P_max, params.g)
    p_forbid = 1.0 if (h_norm < params.tip_thresh or h_norm > params.stem_thresh) else 0.0

    # Total reward
    reward = (
        params.w_anti * r_anti
        + params.w_torque * r_torque
        + params.w_band * r_band
        + params.w_curv * r_curv
        - params.w_col * p_col
        - params.w_press * p_press
        - params.w_forbid * p_forbid
    )

    comps = {
        "hit": 1.0,
        "r_anti": float(r_anti),
        "r_torque": float(r_torque),
        "r_band": float(r_band),
        "r_curv": float(r_curv),
        "p_col": float(p_col),
        "p_press": float(p_press),
        "p_forbid": float(p_forbid),
        "h_norm": float(h_norm),
        "reward": float(reward),
    }
    return float(reward), comps


# -------------------------
# Example (toy SDF) to sanity-check wiring
# -------------------------

def make_sphere_sdf(center: np.ndarray, radius: float):
    center = center.astype(float).copy()

    def sdf(x: np.ndarray) -> float:
        return float(np.linalg.norm(x - center) - radius)

    def grad(x: np.ndarray) -> np.ndarray:
        v = x - center
        n = np.linalg.norm(v)
        if n < 1e-9:
            return np.array([1.0, 0.0, 0.0])
        return v / n

    return sdf, grad


if __name__ == "__main__":
    # Toy strawberry as a sphere
    sdf, grad = make_sphere_sdf(center=np.zeros(3), radius=0.02)

    strawberry = StrawberryModel(
        sdf=sdf,
        grad_sdf=grad,
        curvature=None,  # optional
        com=np.zeros(3),
        axis_e=np.array([0.0, 0.0, 1.0]),
        h_min=-0.02,
        h_max=+0.02,
        mass=0.02,
    )

    # Simple gripper collision samples (a few points around origin)
    sample_pts = np.array([
        [0.0, 0.0, 0.0],
        [0.01, 0.0, 0.0],
        [-0.01, 0.0, 0.0],
        [0.0, 0.01, 0.0],
        [0.0, -0.01, 0.0],
        [0.0, 0.0, 0.02],
    ])
    gripper = GripperModel(sample_points_local=sample_pts)

    # Identity orientation: u = x-axis, a = z-axis
    R = np.eye(3)
    grasp = GraspPose(R=R, t=np.array([0.0, 0.0, 0.0]), jaw_width=0.05)

    params = RewardParams()

    r, comps = compute_grasp_reward(strawberry, grasp, gripper, params)
    print("Reward:", r)
    print("Components:", comps)
