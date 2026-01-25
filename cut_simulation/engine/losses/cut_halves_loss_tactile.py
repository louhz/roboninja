import taichi as ti

from cut_simulation.engine.simulators import MPMSimulatortactile
from cut_simulation.engine.agents import Rigid
from cut_simulation.configs.macros import *


@ti.func
def clip(x):
    return ti.max(ti.min(x, 0.2), 0.025)


@ti.data_oriented
class Cuthalfloss:
    def __init__(self, max_action_steps_global, max_steps_global, weights):
        self.weights = weights
        self.max_action_steps_global = max_action_steps_global
        self.dim = 3

        # --------- Scalars (need grad) ----------
        self.loss = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=True)

        self.cut_loss = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=True)
        self.collision_loss = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=True)
        self.rotation_loss = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=True)
        self.move_loss = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=True)
        self.work_loss = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=True)

        # --------- Curves (no grad) ----------
        self.work_step_loss = ti.field(dtype=DTYPE_TI, shape=(max_steps_global,), needs_grad=False)
        self.ground_dist_curve = ti.field(dtype=DTYPE_TI, shape=(max_steps_global + 1,), needs_grad=False)

        # --------- Weights ----------
        self.cut_weight = ti.field(dtype=DTYPE_TI, shape=())
        self.collision_weight = ti.field(dtype=DTYPE_TI, shape=())
        self.rotation_weight = ti.field(dtype=DTYPE_TI, shape=())
        self.move_weight = ti.field(dtype=DTYPE_TI, shape=())
        self.work_weight = ti.field(dtype=DTYPE_TI, shape=())

        # --------- Hyperparams ----------
        self.x_bnd = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=False)
        self.x_bnd[None] = 0.5

        self.collision_point_num = 5

        self.smooth_eps = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=False)
        self.smooth_eps[None] = 1e-6

        # "progress" term
        self.drop_frac = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=False)
        self.drop_frac[None] = 0.25

        self.progress_scale = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=False)
        self.progress_scale[None] = 1.0

        self.penetration_margin = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=False)
        self.penetration_margin[None] = 1e-4

        # stored for python-side reporting (no grads)
        self._max_steps_global = max_steps_global

    # -------------------- wiring --------------------

    def build(self, sim: MPMSimulatortactile, knife: Rigid, ground):
        """
        ground: can be Static or any object that implements sdf(p: ti.Vector) -> signed distance.
        """
        if not hasattr(ground, "sdf"):
            raise AttributeError("Cuthalfloss.build(...): ground must implement sdf(p).")

        self.sim = sim
        self.knife = knife
        self.ground = ground

        self.cut_weight[None] = self.weights['cut']
        self.collision_weight[None] = self.weights['collision']
        self.rotation_weight[None] = self.weights['rotation']
        self.move_weight[None] = self.weights['move']
        self.work_weight[None] = self.weights['work']

    def reset_grad(self):
        # For custom grad: loss.grad must be seeded
        self.loss.grad[None] = 1
        self.cut_loss.grad[None] = 0
        self.collision_loss.grad[None] = 0
        self.rotation_loss.grad[None] = 0
        self.move_loss.grad[None] = 0
        self.work_loss.grad[None] = 0

    @ti.kernel
    def clear_loss(self):
        self.loss[None] = 0
        self.cut_loss[None] = 0
        self.collision_loss[None] = 0
        self.rotation_loss[None] = 0
        self.move_loss[None] = 0
        self.work_loss[None] = 0

        self.work_step_loss.fill(0)
        self.ground_dist_curve.fill(0)

    # -------------------- helpers --------------------

    @ti.func
    def smooth_relu(self, x):
        eps = self.smooth_eps[None]
        return 0.5 * (x + ti.sqrt(x * x + eps))

    @ti.func
    def knife_sample_point(self, step_i, k):
        # sample across knife width in xy-plane perpendicular to knife heading
        dir = ti.Vector([
            -ti.sin(self.knife.theta_k[step_i]),
             ti.cos(self.knife.theta_k[step_i]),
             0.0
        ], dt=DTYPE_TI)

        denom = ti.cast(self.collision_point_num - 1, DTYPE_TI)
        alpha = ti.cast(k, DTYPE_TI) / denom
        return self.knife.pos_global[step_i] + dir * alpha * self.knife.mesh.knife_width[None]

    @ti.func
    def knife_ground_gap(self, step_i):
        gap = ti.cast(0.0, DTYPE_TI)
        for k in ti.static(range(self.collision_point_num)):
            p = self.knife_sample_point(step_i, k)
            d = self.ground.sdf(p)
            gap += self.smooth_relu(d)  # positive (above-ground) only
        return gap / ti.cast(self.collision_point_num, DTYPE_TI)

    # -------------------- kernels --------------------

    @ti.kernel
    def compute_cut_loss1_kernel(self, step_num: ti.i32):
        # time-weighted squared gap (encourage fast descent)
        denom = ti.cast(step_num + 1, DTYPE_TI)

        for i in range(0, step_num + 1):
            gap = self.knife_ground_gap(i)
            t = (ti.cast(i, DTYPE_TI) + 1.0) / denom
            w = t * t
            ti.atomic_add(self.cut_loss[None], w * gap * gap / denom)
            self.ground_dist_curve[i] = gap

    @ti.kernel
    def compute_cut_loss2_kernel(self, step_num: ti.i32):
        # encourage monotonic progress in gap shrinkage
        for i in range(1, step_num + 1):
            frac = self.drop_frac[None]
            scale = self.progress_scale[None]

            gap_prev = self.knife_ground_gap(i - 1)
            gap_curr = self.knife_ground_gap(i)

            target = (1.0 - frac) * gap_prev
            err = self.smooth_relu(gap_curr - target)

            # (safe since loop empty if step_num==0)
            ti.atomic_add(self.cut_loss[None], scale * err * err / ti.cast(step_num, DTYPE_TI))

    @ti.func
    def collision_loss_func(self, x):
        return x * x * x * x

    @ti.kernel
    def compute_collision_loss_kernel(self, step_num: ti.i32):
        for i in range(0, step_num + 1):
            margin = self.penetration_margin[None]

            for k in ti.static(range(self.collision_point_num)):
                p = self.knife_sample_point(i, k)
                d = self.ground.sdf(p)
                pen = self.smooth_relu(-(d + margin))  # >0 if penetrating

                ti.atomic_add(
                    self.collision_loss[None],
                    self.collision_loss_func(pen) / ti.cast((step_num + 1) * self.collision_point_num, DTYPE_TI)
                )

    @ti.func
    def rotation_loss_func(self, x):
        return x * x

    @ti.kernel
    def compute_rotation_loss_kernel(self, step_num: ti.i32):
        # step_num==0 => loop empty
        denom = ti.cast(ti.max(step_num, 1), DTYPE_TI)
        for i in range(step_num):
            if self.knife.pos_global[i][1] > 0.2:
                ti.atomic_add(
                    self.rotation_loss[None],
                    self.rotation_loss_func(self.knife.theta_k[i + 1] - self.knife.theta_k[i]) / denom
                )

    @ti.func
    def move_loss_func(self, x):
        return x * x

    @ti.kernel
    def compute_move_loss_kernel(self, step_num: ti.i32):
        denom = ti.cast(step_num + 1, DTYPE_TI)
        for i in range(0, step_num + 1):
            if self.knife.pos_global[i][1] > 0.2:
                ti.atomic_add(
                    self.move_loss[None],
                    self.move_loss_func(self.knife.theta_k[i] - self.knife.theta_v[i]) / denom
                )

    @ti.kernel
    def compute_work_loss_kernel(self, substep_num: ti.i32):
        # substep_num==0 => loop empty
        denom = ti.cast(ti.max(substep_num, 1), DTYPE_TI)
        for i in range(substep_num):
            ti.atomic_add(self.work_loss[None], self.knife.work[i] / denom)
            self.work_step_loss[i] = self.knife.work[i]

    @ti.kernel
    def sum_up_loss_kernel(self):
        self.loss[None] = 0
        self.loss[None] += self.cut_loss[None] * self.cut_weight[None]
        self.loss[None] += self.collision_loss[None] * self.collision_weight[None]
        self.loss[None] += self.rotation_loss[None] * self.rotation_weight[None]
        self.loss[None] += self.move_loss[None] * self.move_weight[None]
        self.loss[None] += self.work_loss[None] * self.work_weight[None]

    # -------------------- AD wrapper --------------------

    @ti.ad.grad_replaced
    def compute_loss(self, step_num, substep_num):
        # IMPORTANT FIX: clear each time so losses don't accumulate across calls
        self.clear_loss()

        self.compute_cut_loss1_kernel(step_num)
        # only do cut_loss2 if step_num>=1 (avoid division by 0 inside kernel)
        if step_num >= 1:
            self.compute_cut_loss2_kernel(step_num)

        self.compute_collision_loss_kernel(step_num)
        self.compute_rotation_loss_kernel(step_num)
        self.compute_move_loss_kernel(step_num)
        self.compute_work_loss_kernel(substep_num)
        self.sum_up_loss_kernel()

    @ti.ad.grad_for(compute_loss)
    def compute_loss_grad(self, step_num, substep_num):
        self.sum_up_loss_kernel.grad()
        self.compute_work_loss_kernel.grad(substep_num)
        self.compute_move_loss_kernel.grad(step_num)
        self.compute_rotation_loss_kernel.grad(step_num)
        self.compute_collision_loss_kernel.grad(step_num)
        if step_num >= 1:
            self.compute_cut_loss2_kernel.grad(step_num)
        self.compute_cut_loss1_kernel.grad(step_num)

    # -------------------- public API --------------------

    def get_loss(self, step_num, substep_num, return_tactile=True, return_tactile_particles=False):
        """
        Returns a python dict. By default includes current tactile reading if enabled.
        """
        self.compute_loss(step_num, substep_num)

        info = {
            'loss': float(self.loss[None]),
            'ground_dist_curve': self.ground_dist_curve.to_numpy(),
            'cut_loss': float(self.cut_loss[None]),
            'collision_loss': float(self.collision_loss[None]),
            'rotation_loss': float(self.rotation_loss[None]),
            'move_loss': float(self.move_loss[None]),
            'work_loss': float(self.work_loss[None]),
            'work_curve': self.work_step_loss.to_numpy(),
        }

        if return_tactile and hasattr(self, "sim") and getattr(self.sim, "has_tactile_sensor", False):
            info["tactile"] = self.sim.get_tactile()  # {"contact":..., "pulse":..., "count":...}

            if return_tactile_particles:
                tp = self.sim.get_tactile_particles()
                if tp is not None:
                    contact_np, pulse_np = tp
                    info["tactile_particles_contact"] = contact_np
                    info["tactile_particles_pulse"] = pulse_np

        return info

    def get_loss_grad(self, step_num, substep_num):
        # assumes compute_loss(...) already ran and loss.grad seeded (env.reset_grad does that)
        self.compute_loss_grad(step_num, substep_num)

    def clear(self):
        self.clear_loss()
