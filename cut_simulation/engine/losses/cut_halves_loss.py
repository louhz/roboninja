import taichi as ti
from cut_simulation.engine.meshes.static import Static
from cut_simulation.engine.simulators import MPMSimulator
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
        
        self.loss = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=True)

        self.cut_loss = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=True)
        self.collision_loss = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=True)
        self.rotation_loss = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=True)
        self.move_loss = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=True)
        self.work_loss = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=True)
        self.work_step_loss = ti.field(dtype=DTYPE_TI, shape=(max_steps_global,), needs_grad=False)
        
        self.cut_weight = ti.field(dtype=DTYPE_TI, shape=())
        self.collision_weight = ti.field(dtype=DTYPE_TI, shape=())
        self.rotation_weight = ti.field(dtype=DTYPE_TI, shape=())
        self.move_weight = ti.field(dtype=DTYPE_TI, shape=())
        self.work_weight = ti.field(dtype=DTYPE_TI, shape=())
        self.x_bnd = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=False)
        self.x_bnd[None] = 0.5
        self.collision_point_num = 5

        self.smooth_eps = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=False)
        self.smooth_eps[None] = 1e-6

        # How aggressively you want distance to shrink each step:
        # e.g. 0.25 means "try to reduce distance by 25% per step"
        self.drop_frac = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=False)
        self.drop_frac[None] = 0.25

        # Scale for the progress term (multiplied again by cut_weight in sum_up_loss_kernel)
        self.progress_scale = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=False)
        self.progress_scale[None] = 1.0

        # Allow tiny penetration without huge penalty (numerical tolerance)
        self.penetration_margin = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=False)
        self.penetration_margin[None] = 1e-4

        # Optional debug curve: average positive gap to ground over time
        self.ground_dist_curve = ti.field(dtype=DTYPE_TI, shape=(max_steps_global + 1,), needs_grad=False)



    # def build(self, sim:MPMSimulator, knife:Rigid, bone: Static):
    #     self.sim = sim
    #     self.knife = knife
    #     self.bone = bone

    #     self.cut_weight[None] = self.weights['cut']
    #     self.collision_weight[None] = self.weights['collision']
    #     self.rotation_weight[None] = self.weights['rotation']
    #     self.move_weight[None] = self.weights['move']
    #     self.work_weight[None] = self.weights['work']


    
    def build(self, sim:MPMSimulator, knife:Rigid, ground:Rigid):
        self.sim = sim
        self.knife = knife
        self.ground = ground   # <-- IMPORTANT

        self.cut_weight[None] = self.weights['cut']
        self.collision_weight[None] = self.weights['collision']
        self.rotation_weight[None] = self.weights['rotation']
        self.move_weight[None] = self.weights['move']
        self.work_weight[None] = self.weights['work']

    def reset_grad(self):
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


    @ti.func
    def smooth_relu(self, x):
        # Smooth approximation of max(x, 0) to avoid kinked gradients at 0
        eps = self.smooth_eps[None]
        return 0.5 * (x + ti.sqrt(x * x + eps))

    @ti.func
    def knife_sample_point(self, step_i, k):
        # Same sampling scheme you already use: points across knife width
        # dir is perpendicular to knife forward direction in the xy-plane
        dir = ti.Vector([-ti.sin(self.knife.theta_k[step_i]),
                        ti.cos(self.knife.theta_k[step_i]), 0], DTYPE_TI)
        alpha = ti.cast(k, DTYPE_TI) / ti.cast(self.collision_point_num - 1, DTYPE_TI)
        return self.knife.pos_global[step_i] + dir * alpha * self.knife.mesh.knife_width[None]

    @ti.func
    def knife_ground_gap(self, step_i):
        # “Distance between knife and ground” proxy:
        # mean positive SDF over sampled points (>= 0).
        gap = ti.cast(0.0, DTYPE_TI)
        for k in ti.static(range(self.collision_point_num)):
            p = self.knife_sample_point(step_i, k)
            d = self.ground.sdf(p)          # signed distance to ground
            gap += self.smooth_relu(d)      # keep only above-ground distance
        return gap / ti.cast(self.collision_point_num, DTYPE_TI)

    @ti.kernel
    def compute_cut_loss1_kernel(self, step_num: ti.i32):
        # Minimize (time-weighted) squared gap to ground
        # w(t) increases with time -> waiting is expensive -> encourages fast descent
        for i in range(0, step_num + 1):
            gap = self.knife_ground_gap(i)  # >= 0
            t = (ti.cast(i, DTYPE_TI) + 1.0) / (ti.cast(step_num, DTYPE_TI) + 1.0)
            w = t * t  # quadratic time weight (increase steepness if needed)
            self.cut_loss[None] += w * gap * gap / (step_num + 1)
            self.ground_dist_curve[i] = gap  # debug


    
    @ti.kernel
    def compute_cut_loss2_kernel(self, step_num: ti.i32):
        # ✅ top-level is ONLY the loop
        for i in range(1, step_num + 1):
            frac = self.drop_frac[None]          # moved inside loop
            scale = self.progress_scale[None]    # moved inside loop

            gap_prev = self.knife_ground_gap(i - 1)
            gap_curr = self.knife_ground_gap(i)

            target = (1.0 - frac) * gap_prev
            err = self.smooth_relu(gap_curr - target)

            ti.atomic_add(self.cut_loss[None], scale * err * err / step_num)


    @ti.func
    def collision_loss_func(self, x):
        return x * x * x * x
    @ti.kernel
    def compute_collision_loss_kernel(self, step_num: ti.i32):
        # ✅ kernel top-level now contains ONLY the for-loop
        for i in range(0, step_num + 1):
            margin = self.penetration_margin[None]  # moved inside loop

            for k in ti.static(range(self.collision_point_num)):
                p = self.knife_sample_point(i, k)
                d = self.ground.sdf(p)
                pen = self.smooth_relu(-(d + margin))

                # optional: explicit atomic add (+= is also allowed by Taichi)
                ti.atomic_add(
                    self.collision_loss[None],
                    self.collision_loss_func(pen) / ((step_num + 1) * self.collision_point_num),
                )


    @ti.func
    def rotation_loss_func(self, x):
        return x * x
    @ti.kernel
    def compute_rotation_loss_kernel(self, step_num:ti.i32):
        for i in range(step_num):
            if self.knife.pos_global[i][1] > 0.2:
                self.rotation_loss[None] += self.rotation_loss_func(self.knife.theta_k[i + 1] - self.knife.theta_k[i]) / step_num

    @ti.func
    def move_loss_func(self, x):
        return x * x
    @ti.kernel
    def compute_move_loss_kernel(self, step_num:ti.i32):
        for i in range(0, step_num + 1):
            if self.knife.pos_global[i][1] > 0.2:
                self.move_loss[None] += self.move_loss_func(self.knife.theta_k[i] - self.knife.theta_v[i]) / (step_num + 1)

    @ti.kernel
    def compute_work_loss_kernel(self, substep_num:ti.i32):
        for i in range(substep_num):
            self.work_loss[None] += self.knife.work[i] / substep_num
            self.work_step_loss[i] += self.knife.work[i]

    @ti.kernel
    def sum_up_loss_kernel(self):
        self.loss[None] += self.cut_loss[None] * self.cut_weight[None]
        self.loss[None] += self.collision_loss[None] * self.collision_weight[None]
        self.loss[None] += self.rotation_loss[None] * self.rotation_weight[None]
        self.loss[None] += self.move_loss[None] * self.move_weight[None]
        self.loss[None] += self.work_loss[None] * self.work_weight[None]

    @ti.ad.grad_replaced
    def compute_loss(self, step_num, substep_num):
        self.compute_cut_loss1_kernel(step_num)
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
        self.compute_cut_loss2_kernel.grad(step_num)
        self.compute_cut_loss1_kernel.grad(step_num)


    def get_loss(self, step_num, substep_num):
        
        self.compute_loss(step_num, substep_num)
        loss_info = {
            'loss': self.loss[None],
            'ground_dist_curve': self.ground_dist_curve.to_numpy(),
            'cut_loss': self.cut_loss[None],
            'collision_loss': self.collision_loss[None],
            'rotation_loss': self.rotation_loss[None],
            'move_loss': self.move_loss[None],
            'work_loss': self.work_loss[None],
            'work_curve': self.work_step_loss.to_numpy()
        }
        return loss_info


    def get_loss_grad(self, step_num, substep_num):
        self.compute_loss_grad(step_num, substep_num)

    def clear(self):
        self.clear_loss()