import numpy as np
import taichi as ti
import warnings

from cut_simulation.engine.simulators import MPMSimulatortactile
from cut_simulation.engine.agents import *
from cut_simulation.engine.meshes import Statics
from cut_simulation.engine.renderer import Renderer
from cut_simulation.engine.bodies import Bodies
from cut_simulation.engine.losses import Cuthalfloss
from cut_simulation.configs.macros import *
from cut_simulation.utils.misc import *


@ti.data_oriented
class TaichiEnvtactile:
    """
    TaichiEnvtactile wraps all components in a simulation environment
    and exposes tactile sensor configuration + readings when using MPMSimulatortactile.
    """

    def __init__(
        self,
        dim=3,
        quality=1,
        particle_density=1e6,
        max_steps_local=160,
        device_memory_GB=42,
        max_steps_global=2500,
        horizon=61,
        ckpt_dest='cpu',
        gravity=(0.0, -10.0, 0.0),

        # NEW (optional): pass tactile config dict at construction time
        tactile_sensor_cfg=None,
    ):
        # ---------------------------------------------------------------------
        # Pre-check (and optionally auto-fix) max_steps_local divisibility
        # MPMSimulatortactile does:
        #   dt = 2e-4 / quality / 1.5
        #   n_substeps = int(2e-3 / dt) ~= int(15 * quality)
        #   assert max_steps_local % n_substeps == 0
        # Your default max_steps_local=160 will fail for quality=1 (n_substeps=15).
        # ---------------------------------------------------------------------
        dt_tmp = 2e-4 / quality / 1.5
        n_substeps_tmp = int(2e-3 / dt_tmp)

        if max_steps_local % n_substeps_tmp != 0:
            # Choose a sane nearby value (round DOWN), but keep >= n_substeps
            new_msl = (max_steps_local // n_substeps_tmp) * n_substeps_tmp
            if new_msl < n_substeps_tmp:
                new_msl = n_substeps_tmp

            warnings.warn(
                f"[TaichiEnvtactile] max_steps_local={max_steps_local} is not divisible by "
                f"n_substeps={n_substeps_tmp} (quality={quality}). "
                f"Auto-adjusting max_steps_local -> {new_msl} to satisfy the simulator assertion."
            )
            max_steps_local = new_msl

        if n_substeps_tmp * horizon >= max_steps_global:
            raise ValueError(
                f"Invalid horizon/max_steps_global: n_substeps*horizon = {n_substeps_tmp*horizon} "
                f"must be < max_steps_global ({max_steps_global}). "
                f"(quality={quality}, n_substeps={n_substeps_tmp}, horizon={horizon})"
            )

        # Taichi init
        ti.init(arch=ti.cuda, device_memory_GB=device_memory_GB)

        self.particle_density = particle_density
        self.dim = dim
        self.max_steps_local = max_steps_local
        self.max_steps_global = max_steps_global
        self.horizon = horizon
        self.ckpt_dest = ckpt_dest

        # env components
        self.agent = None
        self.statics = Statics()
        self.bodies = Bodies(dim=self.dim, particle_density=self.particle_density)

        self.simulator = MPMSimulatortactile(
            dim=self.dim,
            quality=quality,
            horizon=self.horizon,
            max_steps_local=self.max_steps_local,
            max_steps_global=self.max_steps_global,
            gravity=gravity,
            ckpt_dest=ckpt_dest,
        )

        self.renderer = None
        self.loss = None

        # NEW: store tactile cfg (optional)
        self._tactile_sensor_cfg = None
        if tactile_sensor_cfg is not None:
            self.setup_tactile_sensor(**tactile_sensor_cfg)

    # ===================== Agent / Renderer / Boundary / Statics / Bodies =====================

    def setup_agent(self, agent_cfg):
        self.agent = eval(agent_cfg.type)(
            max_steps_local=self.max_steps_local,
            max_steps_global=self.max_steps_global,
            max_action_steps_global=self.horizon,
            ckpt_dest=self.ckpt_dest,
            **agent_cfg.get('params', {}),
        )
        for effector_cfg in agent_cfg.effectors:
            self.agent.add_effector(
                type=effector_cfg.type,
                params=effector_cfg.params,
                mesh_cfg=effector_cfg.get('mesh', None),
                boundary_cfg=effector_cfg.boundary,
            )

    def setup_renderer(self, **kwargs):
        self.renderer = Renderer(**kwargs)

    def setup_boundary(self, **kwargs):
        self.simulator.setup_boundary(**kwargs)

    def add_static(self, **kwargs):
        self.statics.add_static(**kwargs)

    def add_body(self, **kwargs):
        self.bodies.add_body(**kwargs)

    def setup_loss(self, **kwargs):
        self.loss = Cuthalfloss(
            max_action_steps_global=self.horizon,
            max_steps_global=self.max_steps_global,
            **kwargs
        )

    # ===================== NEW: Tactile sensor wrappers =====================

    def setup_tactile_sensor(
        self,
        center,
        length,
        height,
        normal_axis=1,
        thickness=None,
        pulse_on_entry=True,
        one_sided=False,
        front_side="positive",
        record_per_particle=False,
    ):
        """
        Proxy to MPMSimulatortactile.setup_tactile_sensor(...)

        IMPORTANT: Call this BEFORE build(), because MPMSimulatortactile allocates
        tactile Taichi fields during build() (after particles are initialized).
        """
        self._tactile_sensor_cfg = dict(
            center=center,
            length=length,
            height=height,
            normal_axis=normal_axis,
            thickness=thickness,
            pulse_on_entry=pulse_on_entry,
            one_sided=one_sided,
            front_side=front_side,
            record_per_particle=record_per_particle,
        )
        self.simulator.setup_tactile_sensor(**self._tactile_sensor_cfg)

    @property
    def has_tactile_sensor(self):
        # Before build(): tactile_cfg may be set but has_tactile_sensor=False
        return (getattr(self.simulator, "tactile_cfg", None) is not None) or \
               bool(getattr(self.simulator, "has_tactile_sensor", False))

    def get_tactile(self, f=None):
        """
        Returns dict: {"contact": int, "pulse": float, "count": int}
        If tactile not enabled, returns zeros.
        """
        return self.simulator.get_tactile(f=f)

    def get_tactile_particles(self):
        """
        If record_per_particle=True: returns (contact_np, pulse_np) for latest sensed frame.
        Else returns None.
        """
        return self.simulator.get_tactile_particles()

    # ===================== Build =====================

    def build(self):
        # particles
        (
            self.init_particles,
            self.particles_material,
            self.particles_used,
            particles_color,
            self.particles_rho,
            self.particles_body_id
        ) = self.bodies.get()

        if self.init_particles is not None:
            self.n_particles = len(self.init_particles)

            # FIXED BUG: len(self.init_particles,) is always 1 due to trailing comma.
            self.particles_color = ti.Vector.field(4, ti.f32, shape=(self.n_particles,))
            self.particles_color.from_numpy(particles_color.astype(np.float32))

            self.has_particles = True
        else:
            self.n_particles = 0
            self.has_particles = False
            self.particles_color = None

        # build simulator (this is where tactile fields get allocated if tactile_cfg was set)
        self.simulator.build(
            agent=self.agent,
            statics=self.statics,
            x=self.init_particles,
            mat=self.particles_material,
            used=self.particles_used,
            p_rho=self.particles_rho,
            body_id=self.particles_body_id
        )

        if self.agent is not None:
            self.agent.build(self.simulator)

        if self.renderer is not None:
            self.renderer.build(self.n_particles)

        if self.loss is not None:
            self.loss.build(
                sim=self.simulator,
                knife=self.agent.effectors[0],
                ground=self.statics[1]
            )

    # ===================== Grad control =====================

    def reset_grad(self):
        self.simulator.reset_grad()
        if self.agent is not None:
            self.agent.reset_grad()
        if self.loss is not None:
            self.loss.reset_grad()

    def enable_grad(self):
        self.simulator.enable_grad()

    def disable_grad(self):
        self.simulator.disable_grad()

    @property
    def grad_enabled(self):
        return self.simulator.grad_enabled

    # ===================== Render / Step =====================

    def render(self, mode='human', iteration=None, t=None, save=False, **kwargs):
        assert self.renderer is not None, 'No renderer available.'

        if self.has_particles:
            frame_state = self.simulator.get_frame_state(self.simulator.cur_step_local)
        else:
            frame_state = None

        if self.loss is not None and iteration == 0:
            tgt_particles = self.loss.tgt_particles_x_f32
        else:
            tgt_particles = None

        img = self.renderer.render_frame(
            frame_state,
            self.particles_material,
            self.particles_color,
            self.agent,
            tgt_particles,
            self.statics,
            iteration,
            t,
            save
        )
        return img

    def step(self, action=None, return_tactile=False):
        """
        Same as before, but optionally returns tactile reading after stepping.
        """
        if action is not None:
            assert self.agent is not None, 'Environment has no agent to execute action.'
            action = np.array(action).astype(DTYPE_NP)

        self.simulator.step(action=action)

        if return_tactile:
            return self.get_tactile()

    def step_grad(self, action=None):
        if action is not None:
            assert self.agent is not None, 'Environment has no agent to execute action.'
            action = np.array(action).astype(DTYPE_NP)
        self.simulator.step_grad(action=action)

    # ===================== Loss =====================

    def get_loss(self):
        assert self.loss is not None
        return self.loss.get_loss(
            step_num=self.simulator.cur_step_global // self.simulator.n_substeps,
            substep_num=self.simulator.cur_step_global
        )

    def get_loss_grad(self):
        assert self.loss is not None
        return self.loss.get_loss_grad(
            step_num=self.simulator.cur_step_global // self.simulator.n_substeps,
            substep_num=self.simulator.cur_step_global
        )

    # ===================== State save/restore =====================

    def get_state(self):
        return {
            'state': self.simulator.get_state(),
            'grad_enabled': self.grad_enabled,
        }

    def set_state(self, state, grad_enabled=False):
        self.simulator.cur_step_global = 0
        self.simulator.set_state(0, state)

        if grad_enabled:
            self.enable_grad()
        else:
            self.disable_grad()

        if self.loss is not None:
            self.loss.clear()

    # ===================== Agent helpers =====================

    def apply_agent_action_p(self, action_p):
        assert self.agent is not None, 'Environment has no agent to execute action.'
        self.agent.apply_action_p(action_p)

    def apply_agent_action_p_grad(self, action_p):
        assert self.agent is not None, 'Environment has no agent to execute action.'
        self.agent.apply_action_p_grad(action_p)

    # ===================== Camera =====================

    def set_camera(self, position=None, lookat=None):
        if self.renderer is None:
            return
        if position is not None:
            self.renderer.camera.position(*position)
        if lookat is not None:
            self.renderer.camera.lookat(*lookat)
