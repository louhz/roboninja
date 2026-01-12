import os
import copy
import trimesh
import numpy as np
import pickle as pkl
from cut_simulation.utils.misc import *
from cut_simulation.utils.mesh import *
from cut_simulation.configs.macros import *
from scipy.spatial.transform import Rotation







class FilledVoxelGrid:
    """
    Simple filled voxel grid with a method compatible with your add_mesh usage:
        voxels.is_filled((particles - pos) / scale)
    where input points are in normalized coords roughly in [-0.5, 0.5]^3.
    """
    def __init__(self, filled: np.ndarray):
        assert filled.ndim == 3 and filled.shape[0] == filled.shape[1] == filled.shape[2]
        self.filled = filled.astype(np.bool_)
        self.res = int(filled.shape[0])

    def is_filled(self, pts_norm: np.ndarray) -> np.ndarray:
        pts_norm = np.asarray(pts_norm, dtype=np.float32)
        if pts_norm.ndim != 2 or pts_norm.shape[1] != 3:
            raise ValueError("pts_norm must be (N,3)")

        valid = np.all((pts_norm >= -0.5) & (pts_norm <= 0.5), axis=1)

        # map [-0.5,0.5] -> [0,res)
        idx = np.floor((pts_norm + 0.5) * self.res).astype(np.int32)
        idx = np.clip(idx, 0, self.res - 1)

        out = np.zeros((pts_norm.shape[0],), dtype=np.bool_)
        idxv = idx[valid]
        out[valid] = self.filled[idxv[:, 0], idxv[:, 1], idxv[:, 2]]
        return out



def pointcloud_to_filled_voxels(
    xyz_norm: np.ndarray,
    res: int = 256,
    shell_radius_vox: int = 2,
    close_iters: int = 2,
) -> np.ndarray:
    """
    xyz_norm: (N,3) in normalized coords, ideally inside [-0.5,0.5]^3
    res: voxel resolution
    shell_radius_vox: dilate radius to thicken surface so it becomes watertight
    close_iters: binary closing iterations to seal small gaps
    Returns: filled (res,res,res) bool
    """
    from scipy.ndimage import binary_dilation, binary_closing, binary_fill_holes

    grid = np.zeros((res, res, res), dtype=np.bool_)

    # voxel indices
    idx = np.floor((xyz_norm + 0.5) * res).astype(np.int32)
    valid = np.all((idx >= 0) & (idx < res), axis=1)
    idx = idx[valid]
    if idx.shape[0] == 0:
        raise ValueError("No points fell inside voxel grid; check normalization/scale.")

    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    if shell_radius_vox > 0:
        k = 2 * shell_radius_vox + 1
        structure = np.ones((k, k, k), dtype=np.bool_)
        grid = binary_dilation(grid, structure=structure, iterations=1)

    if close_iters > 0:
        structure = np.ones((3, 3, 3), dtype=np.bool_)
        grid = binary_closing(grid, structure=structure, iterations=close_iters)

    # fill interior holes -> solid
    filled = binary_fill_holes(grid)
    return filled.astype(np.bool_)



class Bodies:
    # Class for fluid / rigid bodies represented by MPM particles
    def __init__(self, dim, particle_density):
        self.bodies = []
        self.materials = []
        self.used = []
        self.colors = []
        self.rhos = []
        self.body_ids = []
        self.dim = dim
        self.particle_density = particle_density

    def __len__(self):
        return len(self.bodies)

    def add_body(self, type, filling='random', **kwargs):
        state = np.random.get_state()
        np.random.seed(0) #fix seed 0

        assert filling in ['random', 'grid', 'natural'], f'Unsupported filling type: {filling}.'
        if type == 'nowhere':
            self.add_nowhere(**kwargs)
        elif type == 'cube':
            self.add_cube(filling=filling, **kwargs)
        elif type == 'cylinder':
            self.add_cylinder(filling=filling, **kwargs)
        elif type == 'ball':
            self.add_ball(filling=filling, **kwargs)
        elif type == 'mesh':
            self.add_mesh(filling=filling, **kwargs)

        elif type in ['splat', 'strawberry', 'pointcloud']:
            self.add_splat(filling=filling, **kwargs)       

        else:
            raise NotImplementedError(f'Unsupported body type: {type}.')

        np.random.set_state(state)

    def compute_n_particles(self, volume):
        return round(volume * self.particle_density)

    def compute_n_particles_1D(self, length):
        return round(length * np.cbrt(self.particle_density))

    def _add_body(
        self,
        type,
        particles,
        material,
        color=None,
        used=False,
        euler=(0.0, 0.0, 0.0),
        obstacles=list()
    ):

        if color is not None:
            body_color = color
        else:
            body_color = COLOR[material]

        
        # rotate
        R = Rotation.from_euler('zyx', np.array(euler)[::-1], degrees=True).as_matrix()
        particles_COM = particles.mean(0)
        particles = (R @ (particles - particles_COM).T).T + particles_COM

        # avoid obstacle
        for obstacle in obstacles:
            particle_sdf = np.zeros([len(particles)])
            obstacle.check_collision(len(particles), particles, particle_sdf)
            particles = particles[particle_sdf >= 0]        

        orders = np.argsort(particles[:, 2])
        particles = particles[orders]

        body_color = np.tile(body_color, [len(particles), 1])
        body_rho = np.full(len(particles), RHO[material] )
        body_material = np.full(len(particles), material)
        body_used = np.full(len(particles), used)
        body_id = np.full(len(particles), len(self.bodies))


        self.colors.append(body_color)
        self.rhos.append(body_rho)
        self.materials.append(body_material)
        self.used.append(body_used)
        self.body_ids.append(body_id)
        self.bodies.append(particles)

        print(f'===>  {len(particles):7d} particles of {MAT_NAME[material]:>8} {type:>8} added.')

    def sample_cube(self, lower, upper, filling):
        size = upper - lower
        if filling == 'random':
            volume = np.prod(size)
            n_particles = self.compute_n_particles(volume)
            particles = np.random.uniform(low=lower, high=upper, size=(n_particles, self.dim))
        elif filling == 'grid':
            n_x = self.compute_n_particles_1D(size[0])
            n_y = self.compute_n_particles_1D(size[1])
            n_z = self.compute_n_particles_1D(size[2])
            dx = size[0] / n_x
            dy = size[1] / n_y
            dz = size[2] / n_z
            x = np.linspace(lower[0], upper[0], n_x+1)
            y = np.linspace(lower[1], upper[1], n_y+1)
            z = np.linspace(lower[2], upper[2], n_z+1)
            particles = np.stack(np.meshgrid(x, y, z, indexing='ij'), -1).reshape((-1, 3))
        else:
            raise NotImplementedError(f'Unsupported filling type: {filling}.')

        return particles

    def add_nowhere(self, n_particles, **kwargs):
        particles = np.tile(np.array(NOWHERE), (n_particles, 1))
        self._add_body('nowhere', particles, used=False, **kwargs)

    def add_cube(self, lower, filling, upper=None, size=None, **kwargs):
        lower = np.array(lower)
        if size is not None:
            upper = lower + np.array(size)
        else:
            upper = np.array(upper)
        assert (upper >= lower).all()

        if filling == 'natural':
            filling = 'grid' # for cube, natural is the same as grid

        particles = self.sample_cube(lower, upper, filling)

        self._add_body('cube', particles, used=True, **kwargs)

    def add_cylinder(self, center, height, radius, filling, **kwargs):
        radius = np.array(radius)
        center = np.array(center)

        if filling == 'natural':
            n_y = self.compute_n_particles_1D(height)
            n_r = self.compute_n_particles_1D(radius)
            particles = []
            for y_layer in np.linspace(center[1]-height/2, center[1]+height/2, n_y+1):
                for r_layer in np.linspace(0, radius, n_r+1):
                    n_layer = max(self.compute_n_particles_1D(2*np.pi*r_layer), 1)
                    rad_layer = np.linspace(0, np.pi*2, n_layer+1)[:-1]
                    x_layer = np.cos(rad_layer) * r_layer + center[0]
                    z_layer = np.sin(rad_layer) * r_layer + center[2]
                    particles_layer = np.vstack([x_layer, np.repeat(y_layer, n_layer), z_layer])
                    particles.append(particles_layer)
            particles = np.hstack(particles).T
        else: 
            # sample a cube first
            cube_lower = np.array([center[0] - radius, center[1] - height / 2.0, center[2] - radius])
            cube_upper = np.array([center[0] + radius, center[1] + height / 2.0, center[2] + radius])
            particles = self.sample_cube(cube_lower, cube_upper, filling)

            # reject out-of-boundary particles
            particles_r = np.linalg.norm(particles[:, [0, 2]] - center[[0, 2]], axis=1)
            particles = particles[particles_r <= radius]

        self._add_body('cylinder', particles, used=True, **kwargs)

    def add_ball(self, center, radius, filling, **kwargs):
        center = np.array(center)

        if filling == 'natural':
            n_r = self.compute_n_particles_1D(radius)
            particles = []
            for r_sphere in np.linspace(0, radius, n_r+1):
                n_layers = self.compute_n_particles_1D(r_sphere*np.pi)
                for ver_rad_layer in np.linspace(-np.pi/2, np.pi/2, n_layers+1):
                    y_layer = center[1] + np.sin(ver_rad_layer) * r_sphere
                    r_layer = np.sqrt(max(r_sphere**2-(center[1]-y_layer)**2, 0))
                    n_particles_layer = max(self.compute_n_particles_1D(2*np.pi*r_layer), 1)
                    hor_rad_layer = np.linspace(0, np.pi*2, n_particles_layer+1)[:-1]
                    x_layer = np.cos(hor_rad_layer) * r_layer + center[0]
                    z_layer = np.sin(hor_rad_layer) * r_layer + center[2]
                    particles_layer = np.vstack([x_layer, np.repeat(y_layer, n_particles_layer), z_layer])
                    particles.append(particles_layer)
            particles = np.hstack(particles).T
        else: 
            # sample a cube first
            cube_lower = center - radius
            cube_upper = center + radius
            particles = self.sample_cube(cube_lower, cube_upper, filling)

            # reject out-of-boundary particles
            particles_r = np.linalg.norm(particles - center, axis=1)
            particles = particles[particles_r <= radius]

        self._add_body('ball', particles, used=True, **kwargs)

    def add_mesh(self, file, filling, pos=(0.5, 0.5, 0.5), scale=(1.0, 1.0, 1.0), voxelize_res=128, **kwargs):
        assert filling != 'natural', 'natural filling not supported for body type: mesh.'

        raw_file_path = get_raw_mesh_path(file)
        voxelized_file_path = get_voxelized_mesh_path(file, voxelize_res)

        if not os.path.exists(voxelized_file_path):
            print(f'===> Voxelizing mesh {raw_file_path}.')
            voxelized_mesh = voxelize_mesh(raw_file_path, voxelize_res)
            pkl.dump(voxelized_mesh, open(voxelized_file_path, 'wb'))
            print(f'===> Voxelized mesh saved as {voxelized_file_path}.')

        voxels = pkl.load(open(voxelized_file_path, 'rb'))

        # sample a cube first
        scale = np.array(scale)
        pos = np.array(pos)
        cube_lower = pos - scale * 0.5
        cube_upper = pos + scale * 0.5
        particles = self.sample_cube(cube_lower, cube_upper, filling)

        # reject out-of-boundary particles
        particles = particles[voxels.is_filled((particles - pos) / scale)]
        self._add_body('mesh', particles, used=True, **kwargs)

    def get(self):
        if len(self.bodies) == 0:
            return [None] * 6
        else:
            return (
                np.concatenate(self.bodies),
                np.concatenate(self.materials),
                np.concatenate(self.used),
                np.concatenate(self.colors),
                np.concatenate(self.rhos),
                np.concatenate(self.body_ids),
            )



    def add_splat(
        self,
        file,
        filling,
        pos=(0.5, 0.12, 0.5),
        scale=(0.18, 0.18, 0.18),
        voxelize_res=256,
        shell_radius_vox=2,
        close_iters=2,
        normalize=True,
        trim_percentile=0.5,
        cache_voxels=True,
        **kwargs
    ):
        """
        Create a solid body from Gaussian Splatting centers (xyz) by voxel-filling the point cloud.

        - file: path to splat point cloud (usually point_cloud.ply)
        - pos/scale: final placement in sim coordinates (same convention as add_mesh)
        - normalize: if True, normalize the splat point cloud to fit inside [-0.5,0.5]^3
        - trim_percentile: removes outliers by axis percentiles (helps if splat has floaters)
        """
        assert filling in ["random", "grid"], "natural filling not supported for splat."

        raw_xyz = load_splat_centers_xyz(file)

        # Optional outlier trimming (helps splat floaters)
        if trim_percentile is not None and trim_percentile > 0.0:
            lo = np.percentile(raw_xyz, trim_percentile, axis=0)
            hi = np.percentile(raw_xyz, 100.0 - trim_percentile, axis=0)
            keep = np.all((raw_xyz >= lo) & (raw_xyz <= hi), axis=1)
            raw_xyz = raw_xyz[keep]
            if raw_xyz.shape[0] == 0:
                raise ValueError("All points removed by trimming; reduce trim_percentile.")

        # Normalize to [-0.5, 0.5] cube (isotropic scale preserves aspect ratio)
        if normalize:
            mn = raw_xyz.min(axis=0)
            mx = raw_xyz.max(axis=0)
            center = (mn + mx) * 0.5
            extent = (mx - mn)
            s = float(np.max(extent))
            if s < 1e-9:
                raise ValueError("Degenerate point cloud (near-zero extent).")
            xyz_norm = (raw_xyz - center) / s  # about within [-0.5,0.5]
        else:
            xyz_norm = raw_xyz.astype(np.float32)

        # Cache file for filled voxels
        scale = np.array(scale, dtype=np.float32)
        pos = np.array(pos, dtype=np.float32)

        vox_cache_path = os.path.splitext(file)[0] + f"_filledvox_r{voxelize_res}_sr{shell_radius_vox}_c{close_iters}.pkl"

        if cache_voxels and os.path.exists(vox_cache_path):
            filled = pkl.load(open(vox_cache_path, "rb"))
        else:
            filled = pointcloud_to_filled_voxels(
                xyz_norm,
                res=voxelize_res,
                shell_radius_vox=shell_radius_vox,
                close_iters=close_iters,
            )
            if cache_voxels:
                pkl.dump(filled, open(vox_cache_path, "wb"))

        vox = FilledVoxelGrid(filled)

        # Estimate volume from filled voxel fraction -> decide particle count
        filled_count = int(filled.sum())
        if filled_count == 0:
            raise ValueError("Voxel fill produced empty volume; increase shell_radius_vox/close_iters.")

        vol_est = (filled_count / float(voxelize_res ** 3)) * float(np.prod(scale))
        n_particles = self.compute_n_particles(vol_est)
        n_particles = max(n_particles, 1)

        filled_idx = np.argwhere(filled)  # (M,3)

        if filling == "grid":
            # Use voxel centers (can be very dense if res is high!)
            pts_norm = (filled_idx.astype(np.float32) + 0.5) / voxelize_res - 0.5
            particles = pts_norm * scale + pos
        else:
            # Random sample voxels, jitter inside each voxel
            choice = np.random.randint(0, filled_idx.shape[0], size=n_particles)
            vox_sel = filled_idx[choice].astype(np.float32)
            jitter = np.random.rand(n_particles, 3).astype(np.float32)  # [0,1)
            pts_norm = (vox_sel + jitter) / voxelize_res - 0.5
            particles = pts_norm * scale + pos

        # (Optional) you can still filter with vox.is_filled, but here it should all be filled already.
        # particles = particles[vox.is_filled((particles - pos) / scale)]

        self._add_body("splat", particles, used=True, **kwargs)