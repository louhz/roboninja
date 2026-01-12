#!/usr/bin/env python3
"""
mesh_to_mpm_particles.py

Generate a particle (material point) representation for MPM from an input mesh.

Main workflow (volumetric mesh):
  - Read mesh using meshio (supports .msh, .vtk, .vtu, .xdmf, etc.)
  - For each cell (tet/hex/wedge/pyramid), seed `ppc` particles uniformly in the cell
  - Assign each particle a volume = cell_volume / ppc and mass = density * volume
  - Write particles to .npz (default) or .csv, and optionally a VTK point cloud for preview

If the input mesh is *surface-only* (tri/quad) and is watertight, you can enable filling
the interior using trimesh (optional dependency).

Dependencies:
  pip install numpy meshio
Optional (surface filling):
  pip install trimesh rtree

Outputs:
  NPZ: arrays `x` (N,3), `v` (N,3), `mass` (N,), `volume` (N,), `material_id` (N,)
  CSV: columns x,y,z,vx,vy,vz,mass,volume,material_id
  VTK point cloud: .vtp/.vtu via meshio

Notes for MPM users:
  - Choose ppc ~ 1..64 depending on accuracy vs. cost.
  - Ensure mesh units and density units are consistent (e.g., meters + kg/m^3).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import meshio  # type: ignore
except Exception as e:
    print("ERROR: meshio is required. Install with: pip install meshio", file=sys.stderr)
    raise


# ----------------------------
# Geometry helpers
# ----------------------------

def _as_3d(points: np.ndarray) -> np.ndarray:
    """Ensure points are (N,3). If (N,2), append zeros."""
    if points.ndim != 2:
        raise ValueError("mesh points must be a 2D array")
    if points.shape[1] == 3:
        return points.astype(np.float64, copy=False)
    if points.shape[1] == 2:
        p = np.zeros((points.shape[0], 3), dtype=np.float64)
        p[:, :2] = points
        return p
    raise ValueError(f"Unsupported point dimension: {points.shape[1]} (expected 2 or 3)")


def tet_volume(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    """Absolute volume of tetrahedron (a,b,c,d)."""
    v = np.linalg.det(np.stack([b - a, c - a, d - a], axis=1)) / 6.0
    return float(abs(v))


def sample_tet_uniform(verts4: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Uniformly sample n points in a tetrahedron using Dirichlet(1,1,1,1) barycentric weights.
    verts4: (4,3)
    returns: (n,3)
    """
    u = rng.random((n, 4))
    e = -np.log(u)
    w = e / e.sum(axis=1, keepdims=True)
    return (w[:, 0:1] * verts4[0]
            + w[:, 1:2] * verts4[1]
            + w[:, 2:3] * verts4[2]
            + w[:, 3:4] * verts4[3])


# ----------------------------
# Cell decompositions
# ----------------------------

def tet_decomposition_for_cell(cell_type: str) -> Optional[List[Tuple[int, int, int, int]]]:
    """
    Return a list of sub-tetrahedra (as local vertex indices) that decompose the cell.
    Used for both volume estimation and uniform sampling.

    Supported cell types: tetra, hexahedron, wedge, pyramid
    """
    if cell_type == "tetra":
        return [(0, 1, 2, 3)]

    if cell_type == "hexahedron":
        # VTK/meshio ordering for hex:
        # 0-3 bottom face, 4-7 top face
        # Common 5-tet decomposition.
        return [
            (0, 1, 3, 4),
            (1, 2, 3, 6),
            (1, 3, 4, 6),
            (1, 4, 5, 6),
            (3, 4, 6, 7),
        ]

    if cell_type == "wedge":
        # 6-node triangular prism: bottom triangle (0,1,2), top triangle (3,4,5)
        return [
            (0, 1, 2, 3),
            (1, 2, 4, 3),
            (2, 4, 5, 3),
        ]

    if cell_type == "pyramid":
        # 5-node pyramid: base quad (0,1,2,3), apex 4
        return [
            (0, 1, 2, 4),
            (0, 2, 3, 4),
        ]

    return None


# ----------------------------
# Particle generation
# ----------------------------

@dataclass
class ParticleCloud:
    x: np.ndarray            # (N,3)
    v: np.ndarray            # (N,3)
    mass: np.ndarray         # (N,)
    volume: np.ndarray       # (N,)
    material_id: np.ndarray  # (N,)


def generate_particles_from_volume_mesh(
    mesh: "meshio.Mesh",
    density: float,
    ppc: int,
    velocity: np.ndarray,
    default_material_id: int,
    cell_material_field: Optional[str],
    seed: int,
) -> ParticleCloud:
    if ppc <= 0:
        raise ValueError("--ppc must be positive")

    rng = np.random.default_rng(seed)
    pts = _as_3d(np.asarray(mesh.points))

    X_list: List[np.ndarray] = []
    V_list: List[np.ndarray] = []
    mass_list: List[np.ndarray] = []
    vol_list: List[np.ndarray] = []
    mat_list: List[np.ndarray] = []

    per_name = None
    if cell_material_field is not None:
        per_name = mesh.cell_data.get(cell_material_field)
        if per_name is None:
            available = ", ".join(mesh.cell_data.keys()) if mesh.cell_data else "(none)"
            raise ValueError(
                f"Requested cell material field '{cell_material_field}' not found. "
                f"Available cell_data fields: {available}"
            )

    total_cells_used = 0
    total_cells_skipped = 0

    for block_idx, block in enumerate(mesh.cells):
        ctype = block.type
        decomp = tet_decomposition_for_cell(ctype)
        if decomp is None:
            continue  # ignore non-volumetric blocks

        cells = np.asarray(block.data, dtype=np.int64)
        if cells.size == 0:
            continue

        if per_name is not None:
            mats_block = np.asarray(per_name[block_idx]).astype(np.int64, copy=False)
            if mats_block.shape[0] != cells.shape[0]:
                raise ValueError(
                    f"cell_data['{cell_material_field}'] block {block_idx} length mismatch: "
                    f"{mats_block.shape[0]} vs number of {ctype} cells {cells.shape[0]}"
                )
        else:
            mats_block = None

        for ei, conn in enumerate(cells):
            verts = pts[conn]  # (n_nodes, 3)

            sub_vols: List[float] = []
            sub_verts: List[np.ndarray] = []

            for (i0, i1, i2, i3) in decomp:
                v4 = np.stack([verts[i0], verts[i1], verts[i2], verts[i3]], axis=0)
                vol = tet_volume(v4[0], v4[1], v4[2], v4[3])
                if vol > 0.0:
                    sub_vols.append(vol)
                    sub_verts.append(v4)

            cell_vol = float(sum(sub_vols))
            if not (cell_vol > 0.0):
                total_cells_skipped += 1
                continue

            probs = np.asarray(sub_vols, dtype=np.float64) / cell_vol
            picks = rng.choice(len(sub_verts), size=ppc, p=probs)

            X_cell = np.empty((ppc, 3), dtype=np.float64)
            for si in range(len(sub_verts)):
                mask = (picks == si)
                k = int(mask.sum())
                if k == 0:
                    continue
                X_cell[mask] = sample_tet_uniform(sub_verts[si], k, rng)

            vol_p = cell_vol / float(ppc)
            vol_arr = np.full(ppc, vol_p, dtype=np.float64)
            mass_arr = vol_arr * float(density)

            if mats_block is None:
                mat_arr = np.full(ppc, int(default_material_id), dtype=np.int64)
            else:
                mat_arr = np.full(ppc, int(mats_block[ei]), dtype=np.int64)

            V_cell = np.tile(velocity[None, :], (ppc, 1))

            X_list.append(X_cell)
            V_list.append(V_cell)
            vol_list.append(vol_arr)
            mass_list.append(mass_arr)
            mat_list.append(mat_arr)

            total_cells_used += 1

    if total_cells_used == 0:
        raise ValueError(
            "No volumetric cells (tetra/hexahedron/wedge/pyramid) were found in the mesh.\n"
            "If your mesh is surface-only (tri/quad), use --fill-surface to voxel-fill a watertight mesh."
        )

    X = np.concatenate(X_list, axis=0)
    V = np.concatenate(V_list, axis=0)
    volume = np.concatenate(vol_list, axis=0)
    mass = np.concatenate(mass_list, axis=0)
    material_id = np.concatenate(mat_list, axis=0)

    print(f"[mesh_to_mpm_particles] used {total_cells_used} volume cells, skipped {total_cells_skipped} degenerate cells")
    print(f"[mesh_to_mpm_particles] generated {X.shape[0]} particles")

    return ParticleCloud(x=X, v=V, mass=mass, volume=volume, material_id=material_id)


def generate_particles_from_surface_mesh_by_voxels(
    mesh_path: str,
    density: float,
    spacing: float,
    velocity: np.ndarray,
    material_id: int,
) -> ParticleCloud:
    """
    Fill a *watertight* surface mesh with a regular voxel-grid of particles.
    Requires trimesh (+ rtree recommended).
    """
    try:
        import trimesh  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Surface filling requires trimesh. Install with: pip install trimesh rtree"
        ) from e

    surf = trimesh.load(mesh_path, force="mesh")
    if surf.is_empty:
        raise ValueError("Surface mesh is empty / could not be loaded.")
    if not surf.is_watertight:
        raise ValueError("Surface mesh is not watertight; cannot reliably fill its interior for volume particles.")

    bounds = np.asarray(surf.bounds, dtype=np.float64)
    lo, hi = bounds[0], bounds[1]

    nx = max(1, int(math.floor((hi[0] - lo[0]) / spacing)))
    ny = max(1, int(math.floor((hi[1] - lo[1]) / spacing)))
    nz = max(1, int(math.floor((hi[2] - lo[2]) / spacing)))

    xs = lo[0] + (np.arange(nx) + 0.5) * spacing
    ys = lo[1] + (np.arange(ny) + 0.5) * spacing
    zs = lo[2] + (np.arange(nz) + 0.5) * spacing

    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)

    inside = surf.contains(grid)
    X = grid[inside]

    if X.shape[0] == 0:
        raise ValueError("No interior voxel centers found. Try a smaller --spacing or check mesh scale/units.")

    vol_p = float(spacing ** 3)
    volume = np.full(X.shape[0], vol_p, dtype=np.float64)
    mass = volume * float(density)
    V = np.tile(velocity[None, :], (X.shape[0], 1))
    mat = np.full(X.shape[0], int(material_id), dtype=np.int64)

    print(f"[mesh_to_mpm_particles] voxel-filled {X.shape[0]} particles (spacing={spacing})")

    return ParticleCloud(x=X, v=V, mass=mass, volume=volume, material_id=mat)


# ----------------------------
# I/O
# ----------------------------

def write_npz(path: str, cloud: ParticleCloud) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(
        path,
        x=cloud.x,
        v=cloud.v,
        mass=cloud.mass,
        volume=cloud.volume,
        material_id=cloud.material_id,
    )
    print(f"[mesh_to_mpm_particles] wrote NPZ: {path}")


def write_csv(path: str, cloud: ParticleCloud) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "z", "vx", "vy", "vz", "mass", "volume", "material_id"])
        for i in range(cloud.x.shape[0]):
            x = cloud.x[i]
            v = cloud.v[i]
            w.writerow([x[0], x[1], x[2], v[0], v[1], v[2], cloud.mass[i], cloud.volume[i], int(cloud.material_id[i])])
    print(f"[mesh_to_mpm_particles] wrote CSV: {path}")


def write_vtk_point_cloud(path: str, cloud: ParticleCloud) -> None:
    """Write a point cloud viewable in ParaView (.vtp recommended)."""
    points = cloud.x.astype(np.float64, copy=False)
    cells = [("vertex", np.arange(points.shape[0], dtype=np.int64).reshape(-1, 1))]
    m = meshio.Mesh(
        points=points,
        cells=cells,
        point_data={
            "vx": cloud.v[:, 0],
            "vy": cloud.v[:, 1],
            "vz": cloud.v[:, 2],
            "mass": cloud.mass,
            "volume": cloud.volume,
            "material_id": cloud.material_id.astype(np.int32),
        },
    )
    meshio.write(path, m)
    print(f"[mesh_to_mpm_particles] wrote VTK point cloud: {path}")


# ----------------------------
# CLI
# ----------------------------

def parse_vec3(s: str) -> np.ndarray:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected 3 comma-separated values, e.g. '0,0,0'")
    try:
        v = np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float64)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Could not parse vector: {s}") from e
    return v


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate MPM particles from a mesh.")
    ap.add_argument("--mesh", required=True, help="Input mesh file (.msh/.vtk/.vtu/.obj/.stl/...).")
    ap.add_argument("--out", required=True, help="Output file: .npz or .csv (and optionally .vtp via --vtk-out).")
    ap.add_argument("--density", type=float, default=1000.0, help="Material density (kg/m^3 if mesh units are meters).")
    ap.add_argument("--ppc", type=int, default=8, help="Particles per volumetric cell (ignored for voxel fill).")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    ap.add_argument("--material-id", type=int, default=0, help="Default material id if not reading per-cell ids.")
    ap.add_argument(
        "--cell-material-field",
        type=str,
        default=None,
        help="Optional cell_data field name to use as per-cell material id (e.g. 'gmsh:physical').",
    )
    ap.add_argument(
        "--velocity",
        type=parse_vec3,
        default=parse_vec3("0,0,0"),
        help="Initial particle velocity as 'vx,vy,vz'.",
    )

    # Surface fill options
    ap.add_argument(
        "--fill-surface",
        action="store_true",
        help="If mesh is surface-only, attempt to voxel-fill its interior (requires trimesh).",
    )
    ap.add_argument(
        "--spacing",
        type=float,
        default=None,
        help="Voxel spacing for --fill-surface (same units as mesh). Example: 0.01",
    )

    ap.add_argument("--vtk-out", type=str, default=None, help="Optional point cloud output (.vtp/.vtu) for visualization.")

    args = ap.parse_args(argv)

    mesh_path = args.mesh
    out_path = args.out

    try:
        mesh = meshio.read(mesh_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read mesh '{mesh_path}'. meshio error: {e}") from e

    has_volume = any(tet_decomposition_for_cell(cb.type) is not None for cb in mesh.cells)

    if has_volume:
        cloud = generate_particles_from_volume_mesh(
            mesh=mesh,
            density=float(args.density),
            ppc=int(args.ppc),
            velocity=np.asarray(args.velocity, dtype=np.float64),
            default_material_id=int(args.material_id),
            cell_material_field=args.cell_material_field,
            seed=int(args.seed),
        )
    else:
        if not args.fill_surface:
            raise ValueError(
                "Mesh appears to be surface-only (no tetra/hexa/wedge/pyramid cells).\n"
                "Either provide a volumetric mesh, or re-run with --fill-surface --spacing <h> "
                "to voxel-fill a watertight surface mesh."
            )
        if args.spacing is None or not (args.spacing > 0.0):
            raise ValueError("--fill-surface requires --spacing > 0")
        cloud = generate_particles_from_surface_mesh_by_voxels(
            mesh_path=mesh_path,
            density=float(args.density),
            spacing=float(args.spacing),
            velocity=np.asarray(args.velocity, dtype=np.float64),
            material_id=int(args.material_id),
        )

    ext = os.path.splitext(out_path)[1].lower()
    if ext == ".npz":
        write_npz(out_path, cloud)
    elif ext == ".csv":
        write_csv(out_path, cloud)
    else:
        raise ValueError("Unsupported --out extension. Use .npz or .csv")

    if args.vtk_out:
        write_vtk_point_cloud(args.vtk_out, cloud)

    print(f"[mesh_to_mpm_particles] total mass   = {cloud.mass.sum():.6g}")
    print(f"[mesh_to_mpm_particles] total volume = {cloud.volume.sum():.6g}")
    print(f"[mesh_to_mpm_particles] bbox min     = {cloud.x.min(axis=0)}")
    print(f"[mesh_to_mpm_particles] bbox max     = {cloud.x.max(axis=0)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
