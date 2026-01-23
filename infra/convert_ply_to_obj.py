#!/usr/bin/env python3
"""convert_ply_to_obj.py

Recursively scan a root directory for numbered subfolders containing .ply meshes
and export each mesh to .obj.

Why this script exists:
  Your config (expert.yaml) points to `.../0001/mesh.obj`, but the generated
  dataset folders often contain `*.ply` meshes. This script converts them in
  place (or into a separate output root) so the rest of the pipeline can load
  `.obj` meshes.

Features:
  - Sorts numbered subfolders (e.g., 0001, 0002, ...)
  - Finds .ply files (prefers `mesh.ply` if present)
  - Handles trimesh "Scene" by concatenating geometries
  - Skips point clouds (PLY without faces) with a clear warning

Usage examples:
  # Convert in-place (writes mesh.obj next to the .ply)
  python convert_ply_to_obj.py \
    --input_root /home/louhz/Desktop/Rss/roboninja/generateddata/Generated_strawberry \
    --write_in_place

  # Convert into a separate output directory (keeps folder structure)
  python convert_ply_to_obj.py \
    --input_root /home/louhz/Desktop/Rss/roboninja/generateddata/Generated_strawberry \
    --output_root /home/louhz/Desktop/Rss/roboninja/generateddata/Generated_strawberry_obj
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Iterable, Optional, Tuple

import trimesh


NUMERIC_DIR_RE = re.compile(r"^\d+$")


def _is_numeric_dir(p: Path) -> bool:
    return p.is_dir() and bool(NUMERIC_DIR_RE.match(p.name))


def _sorted_numeric_dirs(root: Path) -> list[Path]:
    dirs = [p for p in root.iterdir() if _is_numeric_dir(p)]
    # Sort by integer value so 2 comes before 10
    return sorted(dirs, key=lambda p: int(p.name))


def _choose_ply_file(folder: Path) -> Optional[Path]:
    """Pick which PLY to convert inside a folder.

    Preference order:
      1) mesh.ply
      2) any single *.ply if only one exists
      3) otherwise first *.ply in sorted order
    """
    mesh_ply = folder / "mesh.ply"
    if mesh_ply.exists():
        return mesh_ply

    plys = sorted(folder.glob("*.ply"))
    if not plys:
        return None
    if len(plys) == 1:
        return plys[0]
    return plys[0]


def _load_as_trimesh(ply_path: Path) -> Optional[trimesh.Trimesh]:
    """Load a PLY file into a single trimesh.Trimesh.

    Returns None for point clouds or unsupported content.
    """
    obj = trimesh.load(ply_path, force=None)

    if isinstance(obj, trimesh.Trimesh):
        if obj.faces is None or len(obj.faces) == 0:
            # PLY point cloud or mesh without faces
            return None
        return obj

    if isinstance(obj, trimesh.Scene):
        # Concatenate all mesh geometries into one
        meshes = []
        for g in obj.geometry.values():
            if isinstance(g, trimesh.Trimesh) and g.faces is not None and len(g.faces) > 0:
                meshes.append(g)
        if not meshes:
            return None
        return trimesh.util.concatenate(meshes)

    # Could be PointCloud or something else
    return None


def convert_folder(
    folder: Path,
    output_folder: Path,
    output_name: str = "mesh.obj",
    overwrite: bool = False,
    verbose: bool = True,
) -> Tuple[bool, str]:
    """Convert one numbered folder.

    Returns:
      (ok, message)
    """
    ply_path = _choose_ply_file(folder)
    if ply_path is None:
        return False, f"No .ply found in {folder}"

    output_folder.mkdir(parents=True, exist_ok=True)
    out_path = output_folder / output_name

    if out_path.exists() and not overwrite:
        return True, f"Skip (exists): {out_path}"

    mesh = _load_as_trimesh(ply_path)
    if mesh is None:
        return False, f"Skip (not a triangle mesh / no faces): {ply_path}"

    # A couple of safe cleanups
    try:
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()
        mesh.remove_unreferenced_vertices()
    except Exception:
        # trimesh cleanup can fail on some meshes; still try exporting
        pass

    mesh.export(out_path)
    return True, f"Wrote: {out_path}  (from {ply_path.name})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_root",
        required=True,
        type=Path,
        help="Root folder containing numbered subfolders (0001, 0002, ...)",
    )
    ap.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help="If set, write converted .obj files under this root (mirrors subfolders).",
    )
    ap.add_argument(
        "--write_in_place",
        action="store_true",
        help="Write mesh.obj next to the .ply inside each numbered folder.",
    )
    ap.add_argument(
        "--output_name",
        default="mesh.obj",
        help="Output filename for each converted mesh (default: mesh.obj)",
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    input_root: Path = args.input_root
    if not input_root.exists():
        raise SystemExit(f"Input root does not exist: {input_root}")

    if args.write_in_place and args.output_root is not None:
        raise SystemExit("Use either --write_in_place OR --output_root, not both")
    if (not args.write_in_place) and (args.output_root is None):
        raise SystemExit("Specify --write_in_place or --output_root")

    numeric_dirs = _sorted_numeric_dirs(input_root)
    if not numeric_dirs:
        raise SystemExit(f"No numbered subfolders found under: {input_root}")

    ok_cnt = 0
    fail_cnt = 0
    for d in numeric_dirs:
        out_dir = d if args.write_in_place else (args.output_root / d.name)
        ok, msg = convert_folder(
            folder=d,
            output_folder=out_dir,
            output_name=args.output_name,
            overwrite=args.overwrite,
            verbose=args.verbose,
        )
        if ok:
            ok_cnt += 1
        else:
            fail_cnt += 1
        if args.verbose:
            print(msg)

    print(f"Done. success={ok_cnt}  failed/skipped={fail_cnt}")


if __name__ == "__main__":
    main()
