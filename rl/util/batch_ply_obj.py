#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert mesh.ply -> mesh.obj for folders 0000..0300 (inclusive), only when:
  - mesh.ply exists
  - mesh.obj does NOT exist

Example:
  python convert_missing_ply_to_obj.py \
    --data_root /home/louhz/Desktop/Rss/roboninja/generateddata/Generated_strawberry \
    --start 0 --end 300
"""

import argparse
import re
import sys
from pathlib import Path


def iter_target_dirs(data_root: Path, start: int, end: int):
    """
    If data_root itself is a 4-digit folder (e.g., .../0007), process only it.
    Otherwise, treat data_root as a parent directory containing 0000..0300.
    """
    if re.fullmatch(r"\d{4}", data_root.name) and data_root.is_dir():
        yield data_root
        return

    for i in range(start, end + 1):
        d = data_root / f"{i:04d}"
        if d.is_dir():
            yield d


def load_mesh_any_backend(ply_path: Path):
    """
    Try trimesh first; fallback to open3d if trimesh isn't available.
    Returns an object that can be exported to OBJ, plus an export callable.
    """
    try:
        import trimesh  # type: ignore

        loaded = trimesh.load(str(ply_path), force="mesh")
        if isinstance(loaded, trimesh.Scene):
            # merge all geometries
            geoms = list(loaded.dump())
            if not geoms:
                raise ValueError(f"No geometry found in scene: {ply_path}")
            mesh = trimesh.util.concatenate(geoms)
        else:
            mesh = loaded

        def export_obj(obj_path: Path):
            mesh.export(str(obj_path), file_type="obj")

        return "trimesh", export_obj

    except Exception as e_trimesh:
        # Fallback: open3d
        try:
            import open3d as o3d  # type: ignore

            mesh = o3d.io.read_triangle_mesh(str(ply_path))
            if mesh.is_empty():
                raise ValueError(f"Empty mesh: {ply_path}")

            def export_obj(obj_path: Path):
                ok = o3d.io.write_triangle_mesh(str(obj_path), mesh, write_triangle_uvs=True)
                if not ok:
                    raise RuntimeError(f"open3d failed to write OBJ: {obj_path}")

            return "open3d", export_obj

        except Exception as e_o3d:
            raise RuntimeError(
                f"Failed to load/export mesh via trimesh ({e_trimesh}) and open3d ({e_o3d})."
            ) from e_o3d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Parent dir containing 0000..0300, or a single 4-digit folder.")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=300)
    ap.add_argument("--ply_name", default="mesh.ply")
    ap.add_argument("--obj_name", default="mesh.obj")
    ap.add_argument("--dry_run", action="store_true", help="Print what would be done, without writing files.")
    args = ap.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.exists():
        print(f"[ERROR] data_root does not exist: {data_root}", file=sys.stderr)
        sys.exit(2)

    converted = 0
    skipped_missing_ply = 0
    skipped_obj_exists = 0
    failed = 0

    for d in iter_target_dirs(data_root, args.start, args.end):
        ply_path = d / args.ply_name
        obj_path = d / args.obj_name

        if not ply_path.exists():
            skipped_missing_ply += 1
            continue
        if obj_path.exists():
            skipped_obj_exists += 1
            continue

        try:
            backend, export_obj = load_mesh_any_backend(ply_path)
            if args.dry_run:
                print(f"[DRY] would convert ({backend}): {ply_path} -> {obj_path}")
            else:
                export_obj(obj_path)
                print(f"[OK] converted ({backend}): {ply_path} -> {obj_path}")
            converted += 1
        except Exception as e:
            failed += 1
            print(f"[FAIL] {d}: {e}", file=sys.stderr)

    print("\n=== Summary ===")
    print(f"converted:          {converted}")
    print(f"skipped (no ply):   {skipped_missing_ply}")
    print(f"skipped (obj exists): {skipped_obj_exists}")
    print(f"failed:             {failed}")

    # Note: OBJ typically doesn't preserve PLY vertex colors.
    if converted > 0:
        print("\nNote: OBJ usually does NOT preserve PLY vertex colors; geometry should be preserved.")


if __name__ == "__main__":
    main()
