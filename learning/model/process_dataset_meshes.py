#!/usr/bin/env python3
"""
Process dataset episodes:
- normalize mesh vertices (as point cloud)
- (optional) convert coordinates to right-handed/world frame (axis reorder + sign flip)
- apply per-episode scale (vec3) + translation offset (vec3)
- convert mesh to STL, replacing original

Requires:
  pip install numpy trimesh
Optional (only if using --config):
  pip install pyyaml

YAML list mapping:
  episodes_1 -> scales[0], offsets[0]
  episodes_2 -> scales[1], offsets[1]
  ...
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import trimesh
except ImportError as e:
    raise SystemExit(
        "Missing dependency 'trimesh'. Install with:\n"
        "  pip install trimesh numpy\n"
    ) from e


EP_RE = re.compile(r"^episodes_(\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class EpisodeDir:
    episode_id: int
    path: Path


@dataclass
class TransformConfig:
    normalize: str                    # none | center | bbox
    scale: np.ndarray                 # vec3 scale [sx,sy,sz]
    offset: np.ndarray                # translation vector (3,)
    axis_order: Tuple[int, int, int]  # reorder indices for xyz
    axis_signs: np.ndarray            # signs for xyz: [+1/-1, +1/-1, +1/-1]


@dataclass
class RunConfig:
    parent: Path
    mesh_subdir: str
    dry_run: bool
    backup: bool
    report_csv: Optional[Path]

    # global transform defaults (used if per-episode lists don't exist)
    normalize: str
    axis_order: Tuple[int, int, int]
    axis_signs: np.ndarray
    default_scale: np.ndarray         # vec3
    default_offset: np.ndarray

    # per-episode lists (index = episode_id - 1)
    scales: Optional[List[np.ndarray]]    # each vec3
    offsets: Optional[List[np.ndarray]]


def _coerce_vec3(x: Any, name: str) -> np.ndarray:
    if not isinstance(x, (list, tuple, np.ndarray)) or len(x) != 3:
        raise ValueError(f"Invalid {name}: expected a length-3 list, got {x!r}")
    try:
        arr = np.array([float(x[0]), float(x[1]), float(x[2])], dtype=np.float64)
    except Exception as e:
        raise ValueError(f"Invalid {name}: expected 3 numbers, got {x!r}") from e
    return arr


def _coerce_scale3(x: Any, name: str) -> np.ndarray:
    """
    Accept:
      - scalar -> expanded to [s,s,s]
      - vec3   -> [sx,sy,sz]
    """
    if isinstance(x, (int, float, np.number)):
        s = float(x)
        return np.array([s, s, s], dtype=np.float64)

    if isinstance(x, (list, tuple, np.ndarray)):
        if len(x) == 3 and all(isinstance(v, (int, float, np.number)) for v in x):
            return np.array([float(x[0]), float(x[1]), float(x[2])], dtype=np.float64)

    raise ValueError(f"Invalid {name}: expected a number or length-3 list, got {x!r}")


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "Missing dependency 'pyyaml' for --config.\n"
            "Install with:\n  pip install pyyaml\n"
        ) from e

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping/dict, got: {type(data)}")
    return data


def iter_episode_dirs(parent: Path) -> List[EpisodeDir]:
    eps: List[EpisodeDir] = []
    for p in parent.iterdir():
        if p.is_dir():
            m = EP_RE.match(p.name)
            if m:
                eps.append(EpisodeDir(int(m.group(1)), p))
    eps.sort(key=lambda e: e.episode_id)
    return eps


def find_mesh_files(episode_dir: Path, mesh_subdir: str = "mesh") -> List[Path]:
    """
    If episode_dir/mesh exists, search there.
    Otherwise search recursively under the episode dir.
    """
    candidates: List[Path] = []
    mesh_dir = episode_dir / mesh_subdir
    roots = [mesh_dir] if (mesh_dir.exists() and mesh_dir.is_dir()) else [episode_dir]

    exts = {".ply", ".stl"}
    for root in roots:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in exts:
                candidates.append(path)

    candidates.sort(key=lambda p: str(p).lower())
    return candidates


def load_as_single_mesh(mesh_path: Path) -> trimesh.Trimesh:
    """
    trimesh.load may return a Scene; merge into a single mesh.
    """
    obj = trimesh.load(mesh_path, force=None)
    if isinstance(obj, trimesh.Trimesh):
        return obj
    if isinstance(obj, trimesh.Scene):
        geoms = []
        for g in obj.geometry.values():
            if isinstance(g, trimesh.Trimesh) and len(g.vertices) > 0:
                geoms.append(g)
        if not geoms:
            raise ValueError(f"No mesh geometry found in scene: {mesh_path}")
        return trimesh.util.concatenate(geoms)
    raise TypeError(f"Unsupported mesh type from trimesh.load: {type(obj)}")


def apply_axis_map(vertices: np.ndarray, axis_order: Tuple[int, int, int], axis_signs: np.ndarray) -> np.ndarray:
    v = vertices[:, list(axis_order)]
    v = v * axis_signs.reshape(1, 3)
    return v


def normalize_vertices(vertices: np.ndarray, mode: str) -> np.ndarray:
    """
    mode:
      - none: do nothing
      - center: subtract centroid
      - bbox: center, then scale so max bbox extent == 1
    """
    if mode == "none":
        return vertices

    v = vertices.copy()
    centroid = v.mean(axis=0)
    v -= centroid

    if mode == "center":
        return v

    if mode == "bbox":
        mins = v.min(axis=0)
        maxs = v.max(axis=0)
        extents = (maxs - mins)
        s = float(np.max(extents))
        if s > 0:
            v /= s
        return v

    raise ValueError(f"Unknown normalize mode: {mode}")


def mesh_stats(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = vertices.mean(axis=0)
    bmin = vertices.min(axis=0)
    bmax = vertices.max(axis=0)
    return c, bmin, bmax


def safe_replace_file(src_tmp: Path, dst: Path, backup: bool) -> None:
    """
    Atomically replace dst with src_tmp; optionally keep dst.bak.
    """
    if backup and dst.exists():
        bak = dst.with_suffix(dst.suffix + ".bak")
        if bak.exists():
            bak.unlink()
        shutil.copy2(dst, bak)

    os.replace(str(src_tmp), str(dst))  # atomic on same filesystem


def process_mesh_file(mesh_path: Path, cfg: TransformConfig, dry_run: bool, backup: bool) -> dict:
    mesh = load_as_single_mesh(mesh_path)

    if mesh.vertices is None or len(mesh.vertices) == 0:
        raise ValueError(f"Empty mesh: {mesh_path}")

    v0 = np.asarray(mesh.vertices, dtype=np.float64)
    c0, bmin0, bmax0 = mesh_stats(v0)

    # 1) axis mapping
    v1 = apply_axis_map(v0, cfg.axis_order, cfg.axis_signs)

    # 2) normalize
    v2 = normalize_vertices(v1, cfg.normalize)

    # 3) per-episode scale(vec3) + offset(vec3)
    v3 = (v2 * cfg.scale.reshape(1, 3)) + cfg.offset.reshape(1, 3)

    c3, bmin3, bmax3 = mesh_stats(v3)

    mesh.vertices = v3

    dst_stl = mesh_path.with_suffix(".stl")
    tmp_stl = dst_stl.with_suffix(".stl.__tmp__")

    result = {
        "mesh_path": str(mesh_path),
        "output_stl": str(dst_stl),
        "orig_ext": mesh_path.suffix.lower(),
        "num_vertices": int(len(mesh.vertices)),
        "num_faces": int(len(mesh.faces) if mesh.faces is not None else 0),
        "scale_used": cfg.scale.tolist(),
        "offset_used": cfg.offset.tolist(),
        "centroid_before": c0.tolist(),
        "centroid_after": c3.tolist(),
        "bbox_min_before": bmin0.tolist(),
        "bbox_max_before": bmax0.tolist(),
        "bbox_min_after": bmin3.tolist(),
        "bbox_max_after": bmax3.tolist(),
    }

    if dry_run:
        return result

    mesh.export(tmp_stl, file_type="stl")
    safe_replace_file(tmp_stl, dst_stl, backup=backup)

    if mesh_path.suffix.lower() != ".stl":
        try:
            mesh_path.unlink()
        except Exception:
            pass

    return result


def _parse_scales_and_offsets_from_cfg(
    cfg: Dict[str, Any]
) -> Tuple[Optional[List[np.ndarray]], Optional[List[np.ndarray]], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Supports these YAML keys:

      Per-episode:
        scales:  [ s | [sx,sy,sz], ... ]
        offsets: [ [x,y,z], ... ]

      Defaults:
        scale: [sx,sy,sz]   OR  scale: s
        default_scale: [sx,sy,sz] OR default_scale: s
        offset: [x,y,z]
        default_offset: [x,y,z]

    Also supports: scale: [ ... ] as a per-episode list if it's not a single vec3.
    """
    scales_list: Optional[List[np.ndarray]] = None
    offsets_list: Optional[List[np.ndarray]] = None
    default_scale: Optional[np.ndarray] = None
    default_offset: Optional[np.ndarray] = None

    # ---- scales ----
    if isinstance(cfg.get("scales"), list):
        # per-episode list (each entry can be scalar or vec3)
        scales_list = [_coerce_scale3(v, "scales[]") for v in cfg["scales"]]
    elif isinstance(cfg.get("scale"), list):
        # could be default vec3 OR per-episode list
        scale_val = cfg["scale"]
        if len(scale_val) == 3 and all(isinstance(v, (int, float, np.number)) for v in scale_val):
            default_scale = _coerce_scale3(scale_val, "scale")
        else:
            scales_list = [_coerce_scale3(v, "scale[]") for v in scale_val]
    elif cfg.get("default_scale") is not None:
        default_scale = _coerce_scale3(cfg["default_scale"], "default_scale")
    elif isinstance(cfg.get("scale"), (int, float, np.number)):
        default_scale = _coerce_scale3(cfg["scale"], "scale")

    # ---- offsets ----
    if isinstance(cfg.get("offsets"), list):
        offsets_list = [_coerce_vec3(v, "offsets[]") for v in cfg["offsets"]]
    elif isinstance(cfg.get("offset"), list):
        off_val = cfg["offset"]
        # could be either [x,y,z] or [[x,y,z], ...]
        if len(off_val) == 3 and all(isinstance(v, (int, float, np.number)) for v in off_val):
            default_offset = _coerce_vec3(off_val, "offset")
        else:
            offsets_list = [_coerce_vec3(v, "offset[]") for v in off_val]
    elif cfg.get("default_offset") is not None:
        default_offset = _coerce_vec3(cfg["default_offset"], "default_offset")

    return scales_list, offsets_list, default_scale, default_offset


def build_run_config(args: argparse.Namespace) -> RunConfig:
    cfg: Dict[str, Any] = {}
    if args.config:
        cfg_path = Path(args.config).expanduser().resolve()
        if not cfg_path.exists():
            raise SystemExit(f"Config file not found: {cfg_path}")
        cfg = _load_yaml(cfg_path)

    # base fields
    parent_str = args.parent if args.parent is not None else cfg.get("parent", "example_dataset")
    parent = Path(parent_str).expanduser().resolve()

    normalize = args.normalize if args.normalize is not None else cfg.get("normalize", "center")

    axis_order_raw = args.axis_order if args.axis_order is not None else cfg.get("axis_order", [0, 1, 2])
    if not isinstance(axis_order_raw, (list, tuple)) or len(axis_order_raw) != 3:
        raise ValueError(f"axis_order must be length 3, got {axis_order_raw!r}")
    axis_order = (int(axis_order_raw[0]), int(axis_order_raw[1]), int(axis_order_raw[2]))

    axis_signs_raw = args.axis_signs if args.axis_signs is not None else cfg.get("axis_signs", [1, 1, 1])
    axis_signs = _coerce_vec3(axis_signs_raw, "axis_signs")

    mesh_subdir = args.mesh_subdir if args.mesh_subdir is not None else cfg.get("mesh_subdir", "mesh")

    dry_run_cfg = bool(cfg.get("dry_run", False))
    dry_run = True if args.dry_run else dry_run_cfg

    # backup: YAML "backup" default True; CLI --no-backup forces False
    backup_cfg = bool(cfg.get("backup", True))
    backup = False if args.no_backup else backup_cfg

    report_csv_str = args.report_csv if args.report_csv is not None else cfg.get("report_csv", "")
    report_csv = Path(report_csv_str).expanduser().resolve() if report_csv_str else None

    # per-episode lists + defaults from YAML
    scales_list, offsets_list, yaml_default_scale, yaml_default_offset = _parse_scales_and_offsets_from_cfg(cfg)

    # CLI single fallback values (used only as defaults)
    default_scale = (
        _coerce_scale3(args.scale, "scale") if args.scale is not None else
        (yaml_default_scale if yaml_default_scale is not None else np.array([0.1, 0.1, 0.1], dtype=np.float64))
    )
    default_offset = (
        _coerce_vec3(args.offset, "offset") if args.offset is not None else
        (yaml_default_offset if yaml_default_offset is not None else np.array([0.31, 0.41, 0.36], dtype=np.float64))
    )

    # optional strict validation: if both lists provided, they should match length
    if scales_list is not None and offsets_list is not None and len(scales_list) != len(offsets_list):
        raise ValueError(
            f"Config error: scales and offsets must have the same length, "
            f"got scales={len(scales_list)} offsets={len(offsets_list)}"
        )

    return RunConfig(
        parent=parent,
        mesh_subdir=str(mesh_subdir),
        dry_run=dry_run,
        backup=backup,
        report_csv=report_csv,
        normalize=str(normalize),
        axis_order=axis_order,
        axis_signs=axis_signs,
        default_scale=default_scale.astype(np.float64),
        default_offset=default_offset.astype(np.float64),
        scales=scales_list,
        offsets=offsets_list,
    )


def transform_for_episode(run_cfg: RunConfig, episode_id: int) -> TransformConfig:
    # list mapping is episode_id-1
    idx = episode_id - 1

    if run_cfg.scales is not None:
        if idx < 0 or idx >= len(run_cfg.scales):
            raise ValueError(
                f"No scale entry for episode {episode_id}. "
                f"Need scales[{idx}] but scales has length {len(run_cfg.scales)}."
            )
        scale = run_cfg.scales[idx].astype(np.float64)
    else:
        scale = run_cfg.default_scale.astype(np.float64)

    if run_cfg.offsets is not None:
        if idx < 0 or idx >= len(run_cfg.offsets):
            raise ValueError(
                f"No offset entry for episode {episode_id}. "
                f"Need offsets[{idx}] but offsets has length {len(run_cfg.offsets)}."
            )
        offset = run_cfg.offsets[idx].astype(np.float64)
    else:
        offset = run_cfg.default_offset.astype(np.float64)

    return TransformConfig(
        normalize=run_cfg.normalize,
        scale=scale,
        offset=offset,
        axis_order=run_cfg.axis_order,
        axis_signs=run_cfg.axis_signs,
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--config", type=str, default="",
                    help="YAML config file. If provided, per-episode scales/offsets can be read from it.")

    # optional overrides (if omitted, YAML values or defaults are used)
    ap.add_argument("--parent", type=str, default=None,
                    help="Dataset parent path containing episodes_1, episodes_2, ... (overrides YAML parent)")

    ap.add_argument("--normalize", type=str, choices=["none", "center", "bbox"], default=None,
                    help="Normalize vertices: none, center, or bbox (overrides YAML normalize)")

    ap.add_argument("--axis-order", type=int, nargs=3, default=None,
                    help="Reorder axes before normalization (overrides YAML axis_order)")

    ap.add_argument("--axis-signs", type=int, nargs=3, default=None,
                    help="Flip axes with -1 or keep with +1 (overrides YAML axis_signs)")

    ap.add_argument("--mesh-subdir", type=str, default=None,
                    help="If episode has this subdir, only search meshes there; otherwise search whole episode.")

    # Default fallbacks (used if YAML per-episode lists are not given)
    ap.add_argument("--scale", type=float, nargs=3, default=None,
                    help="Default scale vector sx sy sz if YAML per-episode scales not provided")
    ap.add_argument("--offset", type=float, nargs=3, default=None,
                    help="Default world offset x y z if YAML per-episode offsets not provided")

    ap.add_argument("--dry-run", action="store_true",
                    help="Do not write files; just print what would happen (overrides YAML dry_run)")
    ap.add_argument("--no-backup", action="store_true",
                    help="Do not create .bak backups before replacing (overrides YAML backup)")
    ap.add_argument("--report-csv", type=str, default=None,
                    help="Optional path to save a CSV report (overrides YAML report_csv)")

    args = ap.parse_args()

    run_cfg = build_run_config(args)

    if not run_cfg.parent.exists():
        raise SystemExit(f"Parent path not found: {run_cfg.parent}")

    episodes = iter_episode_dirs(run_cfg.parent)
    if not episodes:
        raise SystemExit(f"No episode directories found like episodes_1 under: {run_cfg.parent}")

    rows: List[dict] = []

    for ep in episodes:
        try:
            ep_cfg = transform_for_episode(run_cfg, ep.episode_id)
        except Exception as e:
            print(f"[FAIL] episodes_{ep.episode_id}: cannot build transform config: {e}")
            continue

        mesh_files = find_mesh_files(ep.path, mesh_subdir=run_cfg.mesh_subdir)
        if not mesh_files:
            print(f"[WARN] No mesh files found in {ep.path}")
            continue

        for mf in mesh_files:
            try:
                info = process_mesh_file(mf, ep_cfg, dry_run=run_cfg.dry_run, backup=run_cfg.backup)

                # add episode context into report
                info["episode_id"] = int(ep.episode_id)
                info["episode_dir"] = str(ep.path)
                rows.append(info)

                c_after = np.array(info["centroid_after"], dtype=float)
                err = np.linalg.norm(c_after - ep_cfg.offset)
                print(
                    f"[OK] ep={ep.episode_id} | scale={ep_cfg.scale.tolist()} | offset={ep_cfg.offset.tolist()} | "
                    f"{mf} -> {info['output_stl']} | "
                    f"centroid_after={info['centroid_after']} | "
                    f"||centroid-offset||={err:.6f}"
                )
            except Exception as e:
                print(f"[FAIL] ep={ep.episode_id} | {mf}: {e}")

    if run_cfg.report_csv:
        run_cfg.report_csv.parent.mkdir(parents=True, exist_ok=True)
        # stable column order: union of keys across rows
        fieldnames: List[str] = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)

        with run_cfg.report_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

        print(f"[REPORT] Wrote CSV report: {run_cfg.report_csv}")


if __name__ == "__main__":
    main()
