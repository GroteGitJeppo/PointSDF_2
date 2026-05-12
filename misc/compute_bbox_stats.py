"""
Compute bounding-box statistics across all complete SfM scans.

Mirrors corepp's find_global_bbox() logic: for each sample find the largest
single axis-aligned dimension (dx, dy, dz), then report the per-sample
distribution and the global maximum.

Usage (run from PointSDF_2/ or pass an absolute path):
    python misc/compute_bbox_stats.py
    python misc/compute_bbox_stats.py --pcd_dir data/3DPotatoTwin/2_sfm/2_pcd

File layout assumed:
    <pcd_dir>/<sample>/<sample>_10000.ply
    e.g. data/3DPotatoTwin/2_sfm/2_pcd/2R1-2/2R1-2_10000.ply

The printed "global dmax" is the value corepp would use as its bbox half-extent
(dmax / 2).  For PointSDF_2 the relevant number is dmax in normalised units
(i.e. after centering + isotropic scale to normalize_half_extent = 0.05 m),
which is always exactly 0.05 m by construction — but knowing the raw metric
dmax tells you how much physical space one normalised unit represents.
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def scan_dir(pcd_dir: str, pattern: str) -> list[Path]:
    """Return all PLY files matching *pattern* inside immediate subdirectories.

    Uses pathlib so forward/back-slashes are handled correctly on both
    Windows and Linux.
    """
    root = Path(pcd_dir)
    files = sorted(root.glob(f"*/{pattern}"))
    return files


def bbox_dims(ply_path: Path):
    """Return (dx, dy, dz, dmax) in metres for the point cloud at ply_path."""
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if len(pcd.points) == 0:
        return None
    pts = np.asarray(pcd.points)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    dx, dy, dz = hi - lo
    return float(dx), float(dy), float(dz), float(max(dx, dy, dz))


def main(pcd_dir: str, pattern: str) -> None:
    files = scan_dir(pcd_dir, pattern)
    if not files:
        root = Path(pcd_dir)
        print(f"No files found matching: {root / '*' / pattern}")
        print(f"  Resolved root: {root.resolve()}")
        return

    print(f"Found {len(files)} PLY files under {Path(pcd_dir).resolve()}\n")

    rows = []
    failed = []
    for f in files:
        result = bbox_dims(f)
        if result is None:
            failed.append(f)
            continue
        dx, dy, dz, dmax = result
        label = f.parent.name          # immediate parent folder = sample label
        rows.append((label, dx, dy, dz, dmax))

    if failed:
        print(f"WARNING: {len(failed)} empty/unreadable files skipped:")
        for f in failed:
            print(f"  {f}")   # Path objects print cleanly on both platforms
        print()

    rows.sort(key=lambda r: r[4], reverse=True)

    dmaxes = np.array([r[4] for r in rows])
    dxs    = np.array([r[1] for r in rows])
    dys    = np.array([r[2] for r in rows])
    dzs    = np.array([r[3] for r in rows])

    col_w = max(len(r[0]) for r in rows)
    header = f"{'label':<{col_w}}   {'dx_mm':>7}  {'dy_mm':>7}  {'dz_mm':>7}  {'dmax_mm':>8}"
    print(header)
    print("-" * len(header))
    for label, dx, dy, dz, dmax in rows:
        print(f"{label:<{col_w}}   {dx*1000:7.1f}  {dy*1000:7.1f}  {dz*1000:7.1f}  {dmax*1000:8.1f}")

    print()
    print("=" * 50)
    print("Summary (all values in mm):")
    print(f"  N scans      : {len(rows)}")
    print(f"  dmax  global : {dmaxes.max()*1000:.1f}  ← corepp bbox side (use half = {dmaxes.max()/2*1000:.1f} mm as half-extent)")
    print(f"  dmax  mean   : {dmaxes.mean()*1000:.1f}")
    print(f"  dmax  median : {np.median(dmaxes)*1000:.1f}")
    print(f"  dmax  p95    : {np.percentile(dmaxes, 95)*1000:.1f}")
    print(f"  dmax  min    : {dmaxes.min()*1000:.1f}")
    print()
    print(f"  dx    mean   : {dxs.mean()*1000:.1f}  (range {dxs.min()*1000:.1f} – {dxs.max()*1000:.1f})")
    print(f"  dy    mean   : {dys.mean()*1000:.1f}  (range {dys.min()*1000:.1f} – {dys.max()*1000:.1f})")
    print(f"  dz    mean   : {dzs.mean()*1000:.1f}  (range {dzs.min()*1000:.1f} – {dzs.max()*1000:.1f})")
    print()

    # PointSDF_2 context
    norm_he = 0.05  # normalize_half_extent in metres
    global_dmax_m = dmaxes.max()
    scale_ratio = global_dmax_m / (2 * norm_he)
    print("PointSDF_2 context (normalize_half_extent = 0.05 m):")
    print(f"  1 normalised unit = {scale_ratio*1000:.1f} mm in metric space")
    print(f"  Potato occupies [-0.05, 0.05]^3 after normalisation by construction.")
    print()
    print("Grid resolution guidance (grid_bbox = 0.05 m, full side = 100 mm):")
    for res in [16, 20, 24, 28, 32, 40, 48, 64]:
        spacing_norm = 0.10 / (res - 1)
        spacing_mm   = spacing_norm * scale_ratio * 1000
        n_pts        = res ** 3
        print(f"  grid_resolution={res:3d} → {n_pts:>8,} points, "
              f"voxel spacing ≈ {spacing_norm*1000:.1f} norm-mm  ({spacing_mm:.1f} metric mm)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute bounding-box statistics across all SfM scans."
    )
    parser.add_argument(
        "--pcd_dir",
        default="data/3DPotatoTwin/2_sfm/2_pcd",
        help="Root directory containing one sub-folder per sample "
             "(default: data/3DPotatoTwin/2_sfm/2_pcd)",
    )
    parser.add_argument(
        "--pattern",
        default="*_10000.ply",
        help="Glob pattern for PLY files inside each sample folder "
             "(default: *_10000.ply, matches e.g. 2R1-2_10000.ply)",
    )
    args = parser.parse_args()
    main(args.pcd_dir, args.pattern)
