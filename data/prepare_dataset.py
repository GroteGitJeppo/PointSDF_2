#!/usr/bin/env python3
"""
Two-stage data preparation pipeline for PointSDF_2.

Sub-commands
------------
pcd
    Convert masked RGB-D images (.png) exported from ROS bags into partial
    point clouds (.ply).  These are the inputs to the PointNet++ encoder.

    Usage::

        python data/prepare_dataset.py pcd \\
            --img_root  data/3DPotatoTwin/1_rgbd/1_image \\
            --intrinsics data/3DPotatoTwin/1_rgbd/0_camera_intrinsics/realsense_d405_camera_intrinsic.json \\
            --out       data/3DPotatoTwin/1_rgbd/2_pcd

sdf
    Generate truncated SDF (TSDF) samples (.npz) from the complete
    laser/SfM point clouds used as Stage 1 ground truth.  Each label's
    output is written to::

        <out>/<label>/samples.npz

    which is the flat layout recognised by ``resolve_samples_npz()``.

    The input point clouds must have outward-pointing normals so that the
    normal direction can be used as a proxy for the inside/outside sign.
    Open3D can estimate and orient normals automatically if they are absent
    (see ``--estimate_normals``).

    Usage::

        python data/prepare_dataset.py sdf \\
            --src   data/3DPotatoTwin/2_sfm/2_pcd \\
            --out   data/3DPotatoTwin/sdfsamples/potato \\
            --ply_pattern "*_20000.ply"

    ``--src`` must contain one sub-folder per potato label, each holding
    one complete-scan PLY file.  Use ``--ply_pattern`` to match the exact
    filename if there are multiple PLY files per label.

augment
    Apply random shape augmentations (scale / rotation / shear) to the
    complete-scan PLYs and generate TSDF ``samples.npz`` for each variant.
    Augmented shapes are stored as ``<out>/<label>_NN/samples.npz`` (NN = 00…).
    Pass the output directory as ``augmented_sdf_data_dir`` in
    ``configs/train_deepsdf.yaml``.

    To generate regular + augmented samples in a single pass, use the
    ``sdf`` command with ``--augment_out`` instead.

    Usage (augmented only)::

        python data/prepare_dataset.py augment \\
            --src   data/3DPotatoTwin/2_sfm/2_pcd \\
            --out   data/3DPotatoTwin/sdfsamples/potato_augmented \\
            --ply_pattern "*_20000.ply"

    Usage (regular + augmented in one pass)::

        python data/prepare_dataset.py sdf \\
            --src         data/3DPotatoTwin/2_sfm/2_pcd \\
            --out         data/3DPotatoTwin/sdfsamples/potato \\
            --augment_out data/3DPotatoTwin/sdfsamples/potato_augmented \\
            --ply_pattern "*_20000.ply"

Adapted from
------------
* ``prepare_3dpotato_dataset.py`` (pcd command) — PointRAFT fork
* ``corepp/data_preparation/prepare_deepsdf_training_data.py`` (sdf command) — CoRe++ (Blok et al., 2025)
* ``corepp/data_preparation/augment.py`` (augment command) — CoRe++ (Blok et al., 2025)
"""

import argparse
import glob
import json
import os
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


# ============================================================================
# Shared utilities
# ============================================================================

def _check_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path)


# ============================================================================
# pcd command — RGB-D images → partial point clouds
# ============================================================================

def _load_intrinsics(intrinsics_file: str) -> o3d.camera.PinholeCameraIntrinsic:
    with open(intrinsics_file) as f:
        data = json.load(f)
    return o3d.camera.PinholeCameraIntrinsic(
        data['width'], data['height'],
        data['intrinsic_matrix'][0], data['intrinsic_matrix'][4],
        data['intrinsic_matrix'][6], data['intrinsic_matrix'][7],
    )


def _remove_distant_clusters(pcd, eps=0.05, min_points=100, distance_threshold=1.0):
    """Remove outlier clusters that are far from the main object."""
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))

    filtered_points, filtered_colors = [], []
    for cid in np.unique(labels):
        if cid == -1:
            continue
        mask = labels == cid
        centroid = points[mask].mean(axis=0)
        if np.linalg.norm(centroid) <= distance_threshold:
            filtered_points.append(points[mask])
            filtered_colors.append(colors[mask])

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(np.vstack(filtered_points))
    out.colors = o3d.utility.Vector3dVector(np.vstack(filtered_colors))
    return out


def _save_pcd(rgb, depth, mask, intrinsics, file_name, write_folder, visualize=False):
    rgbmask   = np.multiply(rgb,   np.expand_dims(mask, axis=2))
    depthmask = np.multiply(depth, mask)

    rgb_o3d   = o3d.geometry.Image((rgbmask[:, :, ::-1]).astype(np.uint8))
    depth_o3d = o3d.geometry.Image(depthmask)
    rgbd      = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d, depth_o3d, depth_scale=1000.0, depth_trunc=0.5,
        convert_rgb_to_intensity=False,
    )
    pcd          = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
    pcd_filtered = _remove_distant_clusters(pcd, eps=0.01, min_points=200, distance_threshold=1)

    if visualize:
        pcd.paint_uniform_color([1, 0, 0])
        o3d.visualization.draw_geometries(
            [pcd, pcd_filtered],
            window_name="Red=removed. Press Q to continue.",
        )

    subfolder   = file_name.split("/")[-2]
    folder_name = os.path.join(write_folder, subfolder)
    _check_dir(folder_name)
    basename  = os.path.basename(file_name)
    write_name = basename.replace("_rgb", "_pcd").replace(os.path.splitext(basename)[-1], ".ply")
    o3d.io.write_point_cloud(os.path.join(folder_name, write_name), pcd_filtered)


def cmd_pcd(args):
    """Convert masked RGB-D images to partial point clouds."""
    _check_dir(args.out)
    intrinsics = _load_intrinsics(args.intrinsics)

    supported = (".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2",
                 ".png", ".pbm", ".pgm", ".ppm", ".tiff", ".tif")

    rgb_files, depth_files = [], []
    for root, _, files in os.walk(args.img_root):
        for f in files:
            lower = f.lower()
            if lower.endswith(supported):
                if "_rgb" in f:
                    rgb_files.append(os.path.join(root, f))
                elif "_depth" in f:
                    depth_files.append(os.path.join(root, f))

    rgb_files.sort()
    depth_files.sort()
    print(f"Found {len(rgb_files)} RGB-D pairs under {args.img_root}")

    for rgb_path, depth_path in zip(rgb_files, depth_files):
        rgba  = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
        rgb   = rgba[:, :, :-1]
        mask  = rgba[:, :, -1]
        depth = cv2.imread(depth_path, -1)
        _save_pcd(rgb, depth, mask.astype(bool), intrinsics, rgb_path, args.out,
                  visualize=args.visualize)

    cv2.destroyAllWindows()
    print(f"Point clouds written to {args.out}")


# ============================================================================
# sdf command — complete PLY → TSDF samples.npz
# ============================================================================

def _generate_tsdf_samples(
    swl_points: np.ndarray,
    no_samples_per_point: int,
    tsdf_positive: float,
    tsdf_negative: float,
):
    """
    Generate TSDF sample coordinates and SDF values by offsetting surface
    points along their outward normal direction.

    Points are offset by a random distance along the outward normal
    (positive SDF = outside) or against it (negative SDF = inside).
    An additional envelope band of small positive offsets is added to give
    the decoder more near-surface supervision.

    Args:
        swl_points:          (N, 6) array — columns 0:3 surface points,
                             columns 3:6 viewpoint (used to infer normal direction).
        no_samples_per_point: number of offset samples per surface point.
        tsdf_positive:       max outward offset distance.
        tsdf_negative:       max inward offset distance (positive scalar).

    Returns:
        Tuple (pos, neg) each of shape (M, 4) — [x, y, z, sdf_value].
    """
    no_points      = swl_points.shape[0]
    offset_vectors = swl_points[:, 3:6] - swl_points[:, 0:3]
    offset_vectors = offset_vectors / np.linalg.norm(offset_vectors, axis=1, keepdims=True)
    offset_vectors = np.repeat(offset_vectors, no_samples_per_point, axis=0)

    n_samp = no_points * no_samples_per_point
    pos_offset     = tsdf_positive * np.random.rand(n_samp, 1)
    env_offset     = 0.1 * tsdf_positive * np.random.rand(n_samp, 1)
    neg_offset_val = (-1.0) * tsdf_negative * np.random.rand(n_samp, 1)

    base = np.repeat(swl_points[:, 0:3], no_samples_per_point, axis=0)

    pos = np.concatenate([base + pos_offset * offset_vectors, pos_offset], axis=1)
    env = np.concatenate([base + env_offset * offset_vectors, env_offset], axis=1)
    neg = np.concatenate([base + neg_offset_val * offset_vectors, neg_offset_val], axis=1)

    pos = np.concatenate([pos, env], axis=0)
    return pos.astype(np.float32), neg.astype(np.float32)


def _find_ply(label_dir: str, ply_pattern: str) -> str | None:
    """Return the first PLY file in label_dir matching ply_pattern, or None."""
    matches = glob.glob(os.path.join(label_dir, ply_pattern))
    if matches:
        return matches[0]
    # Fall back: any PLY file
    matches = glob.glob(os.path.join(label_dir, "*.ply"))
    return matches[0] if matches else None


def _ensure_normals(pcd: o3d.geometry.PointCloud) -> None:
    """Estimate outward-pointing normals in-place if the cloud has none."""
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        pcd.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))
        pcd.normals = o3d.utility.Vector3dVector(-np.asarray(pcd.normals))


def _samples_from_pcd(
    pcd: o3d.geometry.PointCloud,
    no_samples: int,
    tsdf_positive: float,
    tsdf_negative: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (pos, neg) TSDF sample arrays of shape (no_samples, 4) each."""
    points     = np.asarray(pcd.points)
    normals    = np.asarray(pcd.normals)
    viewpoints = points + normals
    swl_points = np.concatenate([points, viewpoints], axis=1)

    no_samples_per_point = max(1, int(np.ceil(no_samples / len(points))))
    pos, neg = _generate_tsdf_samples(swl_points, no_samples_per_point,
                                      tsdf_positive, tsdf_negative)

    rng = np.random.default_rng()
    pos = pos[rng.choice(len(pos), no_samples, replace=len(pos) < no_samples)]
    neg = neg[rng.choice(len(neg), no_samples, replace=len(neg) < no_samples)]
    return pos, neg


def cmd_sdf(args):
    """Generate TSDF samples (and optionally augmented variants) from complete point clouds."""
    _check_dir(args.out)
    do_augment = bool(getattr(args, 'augment_out', None))
    if do_augment:
        _check_dir(args.augment_out)
        aug_cfg = {
            'min_scale':        args.min_scale,
            'max_scale':        args.max_scale,
            'max_rotation_deg': args.max_rotation_deg,
            'max_shear':        args.max_shear,
        }
        print(
            f"Augmentation enabled: {args.no_augmentations} variants/shape → {args.augment_out}\n"
            f"  scale [{args.min_scale}, {args.max_scale}], "
            f"rotation ±{args.max_rotation_deg}°, shear ±{args.max_shear}"
        )

    labels = sorted(
        d for d in os.listdir(args.src)
        if os.path.isdir(os.path.join(args.src, d))
    )
    print(f"Found {len(labels)} label folders under {args.src}")

    skipped = 0
    for label in labels:
        label_dir = os.path.join(args.src, label)
        ply_path  = _find_ply(label_dir, args.ply_pattern)

        if ply_path is None:
            print(f"  [SKIP] {label}: no PLY file found")
            skipped += 1
            continue

        print(f"  {label}: loading {ply_path} ...", flush=True)
        pcd = o3d.io.read_point_cloud(ply_path)
        # Centre so SDF origin matches the encoder's T.Center() on partial scans.
        pcd.translate(-pcd.get_center())
        _ensure_normals(pcd)

        # ── regular samples ──────────────────────────────────────────────────
        out_dir  = os.path.join(args.out, label)
        out_path = os.path.join(out_dir, "samples.npz")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"    [SKIP] {label}: samples.npz exists (use --overwrite)")
        else:
            pos, neg = _samples_from_pcd(pcd, args.no_samples,
                                         args.tsdf_positive, args.tsdf_negative)
            _check_dir(out_dir)
            np.savez(out_path, pos=pos, neg=neg)
            print(f"    saved {args.no_samples} pos + neg → {out_path}")

        # ── augmented variants ───────────────────────────────────────────────
        if do_augment:
            for jdx in range(args.no_augmentations):
                aug_label = f"{label}_{jdx:02d}"
                aug_out   = os.path.join(args.augment_out, aug_label, "samples.npz")
                if os.path.exists(aug_out) and not args.overwrite:
                    print(f"    [SKIP] {aug_label}: samples.npz exists")
                    continue
                aug_pcd = _augment_pcd(pcd, aug_cfg)
                pos, neg = _samples_from_pcd(aug_pcd, args.no_samples,
                                             args.tsdf_positive, args.tsdf_negative)
                _check_dir(os.path.dirname(aug_out))
                np.savez(aug_out, pos=pos, neg=neg)
                print(f"    {aug_label}: saved → {aug_out}")

    print(f"\nDone. Skipped {skipped}/{len(labels)} labels.")


# ============================================================================
# augment command — complete PLY → augmented TSDF samples.npz variants
# ============================================================================

def _augment_pcd(pcd: o3d.geometry.PointCloud, cfg: dict) -> o3d.geometry.PointCloud:
    """Apply random scale / rotation-Z / shear-X to a point cloud."""
    import copy
    tmp = copy.deepcopy(pcd)

    # Isotropic scale (per-axis uniform, matching corepp augment.py)
    scale = np.random.uniform(cfg['min_scale'], cfg['max_scale'], size=(3,))
    pts = np.asarray(tmp.points) * scale
    tmp.points = o3d.utility.Vector3dVector(pts)

    # Rotation around Z axis
    angle = np.random.uniform(-cfg['max_rotation_deg'], cfg['max_rotation_deg']) * np.pi / 180.0
    R = o3d.geometry.get_rotation_matrix_from_xyz(np.array([0.0, 0.0, angle]))
    tmp.rotate(R, center=(0, 0, 0))

    # Shear in X direction (affects Y and Z columns)
    shear = np.random.uniform(-cfg['max_shear'], cfg['max_shear'], size=(2,))
    pts = np.asarray(tmp.points).copy()
    pts[:, 0] += shear[0] * pts[:, 1] + shear[1] * pts[:, 2]
    tmp.points = o3d.utility.Vector3dVector(pts)

    # Re-centre
    tmp.translate(-tmp.get_center())

    # Recompute outward normals (needed for TSDF sample generation)
    tmp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    tmp.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))
    tmp.normals = o3d.utility.Vector3dVector(-np.asarray(tmp.normals))

    return tmp


def cmd_augment(args):
    """Generate augmented TSDF samples from complete point clouds (augment-only mode)."""
    _check_dir(args.out)
    aug_cfg = {
        'min_scale':        args.min_scale,
        'max_scale':        args.max_scale,
        'max_rotation_deg': args.max_rotation_deg,
        'max_shear':        args.max_shear,
    }

    labels = sorted(
        d for d in os.listdir(args.src)
        if os.path.isdir(os.path.join(args.src, d))
    )
    print(f"Found {len(labels)} label folders under {args.src}")
    print(
        f"Generating {args.no_augmentations} variants/shape → {args.out}\n"
        f"  scale [{args.min_scale}, {args.max_scale}], "
        f"rotation ±{args.max_rotation_deg}°, shear ±{args.max_shear}"
    )

    skipped = 0
    for label in labels:
        label_dir = os.path.join(args.src, label)
        ply_path  = _find_ply(label_dir, args.ply_pattern)

        if ply_path is None:
            print(f"  [SKIP] {label}: no PLY file found")
            skipped += 1
            continue

        print(f"  {label}: loading {ply_path} ...", flush=True)
        pcd = o3d.io.read_point_cloud(ply_path)
        pcd.translate(-pcd.get_center())
        _ensure_normals(pcd)

        for jdx in range(args.no_augmentations):
            aug_label = f"{label}_{jdx:02d}"
            out_path  = os.path.join(args.out, aug_label, "samples.npz")

            if os.path.exists(out_path) and not args.overwrite:
                print(f"    [SKIP] {aug_label}: samples.npz exists (use --overwrite)")
                continue

            aug_pcd = _augment_pcd(pcd, aug_cfg)
            pos, neg = _samples_from_pcd(aug_pcd, args.no_samples,
                                         args.tsdf_positive, args.tsdf_negative)
            _check_dir(os.path.dirname(out_path))
            np.savez(out_path, pos=pos, neg=neg)
            print(f"    {aug_label}: saved → {out_path}")

    print(f"\nDone. Skipped {skipped}/{len(labels)} labels (no PLY found).")


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PointSDF_2 data preparation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── pcd ──────────────────────────────────────────────────────────────────
    p_pcd = sub.add_parser("pcd", help="RGB-D images → partial PLY point clouds")
    p_pcd.add_argument(
        "--img_root", required=True,
        help="Root folder of masked RGB-D images (searched recursively for *_rgb.* and *_depth.* files).",
    )
    p_pcd.add_argument(
        "--intrinsics", required=True,
        help="Path to camera intrinsics JSON (RealSense D405 format).",
    )
    p_pcd.add_argument(
        "--out", required=True,
        help="Output folder for partial PLY files (one sub-folder per label).",
    )
    p_pcd.add_argument("--visualize", action="store_true",
                       help="Open an Open3D window to preview each result.")

    # ── sdf ──────────────────────────────────────────────────────────────────
    p_sdf = sub.add_parser(
        "sdf",
        help="Complete PLY scans → TSDF samples.npz for Stage 1 training",
    )
    p_sdf.add_argument(
        "--src", required=True,
        help="Directory containing one sub-folder per potato label, each with a complete PLY scan.",
    )
    p_sdf.add_argument(
        "--out", required=True,
        help=(
            "Output root directory.  Each label is saved as <out>/<label>/samples.npz, "
            "matching the flat layout expected by resolve_samples_npz()."
        ),
    )
    p_sdf.add_argument(
        "--ply_pattern", default="*.ply",
        help="Glob pattern to locate the PLY file within each label folder (default: '*.ply'). "
             "Example: '*_20000.ply' to use the 20k-point SfM cloud.",
    )
    p_sdf.add_argument(
        "--no_samples", default=100_000, type=int,
        help="Number of positive and negative TSDF samples per shape (default: 100000).",
    )
    p_sdf.add_argument(
        "--tsdf_positive", default=0.04, type=float,
        help="Max outward (positive) SDF offset in metres (default: 0.04).",
    )
    p_sdf.add_argument(
        "--tsdf_negative", default=0.01, type=float,
        help="Max inward (negative) SDF offset in metres (default: 0.01).",
    )
    p_sdf.add_argument(
        "--estimate_normals", action="store_true",
        help="Force normal estimation even if normals are present in the PLY file.",
    )
    p_sdf.add_argument(
        "--overwrite", action="store_true",
        help="Re-generate samples.npz even if it already exists.",
    )
    # ── optional augmentation pass (combined mode) ────────────────────────
    p_sdf.add_argument(
        "--augment_out", default=None,
        help=(
            "If set, also generate augmented variants and save them here "
            "(e.g. data/3DPotatoTwin/sdfsamples/potato_augmented). "
            "Runs both regular and augmented SDF generation in a single pass."
        ),
    )
    p_sdf.add_argument("--no_augmentations", default=10, type=int,
                       help="Augmented variants per shape when --augment_out is set (default: 10).")
    p_sdf.add_argument("--min_scale",        default=0.5,  type=float,
                       help="Min per-axis scale factor for augmentation (default: 0.5).")
    p_sdf.add_argument("--max_scale",        default=2.0,  type=float,
                       help="Max per-axis scale factor for augmentation (default: 2.0).")
    p_sdf.add_argument("--max_rotation_deg", default=30.0, type=float,
                       help="Max rotation around Z in degrees for augmentation (default: 30).")
    p_sdf.add_argument("--max_shear",        default=0.5,  type=float,
                       help="Max shear magnitude for augmentation (default: 0.5).")

    # ── augment ──────────────────────────────────────────────────────────────
    p_aug = sub.add_parser(
        "augment",
        help="Complete PLY scans → augmented TSDF samples.npz variants",
    )
    p_aug.add_argument(
        "--src", required=True,
        help="Directory containing one sub-folder per label, each with a complete PLY scan "
             "(same source as the 'sdf' command).",
    )
    p_aug.add_argument(
        "--out", required=True,
        help=(
            "Output root directory.  Each augmented variant is saved as "
            "<out>/<label>_NN/samples.npz (NN = 00, 01, …)."
        ),
    )
    p_aug.add_argument(
        "--ply_pattern", default="*.ply",
        help="Glob pattern to locate the PLY file within each label folder (default: '*.ply'). "
             "Example: '*_20000.ply'.",
    )
    p_aug.add_argument(
        "--no_augmentations", default=10, type=int,
        help="Number of augmented variants per shape (default: 10).",
    )
    p_aug.add_argument(
        "--no_samples", default=100_000, type=int,
        help="Number of positive and negative TSDF samples per augmented shape (default: 100000).",
    )
    p_aug.add_argument(
        "--min_scale", default=0.5, type=float,
        help="Minimum per-axis scale factor (default: 0.5).",
    )
    p_aug.add_argument(
        "--max_scale", default=2.0, type=float,
        help="Maximum per-axis scale factor (default: 2.0).",
    )
    p_aug.add_argument(
        "--max_rotation_deg", default=30.0, type=float,
        help="Maximum rotation around Z axis in degrees (default: 30).",
    )
    p_aug.add_argument(
        "--max_shear", default=0.5, type=float,
        help="Maximum shear magnitude in X direction (default: 0.5).",
    )
    p_aug.add_argument(
        "--tsdf_positive", default=0.04, type=float,
        help="Max outward (positive) SDF offset in metres (default: 0.04).",
    )
    p_aug.add_argument(
        "--tsdf_negative", default=0.01, type=float,
        help="Max inward (negative) SDF offset in metres (default: 0.01).",
    )
    p_aug.add_argument(
        "--overwrite", action="store_true",
        help="Re-generate samples.npz even if it already exists.",
    )

    args = parser.parse_args()
    if args.command == "pcd":
        cmd_pcd(args)
    elif args.command == "sdf":
        cmd_sdf(args)
    else:
        cmd_augment(args)


if __name__ == "__main__":
    main()
