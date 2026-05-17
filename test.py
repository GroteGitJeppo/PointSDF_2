"""
Stage 2 — encoder evaluation.

Loads a trained encoder + decoder checkpoint, runs inference on the test
split, extracts potato volume via convex hull of SDF interior points, and
reports Chamfer distance, precision/recall/F1 (corepp-compatible) and
MAE / RMSE / R² against ground-truth volumes.

Metrics match corepp/test.py exactly:
  - Chamfer: Open3D point-cloud distance, (mean(gt→pred) + mean(pred→gt)) / 2
  - Precision/Recall/F1: percentage of points within 5 mm (0.005 m) threshold
  - GT: complete laser/SfM PLY per tuber, centred to match encoder pre-transform
  - Per-label ``year`` (from ``target_csv``, e.g. mesh_traits) is printed in the per-year summary

Timing (corepp-comparable):
  - Per-row milliseconds with CUDA synchronization between GPU stages so splits are
    meaningful on GPU (small overhead vs decode cost).
  - encoder_ms, latent_save_ms, decoder_ms, convex_hull_ms segment the pipeline;
    exec_time_ms is the wall time for that whole block (same components as before).
  - Excludes PLY load + FPS (process_ply) and Chamfer / P&R.
  - Printed aggregate exec stats exclude the first sample (CUDA warmup), matching
    corepp's skip of the first iteration.

Usage:
    python test.py --config configs/train_encoder.yaml --checkpoint weights/encoder/<run>/checkpoint.pth
"""

import argparse
import glob
import os
import timeit
import warnings
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd
import torch
import torch_fpsample
import torch_geometric.transforms as T
import yaml
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from torch_geometric.data import Data
from torch_geometric.typing import WITH_TORCH_CLUSTER
from tqdm import tqdm

from torch.utils.data import DataLoader

from data.ply_index import load_ply_files
from data.rgbd_corepp_dataset import RgbdCoreppDataset
from models import PointNetEncoder, SDFDecoder
from models.corepp_encoder import build_corepp_encoder, load_corepp_encoder_state
from utils import decode_sdf_hierarchical, get_volume_coords, sdf2mesh
from metrics_3d.chamfer_distance import ChamferDistance
from metrics_3d.precision_recall import PrecisionRecall

warnings.filterwarnings('ignore')

if not WITH_TORCH_CLUSTER:
    raise SystemExit("This code requires 'torch-cluster'")


def _sync_cuda(device: torch.device) -> None:
    """Wait for GPU work to finish so timers bracketing GPU sections are accurate."""
    if device.type == 'cuda':
        torch.cuda.synchronize()


def process_ply(ply_path: str, num_points: int, pre_transform, device,
                normalize_half_extent: float = 0.05):
    """Load, centre, normalise, FPS-sample a .ply file and return a batched PyG Data.

    Mirrors PointCloudLatentDataset.__getitem__: centre → isotropic normalise
    (max abs coord = normalize_half_extent) → FPS.  The scale ratio is stored
    as data.scale so the encoder can recover metric size information.
    """
    pcd = o3d.io.read_point_cloud(ply_path)
    points = torch.tensor(np.asarray(pcd.points), dtype=torch.float)
    data = Data(pos=points)
    data = pre_transform(data)          # centres the cloud
    points = data.pos

    # Isotropic normalisation — same as _normalize_points in encoder_dataset.py
    max_half_extent = points.abs().max().item()
    if max_half_extent > 1e-6:
        scale = max_half_extent / normalize_half_extent
        points = points / scale
    else:
        scale = 1.0

    if points.size(0) > num_points:
        points, _ = torch_fpsample.sample(points, num_points)
    data = Data(pos=points)
    data.batch = torch.zeros(points.size(0), dtype=torch.int64)
    data.scale = torch.tensor([scale], dtype=torch.float)
    return data.to(device)


def _load_gt_pcd(gt_pcd_dir: str, unique_id: str, ply_pattern: str):
    """
    Load the complete laser/SfM PLY for a given tuber, centre it to match
    the T.Center() pre-transform applied to partial scans.

    Returns an open3d.geometry.PointCloud, or None if no file is found.
    """
    matches = glob.glob(os.path.join(gt_pcd_dir, unique_id, ply_pattern))
    if not matches:
        return None
    pcd = o3d.io.read_point_cloud(matches[0])
    pcd.translate(-pcd.get_center())
    return pcd


def _chamfer_and_pr_one_pass(gt_pcd, mesh, cd_metric: ChamferDistance, pr_metric: PrecisionRecall):
    """
    Same semantics as separate ChamferDistance.update + PrecisionRecall.update +
    compute_at_threshold(0.005), but one mesh→point cloud conversion (1M samples)
    and one pair of Open3D distance computations instead of two copies.
    """
    if cd_metric.prediction_is_empty(mesh):
        return 1000.0, 0.0, 0.0, 0.0

    gt_conv = cd_metric.convert_to_pcd(gt_pcd)
    pred_conv = cd_metric.convert_to_pcd(mesh)
    dist_pt_2_gt = np.asarray(pred_conv.compute_point_cloud_distance(gt_conv))
    dist_gt_2_pt = np.asarray(gt_conv.compute_point_cloud_distance(pred_conv))
    chamfer_m = (float(np.mean(dist_gt_2_pt)) + float(np.mean(dist_pt_2_gt))) / 2.0

    t = pr_metric.find_nearest_threshold(0.005)
    p = 100.0 / len(dist_pt_2_gt) * int(np.sum(dist_pt_2_gt < t))
    r = 100.0 / len(dist_gt_2_pt) * int(np.sum(dist_gt_2_pt < t))
    if p == 0 or r == 0:
        f1 = 0.0
    else:
        f1 = 2 * p * r / (p + r)
    return chamfer_m, p, r, f1


def _encoder_settings(cfg: dict) -> dict:
    enc = cfg.get("encoder") or {}
    if not isinstance(enc, dict):
        return {"type": "pointnet"}
    out = dict(enc)
    out.setdefault("type", "pointnet")
    return out


def _append_result_row(
    rows,
    *,
    file_name: str,
    unique_id: str,
    gt_df: pd.DataFrame,
    gt_volume: float,
    pred_volume: float,
    chamfer_mm: float,
    prec: float,
    rec: float,
    f1: float,
    encoder_ms: float,
    latent_save_ms: float,
    decoder_ms: float,
    convex_hull_ms: float,
    elapsed_ms: float,
) -> None:
    cultivar = gt_df.loc[unique_id, "cultivar"] if "cultivar" in gt_df.columns else ""
    season = (
        gt_df.loc[unique_id, "growing_season"]
        if "growing_season" in gt_df.columns
        else ""
    )
    year_val = np.nan
    if "year" in gt_df.columns:
        yv = gt_df.loc[unique_id, "year"]
        if pd.notna(yv):
            try:
                year_val = int(float(yv))
            except (TypeError, ValueError):
                year_val = yv
    rows.append(
        {
            "file_name": file_name,
            "unique_id": unique_id,
            "cultivar": cultivar,
            "growing_season": season,
            "year": year_val,
            "gt_volume_ml": gt_volume,
            "pred_volume_ml": pred_volume,
            "chamfer_mm": chamfer_mm,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "encoder_ms": round(encoder_ms, 2),
            "latent_save_ms": round(latent_save_ms, 2),
            "decoder_ms": round(decoder_ms, 2),
            "convex_hull_ms": round(convex_hull_ms, 2),
            "exec_time_ms": round(elapsed_ms, 2),
        }
    )


def _decode_volume(
    *,
    latent: torch.Tensor,
    decoder,
    device: torch.device,
    grid_coords: torch.Tensor | None,
    grid_center: torch.Tensor,
    hierarchical_decode: bool,
    grid_bbox: float,
    coarse_resolution: int,
    fine_subdiv: int,
    surface_dilation: int,
    max_fine_queries: int | None,
    decode_chunk: int,
    unique_id: str,
) -> tuple[torch.Tensor | None, object | None, float, float, float]:
    """Run SDF decode + convex hull.

    Returns (grid_coords, mesh, pred_volume_ml, decoder_ms, convex_hull_ms).
    """
    t_dec0 = timeit.default_timer()
    if hierarchical_decode:
        grid_coords, pred_sdf = decode_sdf_hierarchical(
            latent=latent,
            decoder=decoder,
            bbox=grid_bbox,
            R_coarse=coarse_resolution,
            subdiv=fine_subdiv,
            surface_dilation=surface_dilation,
            device=device,
            clamp_dist=None,
            max_fine_queries=max_fine_queries,
            decode_chunk=decode_chunk,
            warn_fn=lambda msg: print(f"  {unique_id}: {msg}"),
        )
    else:
        latent_tiled = latent.expand(grid_coords.size(0), -1)
        decoder_input = torch.cat([latent_tiled, grid_coords], dim=1)
        pred_sdf = decoder(decoder_input)
    _sync_cuda(device)
    t_dec1 = timeit.default_timer()
    decoder_ms = (t_dec1 - t_dec0) * 1e3

    pred_volume = float("nan")
    mesh = None
    t_hull0 = timeit.default_timer()
    try:
        mesh = sdf2mesh(pred_sdf, grid_coords)
        if mesh.is_watertight():
            pred_volume = round(mesh.get_volume() * 1e6, 2)
        if float(grid_center.norm()) > 1e-6:
            mesh.translate(-grid_center.cpu().numpy())
    except (ValueError, RuntimeError) as e:
        print(f"  Mesh extraction failed for {unique_id}: {e}")
    _sync_cuda(device)
    t_hull1 = timeit.default_timer()
    convex_hull_ms = (t_hull1 - t_hull0) * 1e3
    return grid_coords, mesh, pred_volume, decoder_ms, convex_hull_ms


def main(cfg: dict, checkpoint_path: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ----- Load architecture config from the decoder config -----
    with open(cfg['decoder_config']) as f:
        decoder_cfg = yaml.safe_load(f)
    latent_size = decoder_cfg['latent_size']
    enc_cfg = _encoder_settings(cfg)
    encoder_type = enc_cfg.get('type', 'pointnet').lower()
    print(f'Encoder type: {encoder_type}')

    decoder = SDFDecoder(
        latent_size=latent_size,
        num_layers=decoder_cfg['num_layers'],
        inner_dim=decoder_cfg['inner_dim'],
        skip_connections=decoder_cfg['skip_connections'],
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    decoder.load_state_dict(ckpt['decoder_state_dict'])
    decoder.eval()

    if encoder_type == 'corepp_rgbd':
        corepp_weights = enc_cfg.get('corepp_weights')
        if not corepp_weights:
            raise ValueError(
                "encoder.type is 'corepp_rgbd' but encoder.corepp_weights is not set in config"
            )
        encoder = build_corepp_encoder(
            variant=enc_cfg.get('variant', 'pool'),
            latent_size=latent_size,
            input_size=int(enc_cfg.get('input_size', 304)),
        ).to(device)
        load_corepp_encoder_state(encoder, corepp_weights, device=str(device))
        print(f'Loaded CoRe++ encoder from {corepp_weights}')
    else:
        encoder = PointNetEncoder(latent_size=latent_size).to(device)
        encoder.load_state_dict(ckpt['encoder_state_dict'])
        print(f'Loaded PointNet encoder from {checkpoint_path}')

    encoder.eval()
    print(f'Loaded decoder from {checkpoint_path}')

    # ----- Dataset paths -----
    splits_df = pd.read_csv(cfg['splits_csv'], delimiter=',')
    test_ids = set(splits_df.loc[splits_df['split'] == 'test', 'label'].astype(str))

    volume_col = cfg.get('volume_column', 'volume_ml')
    gt_df = pd.read_csv(cfg['target_csv'], delimiter=',').set_index('label')

    # Merge optional metadata CSV (cultivar, growing_season) if provided and
    # those columns are not already present in the primary target CSV.
    metadata_csv = cfg.get('metadata_csv', None)
    if metadata_csv:
        meta_df = pd.read_csv(metadata_csv, delimiter=',').set_index('label')
        for col in ('cultivar', 'growing_season'):
            if col not in gt_df.columns and col in meta_df.columns:
                gt_df = gt_df.join(meta_df[[col]], how='left')

    # ----- Year filter -----
    year_filter = cfg.get('_year_filter', 'all')
    if year_filter != 'all':
        if 'year' not in gt_df.columns:
            raise ValueError(
                f"--year {year_filter} requested but target_csv has no 'year' column."
            )
        target_year = int(year_filter)
        year_ids = set(
            gt_df[gt_df['year'].apply(
                lambda v: pd.notna(v) and int(float(v)) == target_year
            )].index.astype(str)
        )
        before = len(test_ids)
        test_ids = test_ids & year_ids
        print(f'Year filter: {target_year} — kept {len(test_ids)}/{before} test labels')
    else:
        print(f'Year filter: all — {len(test_ids)} test labels')

    pre_transform = T.Center()
    num_points = cfg.get('num_points', 1024)
    normalize_half_extent = float(cfg.get('normalize_half_extent', 0.05))
    grid_resolution = cfg.get('grid_resolution', 64)
    grid_bbox = cfg.get('grid_bbox', 0.15)

    hierarchical_decode = bool(cfg.get('hierarchical_decode', False))
    coarse_resolution = int(cfg.get('coarse_resolution', 16))
    fine_subdiv = int(cfg.get('fine_subdiv', 4))
    surface_dilation = int(cfg.get('surface_dilation', 1))
    max_fine_queries = cfg.get('max_fine_queries', None)
    if max_fine_queries is not None:
        max_fine_queries = int(max_fine_queries)
    decode_chunk = int(cfg.get('hierarchical_decode_chunk', 131072))

    # grid_center shifts the SDF query grid from the origin to the position where
    # the complete laser scans actually live in the scanner coordinate frame.
    # Required when the decoder was trained on uncentered data (e.g. corepp weights).
    # Compute the value once on the server with the script in train_encoder.yaml.
    grid_center = torch.tensor(
        cfg.get('grid_center', [0.0, 0.0, 0.0]), dtype=torch.float, device=device
    )
    if float(grid_center.norm()) > 1e-6:
        print(f'SDF grid center offset: {grid_center.cpu().tolist()}')

    if hierarchical_decode:
        R_fine = (coarse_resolution - 1) * fine_subdiv + 1
        effective_resolution = R_fine
        print(
            f'Hierarchical SDF decode: R_coarse={coarse_resolution}, subdiv={fine_subdiv}, '
            f'dilation={surface_dilation} → embedded R_fine={R_fine} (grid_resolution={grid_resolution} unused)'
        )
        grid_coords = None
    else:
        effective_resolution = grid_resolution
        grid_coords = get_volume_coords(resolution=grid_resolution, bbox=grid_bbox).to(device) + grid_center

    # GT point clouds for corepp-compatible Chamfer / P&R
    gt_pcd_dir = cfg.get('gt_pcd_dir', None)
    gt_ply_pattern = cfg.get('gt_ply_pattern', '*.ply')
    compute_shape_metrics = gt_pcd_dir is not None
    if compute_shape_metrics:
        print(f'Shape metrics enabled (GT PLY from {gt_pcd_dir})')
    else:
        print('Shape metrics disabled (set gt_pcd_dir in encoder config to enable)')

    # Metric objects — identical to corepp/test.py
    cd_metric = ChamferDistance()
    pr_metric = PrecisionRecall(0.001, 0.01, 10)

    latent_dir = cfg.get('latent_dir')
    if not latent_dir:
        raise ValueError(
            "test.py requires 'latent_dir' in the encoder config (same base as train/val "
            "Codes). Latents are written to <latent_dir>/test/<ply_stem>.pth"
        )
    latent_test_dir = os.path.join(latent_dir, 'test')
    os.makedirs(latent_test_dir, exist_ok=True)
    print(f'Encoder latents will be saved under {latent_test_dir}')

    # One GT PLY per tuber — avoid repeated glob + disk read each test sample
    gt_pcd_cache: dict[str, o3d.geometry.PointCloud | None] = {}

    # ----- Output columns -----
    columns = [
        'file_name', 'unique_id', 'cultivar', 'growing_season', 'year',
        'gt_volume_ml', 'pred_volume_ml',
        'chamfer_mm', 'precision', 'recall', 'f1',
        'encoder_ms', 'latent_save_ms', 'decoder_ms', 'convex_hull_ms',
        'exec_time_ms',
    ]
    rows = []
    latent_buffer: dict[str, torch.Tensor] = {}
    exec_times = []
    encoder_times: list[float] = []
    latent_save_times: list[float] = []
    decoder_times: list[float] = []
    hull_times: list[float] = []
    chamfer_values = []
    prec_values = []
    rec_values = []
    f1_values = []

    def _run_sample(
        *,
        file_name: str,
        unique_id: str,
        stem: str,
        latent: torch.Tensor,
        encoder_ms: float,
        latent_save_ms: float,
    ) -> None:
        nonlocal grid_coords
        grid_coords, mesh, pred_volume, decoder_ms, convex_hull_ms = _decode_volume(
            latent=latent,
            decoder=decoder,
            device=device,
            grid_coords=grid_coords,
            grid_center=grid_center,
            hierarchical_decode=hierarchical_decode,
            grid_bbox=grid_bbox,
            coarse_resolution=coarse_resolution,
            fine_subdiv=fine_subdiv,
            surface_dilation=surface_dilation,
            max_fine_queries=max_fine_queries,
            decode_chunk=decode_chunk,
            unique_id=unique_id,
        )

        chamfer_mm = float('nan')
        prec = float('nan')
        rec = float('nan')
        f1 = float('nan')
        if compute_shape_metrics and mesh is not None:
            try:
                if unique_id not in gt_pcd_cache:
                    gt_pcd_cache[unique_id] = _load_gt_pcd(
                        gt_pcd_dir, unique_id, gt_ply_pattern
                    )
                gt_pcd = gt_pcd_cache[unique_id]
                if gt_pcd is not None:
                    chamfer_m, prec, rec, f1 = _chamfer_and_pr_one_pass(
                        gt_pcd, mesh, cd_metric, pr_metric
                    )
                    chamfer_mm = round(chamfer_m * 1000, 6)
                    chamfer_values.append(chamfer_m)
                    prec = round(prec, 1)
                    rec = round(rec, 1)
                    f1 = round(f1, 1)
                    prec_values.append(prec)
                    rec_values.append(rec)
                    f1_values.append(f1)
                else:
                    print(f'  GT PLY not found for {unique_id}')
            except Exception as e:
                print(f'  Shape metrics failed for {unique_id}: {e}')

        elapsed_ms = encoder_ms + latent_save_ms + decoder_ms + convex_hull_ms
        exec_times.append(elapsed_ms)
        encoder_times.append(encoder_ms)
        latent_save_times.append(latent_save_ms)
        decoder_times.append(decoder_ms)
        hull_times.append(convex_hull_ms)

        gt_volume = float(gt_df.loc[unique_id, volume_col])
        _append_result_row(
            rows,
            file_name=file_name,
            unique_id=unique_id,
            gt_df=gt_df,
            gt_volume=gt_volume,
            pred_volume=pred_volume,
            chamfer_mm=chamfer_mm,
            prec=prec,
            rec=rec,
            f1=f1,
            encoder_ms=encoder_ms,
            latent_save_ms=latent_save_ms,
            decoder_ms=decoder_ms,
            convex_hull_ms=convex_hull_ms,
            elapsed_ms=elapsed_ms,
        )

    with torch.no_grad():
        if encoder_type == 'corepp_rgbd':
            rgbd_root = enc_cfg.get('rgbd_data_dir')
            if not rgbd_root:
                raise ValueError("encoder.rgbd_data_dir is required for corepp_rgbd")
            rgbd_ds = RgbdCoreppDataset(
                data_root=rgbd_root,
                split='test',
                input_size=int(enc_cfg.get('input_size', 304)),
                detection_input=enc_cfg.get('detection_input', 'mask'),
                normalize_depth=bool(enc_cfg.get('normalize_depth', True)),
                depth_min=float(enc_cfg.get('depth_min', 230)),
                depth_max=float(enc_cfg.get('depth_max', 350)),
                label_filter=test_ids,
            )
            loader = DataLoader(rgbd_ds, batch_size=1, shuffle=False)
            for batch in tqdm(loader, desc='Testing (CoRe++ RGB-D)'):
                unique_id = batch['label'][0]
                if unique_id not in gt_df.index:
                    continue
                file_name = batch['file_name'][0]
                frame_id = batch['frame_id'][0]
                rgb = batch['rgb'].to(device)
                depth = batch['depth'].to(device)
                t0 = timeit.default_timer()
                encoder_input = torch.cat((rgb, depth), dim=1)
                latent = encoder(encoder_input)
                _sync_cuda(device)
                t1 = timeit.default_timer()
                encoder_ms = (t1 - t0) * 1e3
                t_ls0 = timeit.default_timer()
                latent_buffer[frame_id] = latent.detach().cpu().squeeze()
                t_ls1 = timeit.default_timer()
                latent_save_ms = (t_ls1 - t_ls0) * 1e3
                _run_sample(
                    file_name=file_name,
                    unique_id=unique_id,
                    stem=frame_id,
                    latent=latent,
                    encoder_ms=encoder_ms,
                    latent_save_ms=latent_save_ms,
                )
        else:
            ply_files = load_ply_files(cfg['data_root'], test_ids, cfg.get('ply_index_csv'))
            for ply_file in tqdm(ply_files, desc='Testing'):
                unique_id = os.path.basename(os.path.dirname(ply_file))
                if unique_id not in gt_df.index:
                    continue
                data = process_ply(
                    ply_file, num_points, pre_transform, device, normalize_half_extent
                )
                t0 = timeit.default_timer()
                latent = encoder(data)
                _sync_cuda(device)
                t1 = timeit.default_timer()
                encoder_ms = (t1 - t0) * 1e3
                t_ls0 = timeit.default_timer()
                stem = Path(ply_file).stem
                latent_buffer[stem] = latent.detach().cpu().squeeze()
                t_ls1 = timeit.default_timer()
                latent_save_ms = (t_ls1 - t_ls0) * 1e3
                _run_sample(
                    file_name=ply_file,
                    unique_id=unique_id,
                    stem=stem,
                    latent=latent,
                    encoder_ms=encoder_ms,
                    latent_save_ms=latent_save_ms,
                )

    batch_latent_path = os.path.join(latent_test_dir, 'all_latents.pth')
    torch.save(latent_buffer, batch_latent_path)
    print(f'Encoder latents saved to {batch_latent_path}')

    df_out = pd.DataFrame(rows, columns=columns)
    valid = df_out.dropna(subset=['pred_volume_ml'])

    gt_arr = valid['gt_volume_ml'].to_numpy()
    pred_arr = valid['pred_volume_ml'].to_numpy()

    print(f'\nTest results ({len(valid)}/{len(df_out)} with valid meshes):')
    print(f'  MAE volume:    {mean_absolute_error(gt_arr, pred_arr):.2f} mL')
    print(f'  RMSE volume:   {root_mean_squared_error(gt_arr, pred_arr):.2f} mL')
    print(f'  R²:            {r2_score(gt_arr, pred_arr):.3f}')
    if chamfer_values:
        print(f'  Chamfer:       {np.mean(chamfer_values) * 1000:.3f} mm  (n={len(chamfer_values)})')
    if prec_values:
        print(f'  Precision@5mm: {np.mean(prec_values):.1f}%')
        print(f'  Recall@5mm:    {np.mean(rec_values):.1f}%')
        print(f'  F1@5mm:        {np.mean(f1_values):.1f}%')
    if not exec_times:
        print('  Avg exec:      n/a (no samples timed)')
    elif len(exec_times) > 1:
        sl = slice(1, None)
        mean_exec = float(np.mean(exec_times[sl]))
        median_exec = float(np.median(exec_times[sl]))
        print(
            f'  Exec total (excl. 1st sample): median {median_exec:.1f} ms | mean {mean_exec:.1f} ms'
        )
        print(
            f'    mean encoder {float(np.mean(encoder_times[sl])):.1f} ms | '
            f'latent save {float(np.mean(latent_save_times[sl])):.1f} ms | '
            f'decoder {float(np.mean(decoder_times[sl])):.1f} ms | '
            f'convex hull {float(np.mean(hull_times[sl])):.1f} ms'
        )

    def _shape_str(sel):
        cd_vals = sel['chamfer_mm'].dropna()
        f1_vals = sel['f1'].dropna()
        parts = []
        if len(cd_vals) > 0:
            parts.append(f'CD={cd_vals.mean():.3f} mm')
        if len(f1_vals) > 0:
            parts.append(f'F1={f1_vals.mean():.1f}%')
        return (' | ' + ' | '.join(parts)) if parts else ''

    # Per-cultivar breakdown
    if 'cultivar' in df_out.columns and df_out['cultivar'].notna().any():
        print('\n=== Per cultivar ===')
        for cultivar in valid['cultivar'].unique():
            sel = valid[valid['cultivar'] == cultivar]
            print(
                f'  {cultivar}: n={len(sel)} | '
                f'MAE={mean_absolute_error(sel["gt_volume_ml"], sel["pred_volume_ml"]):.2f} mL | '
                f'R²={r2_score(sel["gt_volume_ml"], sel["pred_volume_ml"]):.3f}'
                f'{_shape_str(sel)}'
            )

    # Per-year breakdown (mesh_traits year, e.g. 2023 vs 2025 cohort)
    if 'year' in df_out.columns and valid['year'].notna().any():
        print('\n=== Per year ===')
        for y in sorted(
            valid['year'].dropna().unique(),
            key=lambda v: (float(v) if isinstance(v, (int, float, np.integer)) else str(v)),
        ):
            sel = valid[valid['year'] == y]
            if len(sel) == 0:
                continue
            print(
                f'  {y}: n={len(sel)} | '
                f'MAE={mean_absolute_error(sel["gt_volume_ml"], sel["pred_volume_ml"]):.2f} mL | '
                f'R²={r2_score(sel["gt_volume_ml"], sel["pred_volume_ml"]):.3f}'
                f'{_shape_str(sel)}'
            )

    results_path = os.path.join(
        os.path.dirname(checkpoint_path), f'test_results_{effective_resolution}.csv'
    )
    df_out.to_csv(results_path, index=False)
    print(f'\nResults saved to: {results_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 2: encoder evaluation')
    parser.add_argument('--config', '-c', required=True, help='Path to YAML encoder config')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint.pth')
    parser.add_argument(
        '--year', default='all', choices=['2023', '2025', 'all'],
        help='Restrict evaluation to a single year cohort (2023 / 2025) or run on all test labels (default: all)',
    )
    parser.add_argument(
        '--grid_resolution', type=int, default=None,
        help='Override grid_resolution from config (number of voxels per axis)',
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.grid_resolution is not None:
        cfg['grid_resolution'] = args.grid_resolution
    cfg['_year_filter'] = args.year
    main(cfg, args.checkpoint)
