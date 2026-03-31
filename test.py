"""
Stage 2 — encoder evaluation.

Loads a trained encoder + decoder checkpoint, runs inference on the test
split, extracts potato volume via convex hull of SDF interior points, and
reports MAE / RMSE / R² against ground-truth volumes.

Usage:
    python test.py --config configs/train_encoder.yaml --checkpoint weights/encoder/<run>/checkpoint.pth
"""

import argparse
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

from models import PointNetEncoder, SDFDecoder
from utils import get_volume_coords, sdf2mesh

warnings.filterwarnings('ignore')

if not WITH_TORCH_CLUSTER:
    raise SystemExit("This code requires 'torch-cluster'")


def process_ply(ply_path: str, num_points: int, pre_transform, device):
    """Load, centre, FPS-sample a .ply file and return a batched PyG Data."""
    pcd = o3d.io.read_point_cloud(ply_path)
    points = torch.tensor(np.asarray(pcd.points), dtype=torch.float)
    data = Data(pos=points)
    data = pre_transform(data)
    points = data.pos
    if points.size(0) > num_points:
        points, _ = torch_fpsample.sample(points, num_points)
    data = Data(pos=points)
    data.batch = torch.zeros(points.size(0), dtype=torch.int64)
    return data.to(device)


def main(cfg: dict, checkpoint_path: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ----- Load architecture config from the decoder config -----
    with open(cfg['decoder_config']) as f:
        decoder_cfg = yaml.safe_load(f)
    latent_size = decoder_cfg['latent_size']

    # ----- Load models -----
    encoder = PointNetEncoder(latent_size=latent_size).to(device)
    decoder = SDFDecoder(
        latent_size=latent_size,
        num_layers=decoder_cfg['num_layers'],
        inner_dim=decoder_cfg['inner_dim'],
        skip_connections=decoder_cfg['skip_connections'],
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    decoder.load_state_dict(ckpt['decoder_state_dict'])
    encoder.eval()
    decoder.eval()
    print(f'Loaded checkpoint from {checkpoint_path}')

    # ----- Dataset paths -----
    splits_df = pd.read_csv(cfg['splits_csv'], delimiter=',')
    test_ids = set(splits_df.loc[splits_df['split'] == 'test', 'label'].astype(str))
    all_files = list(Path(cfg['data_root']).rglob('*.ply'))
    ply_files = [str(f) for f in all_files if f.parent.name in test_ids]

    gt_df = pd.read_csv(cfg['target_csv'], delimiter=',').set_index('label')

    pre_transform = T.Center()
    num_points = cfg.get('num_points', 1024)
    grid_resolution = cfg.get('grid_resolution', 64)
    grid_bbox = cfg.get('grid_bbox', 0.15)

    # Pre-compute grid coords (shared across all test samples)
    grid_coords = get_volume_coords(resolution=grid_resolution, bbox=grid_bbox).to(device)

    # ----- Output columns -----
    columns = ['file_name', 'unique_id', 'cultivar', 'growing_season',
               'gt_volume_ml', 'pred_volume_ml', 'exec_time_ms']
    rows = []
    exec_times = []

    with torch.no_grad():
        for ply_file in tqdm(ply_files, desc='Testing'):
            unique_id = os.path.basename(os.path.dirname(ply_file))

            if unique_id not in gt_df.index:
                continue

            gt_volume = float(gt_df.loc[unique_id, 'volume_ml'])

            t0 = timeit.default_timer()

            data = process_ply(ply_file, num_points, pre_transform, device)
            latent = encoder(data)                          # (1, latent_size)

            latent_tiled = latent.expand(grid_coords.size(0), -1)
            decoder_input = torch.cat([latent_tiled, grid_coords], dim=1)
            pred_sdf = decoder(decoder_input)               # (N, 1)

            pred_volume = float('nan')
            try:
                mesh = sdf2mesh(pred_sdf, grid_coords)
                if mesh.is_watertight():
                    pred_volume = round(mesh.get_volume() * 1e6, 2)  # m³ → mL
            except (ValueError, RuntimeError) as e:
                print(f'  Mesh extraction failed for {unique_id}: {e}')

            elapsed_ms = (timeit.default_timer() - t0) * 1e3
            exec_times.append(elapsed_ms)

            cultivar = gt_df.loc[unique_id, 'cultivar'] if 'cultivar' in gt_df.columns else ''
            season = gt_df.loc[unique_id, 'growing_season'] if 'growing_season' in gt_df.columns else ''

            rows.append({
                'file_name': ply_file,
                'unique_id': unique_id,
                'cultivar': cultivar,
                'growing_season': season,
                'gt_volume_ml': gt_volume,
                'pred_volume_ml': pred_volume,
                'exec_time_ms': round(elapsed_ms, 2),
            })

    df_out = pd.DataFrame(rows, columns=columns)
    valid = df_out.dropna(subset=['pred_volume_ml'])

    gt_arr = valid['gt_volume_ml'].to_numpy()
    pred_arr = valid['pred_volume_ml'].to_numpy()

    print(f'\nTest results ({len(valid)}/{len(df_out)} with valid meshes):')
    print(f'  MAE volume:  {mean_absolute_error(gt_arr, pred_arr):.2f} mL')
    print(f'  RMSE volume: {root_mean_squared_error(gt_arr, pred_arr):.2f} mL')
    print(f'  R²:          {r2_score(gt_arr, pred_arr):.3f}')
    print(f'  Avg exec:    {np.mean(exec_times):.1f} ms')

    # Per-cultivar breakdown
    if 'cultivar' in df_out.columns and df_out['cultivar'].notna().any():
        print('\n=== Per cultivar ===')
        for cultivar in valid['cultivar'].unique():
            sel = valid[valid['cultivar'] == cultivar]
            print(
                f'  {cultivar}: n={len(sel)} | '
                f'MAE={mean_absolute_error(sel["gt_volume_ml"], sel["pred_volume_ml"]):.2f} mL | '
                f'R²={r2_score(sel["gt_volume_ml"], sel["pred_volume_ml"]):.3f}'
            )

    # Per-season breakdown
    if 'growing_season' in df_out.columns and df_out['growing_season'].notna().any():
        print('\n=== Per growing season ===')
        for season in valid['growing_season'].unique():
            sel = valid[valid['growing_season'] == season]
            print(
                f'  {season}: n={len(sel)} | '
                f'MAE={mean_absolute_error(sel["gt_volume_ml"], sel["pred_volume_ml"]):.2f} mL | '
                f'R²={r2_score(sel["gt_volume_ml"], sel["pred_volume_ml"]):.3f}'
            )

    results_path = os.path.join(
        os.path.dirname(checkpoint_path), 'test_results.csv'
    )
    df_out.to_csv(results_path, index=False)
    print(f'\nResults saved to: {results_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 2: encoder evaluation')
    parser.add_argument('--config', '-c', required=True, help='Path to YAML encoder config')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint.pth')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg, args.checkpoint)
