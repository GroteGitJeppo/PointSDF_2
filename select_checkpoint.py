"""
Post-training checkpoint selection by validation-set volume RMSE.

Iterates over all periodic snapshots saved by train.py, runs the full
encode → SDF-grid decode → convex-hull mesh → volume pipeline on the
validation split, and selects the epoch with the lowest volume RMSE.

The best checkpoint is copied to <run_dir>/best_vol/checkpoint.pth so
test.py can be pointed at it directly.  The test split is never touched.

Usage:
    python select_checkpoint.py \\
        --config configs/train_encoder.yaml \\
        --run_dir weights/encoder/<run>

Optional flags:
    --split val          (default) split to evaluate on
    --also_best_mse      also copy the MSE-best checkpoint.pth from
                         run_dir into best_vol/ for side-by-side comparison
"""

import argparse
import shutil
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


# ---------------------------------------------------------------------------
# Helpers (identical to test.py)
# ---------------------------------------------------------------------------

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


@torch.no_grad()
def evaluate_checkpoint(
    encoder: PointNetEncoder,
    decoder: SDFDecoder,
    ply_files: list[str],
    gt_df: pd.DataFrame,
    volume_col: str,
    num_points: int,
    grid_coords: torch.Tensor,
    pre_transform,
    device,
) -> tuple[float, float, float, int, int]:
    """
    Run the full pipeline on a list of PLY files and return volume metrics.

    Returns:
        rmse, mae, r2, n_valid, n_failed
    """
    encoder.eval()
    gt_volumes: list[float] = []
    pred_volumes: list[float] = []
    n_failed = 0

    for ply_file in ply_files:
        unique_id = Path(ply_file).parent.name

        if unique_id not in gt_df.index:
            continue

        gt_volume = float(gt_df.loc[unique_id, volume_col])

        data = process_ply(ply_file, num_points, pre_transform, device)
        latent = encoder(data)                              # (1, latent_size)

        latent_tiled = latent.expand(grid_coords.size(0), -1)
        decoder_input = torch.cat([latent_tiled, grid_coords], dim=1)
        pred_sdf = decoder(decoder_input)

        try:
            mesh = sdf2mesh(pred_sdf, grid_coords)
            if mesh.is_watertight():
                pred_volume = mesh.get_volume() * 1e6      # m³ → mL
                gt_volumes.append(gt_volume)
                pred_volumes.append(pred_volume)
            else:
                n_failed += 1
        except (ValueError, RuntimeError):
            n_failed += 1

    n_valid = len(gt_volumes)
    if n_valid < 2:
        return float('nan'), float('nan'), float('nan'), n_valid, n_failed

    gt_arr = np.array(gt_volumes)
    pred_arr = np.array(pred_volumes)
    rmse = float(root_mean_squared_error(gt_arr, pred_arr))
    mae = float(mean_absolute_error(gt_arr, pred_arr))
    r2 = float(r2_score(gt_arr, pred_arr))
    return rmse, mae, r2, n_valid, n_failed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: dict, run_dir: str, split: str, also_best_mse: bool):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ----- Architecture -----
    with open(cfg['decoder_config']) as f:
        decoder_cfg = yaml.safe_load(f)
    latent_size = decoder_cfg['latent_size']

    # ----- Decoder (loaded once, frozen) -----
    decoder = SDFDecoder(
        latent_size=latent_size,
        num_layers=decoder_cfg['num_layers'],
        inner_dim=decoder_cfg['inner_dim'],
        skip_connections=decoder_cfg['skip_connections'],
    ).float().to(device)
    _ckpt = torch.load(cfg['decoder_weights'], map_location=device)
    _sd = _ckpt['model_state_dict'] if 'model_state_dict' in _ckpt else _ckpt
    _sd = {k.removeprefix('module.'): v for k, v in _sd.items()}
    decoder.load_state_dict(_sd)
    for p in decoder.parameters():
        p.requires_grad_(False)
    decoder.eval()
    print(f'Loaded frozen decoder from {cfg["decoder_weights"]}')

    # ----- Val PLY files -----
    splits_df = pd.read_csv(cfg['splits_csv'])
    split_ids = set(splits_df.loc[splits_df['split'] == split, 'label'].astype(str))
    if not split_ids:
        raise ValueError(f"No labels found for split='{split}' in {cfg['splits_csv']}")

    all_files = list(Path(cfg['data_root']).rglob('*.ply'))
    ply_files = [str(f) for f in all_files if f.parent.name in split_ids]
    print(f'Split "{split}": {len(ply_files)} PLY files across {len(split_ids)} labels')

    volume_col = cfg.get('volume_column', 'volume_ml')
    gt_df = pd.read_csv(cfg['target_csv']).set_index('label')

    # ----- Grid coords (built once) -----
    grid_resolution = cfg.get('grid_resolution', 60)
    grid_bbox = cfg.get('grid_bbox', 0.15)
    # grid_center shifts the query grid from the origin to the position where the
    # complete laser scans live in the scanner coordinate frame.  Required when
    # the decoder was trained on uncentered data (e.g. corepp weights).
    grid_center = torch.tensor(
        cfg.get('grid_center', [0.0, 0.0, 0.0]), dtype=torch.float, device=device
    )
    grid_coords = get_volume_coords(resolution=grid_resolution, bbox=grid_bbox).to(device) + grid_center
    center_str = f'  center={grid_center.cpu().tolist()}' if float(grid_center.norm()) > 1e-6 else ''
    print(f'SDF grid: {grid_resolution}³ = {grid_coords.size(0):,} points  bbox=±{grid_bbox}m{center_str}')

    pre_transform = T.Center()
    num_points = cfg.get('num_points', 1024)

    # ----- Discover snapshots -----
    snapshots_dir = Path(run_dir) / 'snapshots'
    if not snapshots_dir.exists():
        raise FileNotFoundError(
            f"No snapshots directory found at {snapshots_dir}. "
            "Make sure snapshot_frequency > 0 was set during training."
        )

    snapshot_dirs = sorted(
        [d for d in snapshots_dir.iterdir() if d.is_dir() and (d / 'checkpoint.pth').exists()],
        key=lambda d: d.name,
    )
    if not snapshot_dirs:
        raise FileNotFoundError(f"No checkpoint.pth files found under {snapshots_dir}")

    print(f'Found {len(snapshot_dirs)} snapshot(s): epochs '
          f'{snapshot_dirs[0].name}–{snapshot_dirs[-1].name}\n')

    # ----- Sweep -----
    encoder = PointNetEncoder(latent_size=latent_size).to(device)
    results: list[dict] = []

    for snap_dir in snapshot_dirs:
        epoch = int(snap_dir.name)
        ckpt_path = snap_dir / 'checkpoint.pth'

        ckpt = torch.load(str(ckpt_path), map_location=device)
        encoder.load_state_dict(ckpt['encoder_state_dict'])

        rmse, mae, r2, n_valid, n_failed = evaluate_checkpoint(
            encoder, decoder, ply_files, gt_df, volume_col,
            num_points, grid_coords, pre_transform, device,
        )

        results.append({
            'epoch': epoch,
            'rmse_ml': round(rmse, 3) if not np.isnan(rmse) else float('nan'),
            'mae_ml':  round(mae, 3)  if not np.isnan(mae)  else float('nan'),
            'r2':      round(r2, 4)   if not np.isnan(r2)   else float('nan'),
            'n_valid': n_valid,
            'n_failed': n_failed,
            'checkpoint': str(ckpt_path),
        })

        status = f'RMSE={rmse:.2f} mL  MAE={mae:.2f} mL  R²={r2:.3f}  valid={n_valid}/{n_valid+n_failed}'
        print(f'Epoch {epoch:04d} | {status}')

    # ----- Rank and report -----
    df = pd.DataFrame(results)
    df_valid = df.dropna(subset=['rmse_ml'])

    if df_valid.empty:
        print('\nNo valid results — no watertight meshes produced for any checkpoint.')
        return

    df_sorted = df_valid.sort_values('rmse_ml').reset_index(drop=True)

    print(f'\n{"="*60}')
    print(f'Ranked by val volume RMSE (split={split}):')
    print(f'{"="*60}')
    print(df_sorted[['epoch', 'rmse_ml', 'mae_ml', 'r2', 'n_valid', 'n_failed']].to_string(index=False))

    best_row = df_sorted.iloc[0]
    best_epoch = int(best_row['epoch'])
    best_rmse = float(best_row['rmse_ml'])
    best_ckpt = best_row['checkpoint']

    print(f'\nBest checkpoint: epoch {best_epoch:04d}  RMSE={best_rmse:.3f} mL')

    # ----- Save CSV -----
    csv_path = Path(run_dir) / 'val_volume_selection.csv'
    df.to_csv(str(csv_path), index=False)
    print(f'Results saved to: {csv_path}')

    # ----- Copy best checkpoint -----
    best_vol_dir = Path(run_dir) / 'best_vol'
    best_vol_dir.mkdir(exist_ok=True)
    dest = best_vol_dir / 'checkpoint.pth'
    shutil.copy2(best_ckpt, str(dest))
    print(f'Best checkpoint copied to: {dest}')

    if also_best_mse:
        mse_src = Path(run_dir) / 'checkpoint.pth'
        if mse_src.exists():
            mse_dest = best_vol_dir / 'checkpoint_best_mse.pth'
            shutil.copy2(str(mse_src), str(mse_dest))
            print(f'MSE-best checkpoint copied to: {mse_dest}')
        else:
            print(f'WARNING: --also_best_mse requested but {mse_src} not found.')

    print(f'\nNext step:')
    print(f'  python test.py --config configs/train_encoder.yaml \\')
    print(f'      --checkpoint {dest}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Select best encoder checkpoint by validation-set volume RMSE.'
    )
    parser.add_argument('--config',  '-c', required=True, help='Path to train_encoder.yaml')
    parser.add_argument('--run_dir', '-r', required=True,
                        help='Path to the training run directory (e.g. weights/encoder/01_05_120000)')
    parser.add_argument('--split', default='val',
                        help='Dataset split to evaluate on (default: val)')
    parser.add_argument('--also_best_mse', action='store_true',
                        help='Also copy the MSE-best checkpoint for side-by-side comparison')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg, args.run_dir, args.split, args.also_best_mse)
