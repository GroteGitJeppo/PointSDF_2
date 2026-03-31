"""
Stage 2 — PointNet++ encoder training.

Trains the encoder to predict latent codes from partial point clouds,
supervised by the Stage 1 latent codes stored on disk.
The Stage 1 decoder is loaded but kept frozen.

Saves:
  <output_dir>/
    encoder.pth            — best encoder weights
    checkpoint.pth         — full checkpoint (encoder + decoder state dicts)
    config.yaml
    events.out.tfevents.*

Usage:
    python train.py --config configs/train_encoder.yaml
"""

import argparse
import os
import random
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.loader import DataLoader
from torch_geometric.typing import WITH_TORCH_CLUSTER

from data.encoder_dataset import PointCloudLatentDataset
from models import PointNetEncoder, SDFDecoder

warnings.filterwarnings('ignore')

if not WITH_TORCH_CLUSTER:
    raise SystemExit("This code requires 'torch-cluster'")


# ---------------------------------------------------------------------------
# Training / validation steps
# ---------------------------------------------------------------------------

def train_epoch(encoder, optimizer, loader, sigma, device):
    encoder.train()
    total_loss = total_mse = total_reg = 0.0

    for batch in loader:
        batch = batch.to(device)
        latent_gt = batch.latent  # (B, latent_size), set by PointCloudLatentDataset

        pred_latent = encoder(batch)  # (B, latent_size)

        mse = F.mse_loss(pred_latent, latent_gt.detach())
        reg = sigma ** 2 * pred_latent.norm(dim=1).mean()
        loss = mse + reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mse += mse.item()
        total_reg += reg.item()

    n = len(loader)
    return total_loss / n, total_mse / n, total_reg / n


@torch.no_grad()
def val_epoch(encoder, loader, sigma, device):
    encoder.eval()
    total_loss = total_mse = total_reg = 0.0

    for batch in loader:
        batch = batch.to(device)
        latent_gt = batch.latent

        pred_latent = encoder(batch)

        mse = F.mse_loss(pred_latent, latent_gt)
        reg = sigma ** 2 * pred_latent.norm(dim=1).mean()
        loss = mse + reg

        total_loss += loss.item()
        total_mse += mse.item()
        total_reg += reg.item()

    n = len(loader)
    return total_loss / n, total_mse / n, total_reg / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: dict):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    torch.manual_seed(cfg.get('seed', 133))
    random.seed(cfg.get('seed', 133))
    np.random.seed(cfg.get('seed', 133))

    run_tag = datetime.now().strftime('%d_%m_%H%M%S')
    output_dir = os.path.join(cfg['output_dir'], run_tag)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f)
    writer = SummaryWriter(log_dir=output_dir)

    # ----- Load Stage 1 decoder (frozen) -----
    decoder_cfg_path = cfg['decoder_config']
    with open(decoder_cfg_path) as f:
        decoder_cfg = yaml.safe_load(f)

    decoder = SDFDecoder(
        latent_size=decoder_cfg['latent_size'],
        num_layers=decoder_cfg['num_layers'],
        inner_dim=decoder_cfg['inner_dim'],
        skip_connections=decoder_cfg['skip_connections'],
    ).float().to(device)
    decoder.load_state_dict(torch.load(cfg['decoder_weights'], map_location=device))
    for p in decoder.parameters():
        p.requires_grad_(False)
    decoder.eval()
    print(f'Loaded frozen decoder from {cfg["decoder_weights"]}')

    # ----- Datasets -----
    train_ds = PointCloudLatentDataset(
        data_root=cfg['data_root'],
        splits_csv=cfg['splits_csv'],
        latent_dir=cfg['latent_dir'],
        split='train',
        num_points=cfg.get('num_points', 1024),
        apply_augmentation=True,
    )
    val_ds = PointCloudLatentDataset(
        data_root=cfg['data_root'],
        splits_csv=cfg['splits_csv'],
        latent_dir=cfg['latent_dir'],
        split='val',
        num_points=cfg.get('num_points', 1024),
        apply_augmentation=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.get('batch_size', 16),
        shuffle=True, num_workers=4,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.get('batch_size', 16),
        shuffle=False, num_workers=4,
    )

    # ----- Encoder -----
    encoder = PointNetEncoder(latent_size=decoder_cfg['latent_size']).to(device)

    optimizer = optim.Adam(
        encoder.parameters(),
        lr=cfg.get('lr', 0.001),
        weight_decay=cfg.get('weight_decay', 1e-4),
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.get('lr_gamma', 0.97))

    sigma = cfg.get('sigma_regulariser', 0.01)

    # ----- Training loop -----
    best_val_loss = float('inf')
    for epoch in range(1, cfg.get('epochs', 100) + 1):
        t0 = time.time()
        train_loss, train_mse, train_reg = train_epoch(encoder, optimizer, train_loader, sigma, device)
        val_loss, val_mse, val_reg = val_epoch(encoder, val_loader, sigma, device)
        elapsed = time.time() - t0

        print(
            f'Epoch {epoch:03d}/{cfg.get("epochs", 100)} | '
            f'train {train_loss:.5f} (mse={train_mse:.5f}) | '
            f'val {val_loss:.5f} (mse={val_mse:.5f}) | '
            f'{elapsed:.1f}s'
        )

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/train_mse', train_mse, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Loss/val_mse', val_mse, epoch)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(encoder.state_dict(), os.path.join(output_dir, 'encoder.pth'))
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(output_dir, 'checkpoint.pth'))
            print(f'  Saved best model (val {best_val_loss:.5f})')

        scheduler.step()

    writer.close()
    print(f'\nTraining complete. Best val loss: {best_val_loss:.5f}')
    print(f'Outputs in: {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 2: encoder training')
    parser.add_argument('--config', '-c', required=True, help='Path to YAML config file')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg)
