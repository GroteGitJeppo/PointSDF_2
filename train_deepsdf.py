"""
Stage 1 — DeepSDF autodecoder training.

Trains the SDF decoder jointly with per-shape latent codes on complete
laser-scan SDF samples.  After training, saves:
  <output_dir>/
    decoder.pth            — best decoder weights
    latent_codes/          — one <label>.pth tensor per potato
    optimizer_model.pt     — model optimiser state (for resuming)
    optimizer_latent.pt    — latent optimiser state (for resuming)
    config.yaml            — copy of the run config
    events.out.tfevents.*  — TensorBoard logs

Usage:
    python train_deepsdf.py --config configs/train_deepsdf.yaml
"""

import argparse
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.sdf_samples import SDFSamplesDataset
from models import SDFDecoder
from utils import sdf_loss


# ---------------------------------------------------------------------------
# Training / validation helpers
# ---------------------------------------------------------------------------

def train_epoch(model, latent_codes, optimizer_model, optimizer_latent, loader, cfg, device):
    model.train()
    total_loss = total_l1 = total_l2 = 0.0

    for xyz_latent, sdf_gt in loader:
        xyz_latent = xyz_latent.to(device)
        sdf_gt = sdf_gt.to(device)

        latent_idx = xyz_latent[:, 0].long()
        xyz = xyz_latent[:, 1:]

        latent_batch = latent_codes[latent_idx]
        decoder_input = torch.cat([latent_batch, xyz], dim=1)

        optimizer_model.zero_grad()
        optimizer_latent.zero_grad()

        pred = model(decoder_input)
        if cfg['clamp']:
            pred = torch.clamp(pred, -cfg['clamp_value'], cfg['clamp_value'])

        loss, l1, l2 = sdf_loss(pred, sdf_gt, latent_batch, sigma=cfg['sigma_regulariser'])
        loss = loss * cfg['loss_multiplier']
        loss.backward()

        optimizer_model.step()
        optimizer_latent.step()

        total_loss += loss.item()
        total_l1 += l1.item()
        total_l2 += l2.item()

    n = len(loader)
    return total_loss / n, total_l1 / n, total_l2 / n


@torch.no_grad()
def val_epoch(model, latent_codes, loader, cfg, device):
    model.eval()
    total_loss = total_l1 = total_l2 = 0.0

    for xyz_latent, sdf_gt in loader:
        xyz_latent = xyz_latent.to(device)
        sdf_gt = sdf_gt.to(device)

        latent_idx = xyz_latent[:, 0].long()
        xyz = xyz_latent[:, 1:]

        latent_batch = latent_codes[latent_idx]
        decoder_input = torch.cat([latent_batch, xyz], dim=1)

        pred = model(decoder_input)
        if cfg['clamp']:
            pred = torch.clamp(pred, -cfg['clamp_value'], cfg['clamp_value'])

        loss, l1, l2 = sdf_loss(pred, sdf_gt, latent_batch, sigma=cfg['sigma_regulariser'])
        total_loss += loss.item()
        total_l1 += l1.item()
        total_l2 += l2.item()

    n = len(loader)
    return total_loss / n, total_l1 / n, total_l2 / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: dict):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    torch.manual_seed(cfg.get('seed', 42))
    np.random.seed(cfg.get('seed', 42))

    run_tag = datetime.now().strftime('%d_%m_%H%M%S')
    output_dir = os.path.join(cfg['output_dir'], run_tag)
    latent_save_dir = os.path.join(output_dir, 'latent_codes')
    os.makedirs(latent_save_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f)
    writer = SummaryWriter(log_dir=output_dir)

    # ----- Datasets -----
    # Train latent codes on train+val shapes; hold out test shapes for Stage 2 evaluation.
    clamp_val = cfg['clamp_value'] if cfg['clamp'] else None
    train_ds = SDFSamplesDataset(
        cfg['sdf_data_dir'], cfg['splits_csv'],
        split='train', clamp_value=clamp_val,
    )
    val_ds = SDFSamplesDataset(
        cfg['sdf_data_dir'], cfg['splits_csv'],
        split='val', clamp_value=clamp_val,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg['batch_size'],
        shuffle=True, drop_last=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg['batch_size'],
        shuffle=False, drop_last=False, num_workers=4, pin_memory=True,
    )

    # ----- Model -----
    model = SDFDecoder(
        latent_size=cfg['latent_size'],
        num_layers=cfg['num_layers'],
        inner_dim=cfg['inner_dim'],
        skip_connections=cfg['skip_connections'],
    ).float().to(device)

    # One learnable latent code per training shape, init ~ N(0, 0.01)
    num_train_shapes = train_ds.num_shapes
    latent_codes = torch.normal(
        0, 0.01, size=(num_train_shapes, cfg['latent_size']),
    ).float().to(device).requires_grad_(True)

    optimizer_model = optim.Adam(model.parameters(), lr=cfg['lr_model'])
    optimizer_latent = optim.Adam([latent_codes], lr=cfg['lr_latent'])

    if cfg.get('lr_scheduler', False):
        scheduler_model = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_model, mode='min',
            factor=cfg['lr_multiplier'], patience=cfg['patience'],
        )
        scheduler_latent = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_latent, mode='min',
            factor=cfg['lr_multiplier'], patience=cfg['patience'],
        )

    # ----- Resume -----
    if cfg.get('pretrained', False) and cfg.get('pretrain_dir', ''):
        d = cfg['pretrain_dir']
        model.load_state_dict(torch.load(os.path.join(d, 'decoder.pth'), map_location=device))
        optimizer_model.load_state_dict(
            torch.load(os.path.join(d, 'optimizer_model.pt'), map_location=device)
        )
        optimizer_latent.load_state_dict(
            torch.load(os.path.join(d, 'optimizer_latent.pt'), map_location=device)
        )
        print(f'Resumed from {d}')

    # ----- Training loop -----
    best_val_loss = float('inf')
    for epoch in range(cfg['epochs']):
        t0 = time.time()
        train_loss, train_l1, train_l2 = train_epoch(
            model, latent_codes, optimizer_model, optimizer_latent,
            train_loader, cfg, device,
        )
        val_loss, val_l1, val_l2 = val_epoch(
            model, latent_codes, val_loader, cfg, device,
        )
        elapsed = time.time() - t0

        print(
            f'Epoch {epoch+1:04d}/{cfg["epochs"]} | '
            f'train {train_loss:.5f} (l1={train_l1:.5f} l2={train_l2:.5f}) | '
            f'val {val_loss:.5f} | {elapsed:.1f}s'
        )

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/train_l1', train_l1, epoch)
        writer.add_scalar('Loss/train_l2', train_l2, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)

        if cfg.get('lr_scheduler', False):
            scheduler_model.step(val_loss)
            scheduler_latent.step(val_loss)
            writer.add_scalar('LR/model', scheduler_model.get_last_lr()[0], epoch)
            writer.add_scalar('LR/latent', scheduler_latent.get_last_lr()[0], epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'decoder.pth'))
            torch.save(optimizer_model.state_dict(), os.path.join(output_dir, 'optimizer_model.pt'))
            torch.save(optimizer_latent.state_dict(), os.path.join(output_dir, 'optimizer_latent.pt'))

            best_codes = latent_codes.detach().cpu().clone()
            for label, idx in train_ds.label_to_idx.items():
                torch.save(best_codes[idx], os.path.join(latent_save_dir, f'{label}.pth'))

            print(f'  Saved best model (val {best_val_loss:.5f})')

    writer.close()
    print(f'\nTraining complete. Outputs in: {output_dir}')
    print(f'Best val loss: {best_val_loss:.5f}')
    print(f'Latent codes saved to: {latent_save_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 1: DeepSDF autodecoder training')
    parser.add_argument('--config', '-c', required=True, help='Path to YAML config file')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg)
