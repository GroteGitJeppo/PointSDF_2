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
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import WeightedRandomSampler

from torch_geometric.loader import DataLoader
from torch_geometric.typing import WITH_TORCH_CLUSTER

from data.encoder_dataset import PointCloudLatentDataset, TuberBatchSampler
from models import PointNetEncoder, SDFDecoder

warnings.filterwarnings('ignore')

if not WITH_TORCH_CLUSTER:
    raise SystemExit("This code requires 'torch-cluster'")


# ---------------------------------------------------------------------------
# Contrastive loss (AttRepLoss) — ported from corepp/loss.py
# ---------------------------------------------------------------------------

def att_rep_loss(latents: torch.Tensor, labels: list, delta_rep: float = 0.5) -> torch.Tensor:
    """
    Attraction-repulsion contrastive loss (Magistri et al., 2022 / corepp).

    For each pair (i, j):
      - same label  → attract: penalise ||z_i - z_j||
      - diff label  → repel:   penalise max(0, delta_rep - ||z_i - z_j||)

    The raw sum is divided by B² so the return value is on the same scale
    regardless of batch size, and comparable to the per-element MSE loss.

    Args:
        latents:   (B, latent_size) predicted latent vectors
        labels:    list of B label strings (potato tuber IDs)
        delta_rep: repulsion margin (default 0.5, matching corepp)
    Returns:
        scalar loss (mean over all B² pairs)
    """
    B = latents.size(0)
    hinged = torch.nn.HingeEmbeddingLoss(margin=delta_rep, reduction='none')
    h_loss = torch.tensor(0.0, device=latents.device)
    for lbl, z in zip(labels, latents):
        dist = torch.linalg.norm(z - latents, dim=1)           # (B,)
        # +1 for same label (attract), -1 for different (repel)
        same = torch.tensor(
            [1 if l == lbl else -1 for l in labels],
            dtype=torch.float, device=latents.device,
        )
        h_loss = h_loss + hinged(dist, same).sum()
    return h_loss / (B * B)


# ---------------------------------------------------------------------------
# Training / validation steps
# ---------------------------------------------------------------------------

def train_epoch(
    encoder, decoder, optimizer, loader,
    sigma, sdf_loss_weight, device,
    contrastive: bool = False, lambda_attraction: float = 0.05, delta_rep: float = 0.5,
):
    encoder.train()
    total_loss = total_mse = total_reg = total_sdf = total_att = 0.0

    for batch in loader:
        batch = batch.to(device)
        latent_gt = batch.latent  # (B, latent_size), set by PointCloudLatentDataset

        pred_latent = encoder(batch)  # (B, latent_size)

        mse = F.mse_loss(pred_latent, latent_gt.detach())
        reg = sigma ** 2 * pred_latent.pow(2).sum(dim=1).mean()
        loss = mse + reg

        # Contrastive loss (matches corepp's AttRepLoss with lambda_attraction=0.05)
        att_l = torch.tensor(0.0, device=device)
        if contrastive and hasattr(batch, 'label'):
            att_l = att_rep_loss(pred_latent, batch.label, delta_rep)
            loss = loss + lambda_attraction * att_l

        # End-to-end SDF loss: run predicted latent through the frozen decoder
        # on the GT SDF query points, supervise with GT SDF values.
        # Gradients flow through the decoder input (latent) back to the encoder;
        # decoder weights are frozen so they do not update.
        sdf_l = torch.tensor(0.0, device=device)
        if sdf_loss_weight > 0.0 and batch.sdf_xyz is not None:
            sdf_xyz = batch.sdf_xyz              # (B, N_sdf, 3)
            sdf_gt_b = batch.sdf_gt              # (B, N_sdf, 1)
            B = pred_latent.size(0)
            N_sdf = sdf_xyz.size(1)
            lat_tiled = pred_latent.unsqueeze(1).expand(-1, N_sdf, -1).reshape(B * N_sdf, -1)
            xyz_flat = sdf_xyz.reshape(B * N_sdf, 3)
            pred_sdf = decoder(torch.cat([lat_tiled, xyz_flat], dim=1))  # (B*N_sdf, 1)
            sdf_l = F.l1_loss(pred_sdf, sdf_gt_b.reshape(B * N_sdf, 1))
            loss = loss + sdf_loss_weight * sdf_l

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mse += mse.item()
        total_reg += reg.item()
        total_sdf += sdf_l.item()
        total_att += att_l.item()

    n = len(loader)
    return total_loss / n, total_mse / n, total_reg / n, total_sdf / n, total_att / n


@torch.no_grad()
def val_epoch(
    encoder, decoder, loader,
    sigma, sdf_loss_weight, device,
    contrastive: bool = False, lambda_attraction: float = 0.05, delta_rep: float = 0.5,
):
    encoder.eval()
    total_loss = total_mse = total_reg = total_sdf = total_att = 0.0

    for batch in loader:
        batch = batch.to(device)
        latent_gt = batch.latent

        pred_latent = encoder(batch)

        mse = F.mse_loss(pred_latent, latent_gt)
        reg = sigma ** 2 * pred_latent.pow(2).sum(dim=1).mean()
        loss = mse + reg

        att_l = torch.tensor(0.0, device=device)
        if contrastive and hasattr(batch, 'label'):
            att_l = att_rep_loss(pred_latent, batch.label, delta_rep)
            loss = loss + lambda_attraction * att_l

        sdf_l = torch.tensor(0.0, device=device)
        if sdf_loss_weight > 0.0 and batch.sdf_xyz is not None:
            sdf_xyz = batch.sdf_xyz
            sdf_gt_b = batch.sdf_gt
            B = pred_latent.size(0)
            N_sdf = sdf_xyz.size(1)
            lat_tiled = pred_latent.unsqueeze(1).expand(-1, N_sdf, -1).reshape(B * N_sdf, -1)
            xyz_flat = sdf_xyz.reshape(B * N_sdf, 3)
            pred_sdf = decoder(torch.cat([lat_tiled, xyz_flat], dim=1))
            sdf_l = F.l1_loss(pred_sdf, sdf_gt_b.reshape(B * N_sdf, 1))
            loss = loss + sdf_loss_weight * sdf_l

        total_loss += loss.item()
        total_mse += mse.item()
        total_reg += reg.item()
        total_sdf += sdf_l.item()
        total_att += att_l.item()

    n = len(loader)
    return total_loss / n, total_mse / n, total_reg / n, total_sdf / n, total_att / n


# ---------------------------------------------------------------------------
# Sampling helpers (morphology-balanced training)
# ---------------------------------------------------------------------------

def _make_sample_weights(
    sample_labels: list[str],
    target_csv: str | None,
    trait_col: str = "volume (cm3)",
    bin_edges: list[float] | None = None,
    balance_cultivar: bool = False,
    metadata_csv: str | None = None,
) -> tuple[torch.Tensor, list[str], dict]:
    """
    Build inverse-frequency sample weights using fixed-width volume bins.

    Mirrors the PointRAFT approach: fixed 50 ml boundaries (0, 50, 100, …, 450, inf)
    produce 10 classes whose sampling probability is inversely proportional to how
    many training samples fall in each class.

    Returns:
        weights: (N_samples,) float tensor
        keys:    per-sample key used for balancing (e.g., '100-150|Kitahime')
        stats:   summary dict for logging
    """
    # Default: 50 ml-wide bins matching PointRAFT's weight classes
    if bin_edges is None:
        bin_edges = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, float("inf")]

    n = len(sample_labels)
    if n == 0 or not target_csv:
        return torch.ones(n, dtype=torch.double), ["all"] * n, {"mode": "uniform"}

    target_df = pd.read_csv(target_csv)
    if "label" not in target_df.columns or trait_col not in target_df.columns:
        return torch.ones(n, dtype=torch.double), ["all"] * n, {
            "mode": "uniform",
            "reason": f"missing required columns in target_csv: label + {trait_col}",
        }
    target_df = target_df.set_index("label")

    unique_labels = sorted(set(sample_labels))
    vals = pd.to_numeric(
        target_df.reindex(unique_labels)[trait_col], errors="coerce"
    )

    edges = sorted(bin_edges)
    bin_labels = [
        f"{int(edges[i])}-{int(edges[i+1])}" if edges[i+1] != float("inf")
        else f"{int(edges[i])}+"
        for i in range(len(edges) - 1)
    ]

    label_to_bin: dict[str, str] = {}
    for lbl in unique_labels:
        v = vals.get(lbl, float("nan"))
        if pd.isna(v):
            label_to_bin[lbl] = "bin_missing"
            continue
        assigned = False
        for i in range(len(edges) - 1):
            if edges[i] <= v < edges[i + 1]:
                label_to_bin[lbl] = bin_labels[i]
                assigned = True
                break
        if not assigned:
            # Value exactly equals the last edge or exceeds it
            label_to_bin[lbl] = bin_labels[-1]

    label_to_cultivar: dict[str, str] = {lbl: "cultivar_unknown" for lbl in unique_labels}
    if balance_cultivar:
        if "cultivar" in target_df.columns:
            c = target_df.reindex(unique_labels)["cultivar"].fillna("unknown").astype(str)
            label_to_cultivar.update({lbl: f"cultivar_{v}" for lbl, v in c.items()})
        elif metadata_csv:
            meta_df = pd.read_csv(metadata_csv)
            if "label" in meta_df.columns and "cultivar" in meta_df.columns:
                meta_df = meta_df.set_index("label")
                c = meta_df.reindex(unique_labels)["cultivar"].fillna("unknown").astype(str)
                label_to_cultivar.update({lbl: f"cultivar_{v}" for lbl, v in c.items()})

    keys = []
    for lbl in sample_labels:
        bin_key = label_to_bin.get(lbl, "bin_missing")
        if balance_cultivar:
            cult_key = label_to_cultivar.get(lbl, "cultivar_unknown")
            keys.append(f"{bin_key}|{cult_key}")
        else:
            keys.append(bin_key)

    counts = Counter(keys)
    weights = torch.tensor(
        [1.0 / max(counts[k], 1) for k in keys], dtype=torch.double
    )

    stats = {
        "mode": "fixed_width_bins",
        "trait_col": trait_col,
        "bin_edges": edges,
        "num_bins_effective": len({v for v in label_to_bin.values() if v != "bin_missing"}),
        "balance_cultivar": bool(balance_cultivar),
        "group_counts": dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True)),
    }
    return weights, keys, stats


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
    print("Evaluation protocol: 2025 is strict blind test-only. "
          "Checkpoint selection uses train/val only.")

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
    _ckpt = torch.load(cfg['decoder_weights'], map_location=device)
    _sd = _ckpt['model_state_dict'] if 'model_state_dict' in _ckpt else _ckpt
    _sd = {k.removeprefix('module.'): v for k, v in _sd.items()}
    decoder.load_state_dict(_sd)
    for p in decoder.parameters():
        p.requires_grad_(False)
    decoder.eval()
    print(f'Loaded frozen decoder from {cfg["decoder_weights"]}')

    # ----- Datasets -----
    sdf_data_dir = cfg.get('sdf_data_dir', None)
    sdf_samples = int(cfg.get('sdf_samples_per_shape', 1024))
    sdf_clamp = cfg.get('sdf_clamp_value', None)

    use_augmentation = bool(cfg.get('augmentation_enabled', True))
    if not use_augmentation:
        print('Augmentation disabled (augmentation_enabled: false).')

    normalize_half_extent = float(cfg.get('normalize_half_extent', 0.05))

    train_ds = PointCloudLatentDataset(
        data_root=cfg['data_root'],
        splits_csv=cfg['splits_csv'],
        latent_dir=cfg['latent_dir'],
        split='train',
        num_points=cfg.get('num_points', 1024),
        apply_augmentation=use_augmentation,
        augmentation_cfg=cfg.get('augmentation', None),
        sdf_data_dir=sdf_data_dir,
        sdf_samples_per_shape=sdf_samples,
        sdf_clamp_value=sdf_clamp,
        normalize_half_extent=normalize_half_extent,
    )
    val_ds = PointCloudLatentDataset(
        data_root=cfg['data_root'],
        splits_csv=cfg['splits_csv'],
        latent_dir=cfg['latent_dir'],
        split='val',
        num_points=cfg.get('num_points', 1024),
        apply_augmentation=False,
        sdf_data_dir=sdf_data_dir,
        sdf_samples_per_shape=sdf_samples,
        sdf_clamp_value=sdf_clamp,
        normalize_half_extent=normalize_half_extent,
    )

    # ----- Sampler selection (mutually exclusive) -----
    # Priority: tuber_sampler > weighted sampler > plain shuffle.
    tuber_cfg = cfg.get("tuber_sampler", {})
    use_tuber_sampler = bool(tuber_cfg.get("enabled", False))

    sampler_cfg = cfg.get("sampler", {})
    use_weighted_sampler = bool(sampler_cfg.get("enabled", False)) and not use_tuber_sampler

    if use_tuber_sampler and use_weighted_sampler:
        print("WARNING: tuber_sampler and sampler are both enabled — tuber_sampler takes priority.")

    if use_tuber_sampler:
        k_scans = int(tuber_cfg.get("k_scans", 2))
        n_labels = int(tuber_cfg.get("n_labels", 8))
        label_to_indices = train_ds.get_label_to_indices()
        batch_sampler = TuberBatchSampler(
            label_to_indices=label_to_indices,
            n_labels=n_labels,
            k_scans=k_scans,
            drop_last=False,
        )
        print(
            f"TuberBatchSampler enabled: {n_labels} labels × {k_scans} scans "
            f"= {batch_sampler.batch_size} per batch | "
            f"{len(batch_sampler)} batches/epoch over {len(batch_sampler._labels)} labels"
        )
        train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=4)
    else:
        train_sampler = None
        if use_weighted_sampler:
            sample_labels = [lbl for _, lbl, _ in train_ds.samples]
            sample_weights, _, sampler_stats = _make_sample_weights(
                sample_labels=sample_labels,
                target_csv=cfg.get("target_csv", None),
                trait_col=sampler_cfg.get("trait_column", cfg.get("volume_column", "volume (cm3)")),
                bin_edges=sampler_cfg.get("bin_edges", None),
                balance_cultivar=bool(sampler_cfg.get("balance_cultivar", False)),
                metadata_csv=cfg.get("metadata_csv", None),
            )
            train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            print("Weighted sampler enabled:")
            print(f"  trait={sampler_stats.get('trait_col')} | bins={sampler_stats.get('num_bins_effective')}")
            print(f"  group counts: {sampler_stats.get('group_counts')}")
        else:
            print("Weighted sampler disabled; using standard random shuffle.")

        train_loader = DataLoader(
            train_ds, batch_size=cfg.get('batch_size', 16),
            shuffle=(train_sampler is None), sampler=train_sampler, num_workers=4,
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
    total_epochs = cfg.get('epochs', 100)
    # lr_gamma is the per-epoch decay factor calibrated for 100 epochs.
    # The actual per-epoch gamma is scaled so the total decay is the same
    # regardless of how many epochs are configured: gamma_actual = lr_gamma^(100/epochs).
    _gamma = float(cfg.get('lr_gamma', 0.97)) ** (100.0 / max(total_epochs, 1))
    print(f'LR scheduler: ExponentialLR gamma={_gamma:.6f} '
          f'(LR {cfg.get("lr", 0.001):.2e} → {cfg.get("lr", 0.001) * _gamma**total_epochs:.2e} over {total_epochs} epochs)')
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=_gamma)

    sigma = cfg.get('sigma_regulariser', 0.01)
    sdf_loss_weight = float(cfg.get('sdf_loss_weight', 0.0))
    if sdf_loss_weight > 0.0 and sdf_data_dir is None:
        print('WARNING: sdf_loss_weight > 0 but sdf_data_dir is not set — SDF loss disabled.')
        sdf_loss_weight = 0.0

    use_contrastive = bool(cfg.get('contrastive_loss', False))
    lambda_attraction = float(cfg.get('lambda_attraction', 0.05))
    delta_rep = float(cfg.get('delta_rep', 0.5))
    if use_contrastive:
        print(f'Contrastive loss (AttRepLoss) enabled — lambda={lambda_attraction}, delta_rep={delta_rep}')

    # ----- Optional resume from snapshot checkpoint -----
    resume_path = cfg.get('_resume_checkpoint', None)
    start_epoch = 0
    best_val_loss = float('inf')
    if resume_path:
        print(f'Resuming from checkpoint: {resume_path}')
        ckpt = torch.load(resume_path, map_location=device)
        encoder.load_state_dict(ckpt['encoder_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = int(ckpt.get('epoch', 0))
        best_val_loss = float(ckpt.get('val_loss', float('inf')))
        # Restore scheduler state: step() was called start_epoch times already
        for _ in range(start_epoch):
            scheduler.step()
        print(f'  Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.5f}')

    # ----- Training loop -----
    snapshot_freq = int(cfg.get('snapshot_frequency', 10))
    snapshots_dir = os.path.join(output_dir, 'snapshots')
    os.makedirs(snapshots_dir, exist_ok=True)

    for epoch in range(start_epoch + 1, cfg.get('epochs', 100) + 1):
        t0 = time.time()
        train_loss, train_mse, train_reg, train_sdf, train_att = train_epoch(
            encoder, decoder, optimizer, train_loader, sigma, sdf_loss_weight, device,
            contrastive=use_contrastive, lambda_attraction=lambda_attraction, delta_rep=delta_rep,
        )
        val_loss, val_mse, val_reg, val_sdf, val_att = val_epoch(
            encoder, decoder, val_loader, sigma, sdf_loss_weight, device,
            contrastive=use_contrastive, lambda_attraction=lambda_attraction, delta_rep=delta_rep,
        )
        elapsed = time.time() - t0

        sdf_str = f' sdf={train_sdf:.5f}|{val_sdf:.5f}' if sdf_loss_weight > 0.0 else ''
        att_str = f' att={train_att:.5f}|{val_att:.5f}' if use_contrastive else ''
        print(
            f'Epoch {epoch:03d}/{cfg.get("epochs", 100)} | '
            f'train {train_loss:.5f} (mse={train_mse:.5f}) | '
            f'val {val_loss:.5f} (mse={val_mse:.5f}) |'
            f'{sdf_str}{att_str} {elapsed:.1f}s'
        )

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/train_mse', train_mse, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Loss/val_mse', val_mse, epoch)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
        if sdf_loss_weight > 0.0:
            writer.add_scalar('Loss/train_sdf', train_sdf, epoch)
            writer.add_scalar('Loss/val_sdf', val_sdf, epoch)
        if use_contrastive:
            writer.add_scalar('Loss/train_att', train_att, epoch)
            writer.add_scalar('Loss/val_att', val_att, epoch)

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

        # Periodic snapshots — run test.py on these after training to pick
        # the checkpoint with the best volume RMSE (paper's key selection criterion).
        if snapshot_freq > 0 and epoch % snapshot_freq == 0:
            snap_dir = os.path.join(snapshots_dir, f'{epoch:04d}')
            os.makedirs(snap_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(snap_dir, 'checkpoint.pth'))

        scheduler.step()

    writer.close()
    print(f'\nTraining complete. Best val loss: {best_val_loss:.5f}')
    print(f'Outputs in: {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 2: encoder training')
    parser.add_argument('--config', '-c', required=True, help='Path to YAML config file')
    parser.add_argument(
        '--resume', dest='resume_checkpoint', default=None, metavar='CHECKPOINT_PATH',
        help='Path to a snapshot checkpoint.pth to resume training from '
             '(e.g. weights/encoder/<run>/snapshots/0090/checkpoint.pth)',
    )
    parser.add_argument(
        '--epochs', dest='epochs', type=int, default=None,
        help='Override the total number of epochs in the config',
    )

    # Per-run overrides — set these in your SLURM script instead of editing the YAML.
    # Each flag overrides the corresponding key in the config when provided.
    # Usage: --augmentation true  /  --augmentation false
    def str2bool(v):
        if v.lower() in ('true', '1', 'yes'):
            return True
        if v.lower() in ('false', '0', 'no'):
            return False
        raise argparse.ArgumentTypeError(f"Expected true/false, got: {v!r}")

    parser.add_argument(
        '--augmentation', dest='augmentation_enabled', type=str2bool, default=None,
        metavar='BOOL',
        help='Override augmentation_enabled in config (true/false)',
    )
    parser.add_argument(
        '--sampler', dest='sampler', default=None,
        choices=['weighted', 'tuber', 'none'],
        help=(
            'Override which sampler to use: '
            '"weighted" (volume-bin balanced), '
            '"tuber" (guarantees same-tuber pairs per batch), '
            '"none" (plain random shuffle)'
        ),
    )
    parser.add_argument(
        '--contrastive-loss', dest='contrastive_loss', type=str2bool, default=None,
        metavar='BOOL',
        help='Override contrastive_loss in config (true/false)',
    )

    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.augmentation_enabled is not None:
        cfg['augmentation_enabled'] = args.augmentation_enabled
    if args.sampler is not None:
        cfg.setdefault('sampler', {})['enabled'] = (args.sampler == 'weighted')
        cfg.setdefault('tuber_sampler', {})['enabled'] = (args.sampler == 'tuber')
    if args.contrastive_loss is not None:
        cfg['contrastive_loss'] = args.contrastive_loss
    if args.resume_checkpoint is not None:
        cfg['_resume_checkpoint'] = args.resume_checkpoint
    if args.epochs is not None:
        cfg['epochs'] = args.epochs

    main(cfg)
