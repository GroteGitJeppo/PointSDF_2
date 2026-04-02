"""
Stage 1 — DeepSDF autodecoder training.

Trains the SDF decoder jointly with per-shape latent codes. One optimiser step
per shape per epoch, random subsample of SDF points per shape.

Saves under <output_dir>/<run_tag>/:
  decoder.pth, latent_embedding.pt, optimizer.pt, meta.pt
  latent_codes/<label>.pth
  snapshots/<epoch>/...  (every snapshot_frequency)
  latest/...             (every log_frequency)
  config.yaml, TensorBoard events

Usage:
    python train_deepsdf.py --config configs/train_deepsdf.yaml
"""

import argparse
import math
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter

from data.sdf_scene_dataset import SDFSceneDataset
from models import SDFDecoder
from utils import sdf_autodecoder_loss_chunk


# ---------------------------------------------------------------------------
# Learning rate schedules for decoder vs latent parameter groups
# ---------------------------------------------------------------------------


class ConstantLearningRateSchedule:
    def __init__(self, value):
        self.value = value

    def get_learning_rate(self, epoch):
        return self.value


class StepLearningRateSchedule:
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):
        return self.initial * (self.factor ** (epoch // self.interval))


class WarmupLearningRateSchedule:
    def __init__(self, initial, warmed_up, length):
        self.initial = initial
        self.warmed_up = warmed_up
        self.length = length

    def get_learning_rate(self, epoch):
        if epoch > self.length:
            return self.warmed_up
        return self.initial + (self.warmed_up - self.initial) * epoch / self.length


def build_lr_schedules(spec_list: list) -> list:
    schedules = []
    for s in spec_list:
        t = s['Type']
        if t == 'Step':
            schedules.append(
                StepLearningRateSchedule(
                    s['Initial'], s['Interval'], s['Factor'],
                )
            )
        elif t == 'Warmup':
            schedules.append(
                WarmupLearningRateSchedule(
                    s['Initial'], s['Final'], s['Length'],
                )
            )
        elif t == 'Constant':
            schedules.append(ConstantLearningRateSchedule(s['Value']))
        else:
            raise ValueError(f'Unknown LR schedule Type: {t}')
    return schedules


def adjust_learning_rate(schedules, optimizer, epoch):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = schedules[i].get_learning_rate(epoch)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def save_run_checkpoint(
    run_dir: str,
    subdir: str,
    model: torch.nn.Module,
    lat_vecs: torch.nn.Embedding,
    optimizer: optim.Optimizer,
    epoch: int,
    label_to_idx: dict,
):
    root = os.path.join(run_dir, subdir)
    latent_dir = os.path.join(root, 'latent_codes')
    os.makedirs(latent_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(root, 'decoder.pth'))
    torch.save(lat_vecs.state_dict(), os.path.join(root, 'latent_embedding.pt'))
    torch.save(optimizer.state_dict(), os.path.join(root, 'optimizer.pt'))
    torch.save({'epoch': epoch}, os.path.join(root, 'meta.pt'))

    for label, idx in label_to_idx.items():
        torch.save(lat_vecs.weight.data[idx].cpu(), os.path.join(latent_dir, f'{label}.pth'))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_epoch(
    model: torch.nn.Module,
    lat_vecs: torch.nn.Embedding,
    optimizer: optim.Optimizer,
    scene_ds: SDFSceneDataset,
    cfg: dict,
    device: torch.device,
    epoch_1based: int,
    grad_clip: float | None,
) -> tuple[float, float, float]:
    model.train()
    num_samp = cfg['samples_per_scene']
    batch_split = max(1, int(cfg.get('batch_split', 1)))

    g = torch.Generator(device='cpu')
    g.manual_seed(int(cfg.get('seed', 42)) + epoch_1based)
    order = torch.randperm(len(scene_ds), generator=g).tolist()

    total_loss = 0.0
    total_l1 = 0.0
    total_reg = 0.0

    do_l2 = cfg.get('code_regularization', True)
    do_sphere = cfg.get('code_regularization_sphere', False)
    code_lambda = float(cfg.get('code_regularization_lambda', 1e-4))
    ramp_epochs = int(cfg.get('reg_ramp_epochs', 100))
    loss_mult = float(cfg.get('loss_multiplier', 1.0))

    for scene_order_idx in order:
        sample = scene_ds[scene_order_idx]
        sdf_data = sample['sdf_data']
        latent_idx = int(sample['latent_idx'])

        xyz = sdf_data[:, :3].to(device)
        sdf_gt = sdf_data[:, 3:4].to(device)

        xyz_chunks = torch.chunk(xyz, batch_split)
        sdf_chunks = torch.chunk(sdf_gt, batch_split)
        idx_full = torch.full(
            (num_samp,), latent_idx, dtype=torch.long, device=device,
        )
        idx_chunks = torch.chunk(idx_full, batch_split)

        optimizer.zero_grad(set_to_none=True)
        scene_loss = 0.0
        scene_l1 = 0.0
        scene_reg = 0.0

        for ci in range(batch_split):
            batch_vecs = lat_vecs(idx_chunks[ci])
            decoder_input = torch.cat([batch_vecs, xyz_chunks[ci]], dim=1)
            pred = model(decoder_input)
            if cfg.get('clamp', False):
                v = float(cfg['clamp_value'])
                pred = torch.clamp(pred, -v, v)

            chunk_loss, l1_part, reg_l2, reg_s = sdf_autodecoder_loss_chunk(
                pred,
                sdf_chunks[ci],
                batch_vecs,
                num_samp,
                epoch_1based,
                code_lambda,
                ramp_epochs,
                do_l2,
                do_sphere,
            )
            chunk_loss = chunk_loss * loss_mult
            chunk_loss.backward()

            scene_loss += float(chunk_loss.item())
            scene_l1 += float(l1_part.item()) * loss_mult
            scene_reg += (float(reg_l2.item()) + float(reg_s.item())) * loss_mult

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(lat_vecs.parameters(), grad_clip)

        optimizer.step()
        total_loss += scene_loss
        total_l1 += scene_l1
        total_reg += scene_reg

    n = len(scene_ds)
    return total_loss / n, total_l1 / n, total_reg / n


def main(cfg: dict):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    seed = int(cfg.get('seed', 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    run_tag = datetime.now().strftime('%d_%m_%H%M%S')
    output_dir = os.path.join(cfg['output_dir'], run_tag)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f, sort_keys=False)
    writer = SummaryWriter(log_dir=output_dir)

    stage_splits = cfg.get('stage1_splits', ['train', 'val'])
    clamp_val = cfg['clamp_value'] if cfg.get('clamp', False) else None

    scene_ds = SDFSceneDataset(
        cfg['sdf_data_dir'],
        cfg['splits_csv'],
        split=stage_splits,
        samples_per_scene=int(cfg['samples_per_scene']),
        clamp_value=clamp_val,
    )

    model = SDFDecoder(
        latent_size=cfg['latent_size'],
        num_layers=cfg['num_layers'],
        inner_dim=cfg['inner_dim'],
        skip_connections=cfg['skip_connections'],
    ).float().to(device)

    latent_size = int(cfg['latent_size'])
    num_shapes = scene_ds.num_shapes
    code_bound = cfg.get('code_bound', None)
    max_norm = float(code_bound) if code_bound is not None else None

    lat_vecs = torch.nn.Embedding(
        num_shapes,
        latent_size,
        max_norm=max_norm,
    ).float().to(device)

    init_std = float(cfg.get('code_init_std_dev', 1.0))
    torch.nn.init.normal_(
        lat_vecs.weight.data,
        0.0,
        init_std / math.sqrt(latent_size),
    )

    lr_specs = cfg['learning_rate_schedule']
    if len(lr_specs) != 2:
        raise ValueError('learning_rate_schedule must have exactly 2 entries: model, latent')
    schedules = build_lr_schedules(lr_specs)

    optimizer = optim.Adam(
        [
            {
                'params': model.parameters(),
                'lr': schedules[0].get_learning_rate(1),
            },
            {
                'params': lat_vecs.parameters(),
                'lr': schedules[1].get_learning_rate(1),
            },
        ],
    )

    grad_clip = cfg.get('gradient_clip_norm', None)
    if grad_clip is not None:
        grad_clip = float(grad_clip)

    num_epochs = int(cfg['epochs'])
    snapshot_freq = int(cfg.get('snapshot_frequency', 50))
    log_freq = int(cfg.get('log_frequency', 10))

    start_epoch = 1
    if cfg.get('pretrained', False) and cfg.get('pretrain_dir', ''):
        d = cfg['pretrain_dir']
        model.load_state_dict(
            torch.load(os.path.join(d, 'decoder.pth'), map_location=device),
        )
        emb_path = os.path.join(d, 'latent_embedding.pt')
        if os.path.isfile(emb_path):
            lat_vecs.load_state_dict(torch.load(emb_path, map_location=device))
        opt_path = os.path.join(d, 'optimizer.pt')
        if os.path.isfile(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))
        meta_path = os.path.join(d, 'meta.pt')
        if os.path.isfile(meta_path):
            meta = torch.load(meta_path, map_location='cpu')
            start_epoch = int(meta.get('epoch', 0)) + 1
        print(f'Resumed from {d}, starting epoch {start_epoch}')

    for epoch in range(start_epoch, num_epochs + 1):
        t0 = time.time()
        adjust_learning_rate(schedules, optimizer, epoch)

        train_loss, train_l1, train_reg = train_epoch(
            model,
            lat_vecs,
            optimizer,
            scene_ds,
            cfg,
            device,
            epoch,
            grad_clip,
        )
        elapsed = time.time() - t0

        print(
            f'Epoch {epoch:04d}/{num_epochs} | '
            f'train {train_loss:.5f} (l1={train_l1:.5f} reg={train_reg:.5f}) | '
            f'{elapsed:.1f}s'
        )

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/train_l1', train_l1, epoch)
        writer.add_scalar('Loss/train_reg', train_reg, epoch)
        writer.add_scalar('LR/model', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('LR/latent', optimizer.param_groups[1]['lr'], epoch)

        if snapshot_freq > 0 and epoch % snapshot_freq == 0:
            snap = os.path.join('snapshots', f'{epoch:04d}')
            save_run_checkpoint(
                output_dir, snap, model, lat_vecs, optimizer, epoch, scene_ds.label_to_idx,
            )
            print(f'  Snapshot saved: {snap}')

        if log_freq > 0 and epoch % log_freq == 0:
            save_run_checkpoint(
                output_dir, 'latest', model, lat_vecs, optimizer, epoch, scene_ds.label_to_idx,
            )

    save_run_checkpoint(
        output_dir, 'final', model, lat_vecs, optimizer, num_epochs, scene_ds.label_to_idx,
    )
    torch.save(model.state_dict(), os.path.join(output_dir, 'decoder.pth'))
    torch.save(lat_vecs.state_dict(), os.path.join(output_dir, 'latent_embedding.pt'))
    torch.save(optimizer.state_dict(), os.path.join(output_dir, 'optimizer.pt'))
    torch.save({'epoch': num_epochs}, os.path.join(output_dir, 'meta.pt'))

    latent_save_dir = os.path.join(output_dir, 'latent_codes')
    os.makedirs(latent_save_dir, exist_ok=True)
    for label, idx in scene_ds.label_to_idx.items():
        torch.save(lat_vecs.weight.data[idx].cpu(), os.path.join(latent_save_dir, f'{label}.pth'))

    writer.close()
    print(f'\nTraining complete. Outputs in: {output_dir}')
    print(f'Final train loss: {train_loss:.5f}')
    print(f'Latent codes: {latent_save_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 1: DeepSDF autodecoder training')
    parser.add_argument('--config', '-c', required=True, help='Path to YAML config file')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg)
