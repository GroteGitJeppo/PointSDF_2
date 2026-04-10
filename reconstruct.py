#!/usr/bin/env python3
"""
Test-time latent optimisation for PointSDF_2 — mirrors corepp/reconstruct_deep_sdf.py.

Two main uses:

1. Checkpoint selection (run on val split for every saved checkpoint):
       python reconstruct.py -c configs/train_deepsdf.yaml \\
           --experiment_dir weights/deepsdf/09_04_210939 \\
           --checkpoint 500 --split val --chamfer

   Prints Chamfer distance on the val set so you can pick the best epoch E*.

2. Producing Stage 2 latent targets (run on train split with the best checkpoint):
       python reconstruct.py -c configs/train_deepsdf.yaml \\
           --experiment_dir weights/deepsdf/09_04_210939 \\
           --checkpoint <E*> --split train

   Writes one <label>.pth per shape under
       <experiment_dir>/Reconstructions/<epoch>/Codes/<split>/
   Point configs/train_encoder.yaml latent_dir at that folder.
"""

import argparse
import logging
import math
import os

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from data.sdf_samples import resolve_samples_npz
from models.decoder import Decoder
from models import SDFDecoder

MODEL_PARAMS_SUBDIR = "ModelParameters"
LATENT_CODES_SUBDIR = "LatentCodes"
RECONSTRUCTIONS_SUBDIR = "Reconstructions"
CODES_SUBDIR = "Codes"


# ---------------------------------------------------------------------------
# Decoder loading
# ---------------------------------------------------------------------------

def _load_decoder(experiment_dir: str, checkpoint: str, cfg: dict) -> torch.nn.Module:
    """Load Stage 1 decoder from a DataParallel checkpoint, return bare module on CUDA."""
    ckpt_path = os.path.join(experiment_dir, MODEL_PARAMS_SUBDIR, checkpoint + ".pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Decoder checkpoint not found: {ckpt_path}")

    data = torch.load(ckpt_path, map_location="cpu")
    sd = data["model_state_dict"]
    sd = {k.removeprefix("module."): v for k, v in sd.items()}

    if cfg.get("use_facebook_decoder_specs", False):
        ns = cfg["network_specs"]
        decoder = Decoder(cfg["latent_size"], **ns)
    else:
        decoder = SDFDecoder(
            latent_size=cfg["latent_size"],
            num_layers=cfg["num_layers"],
            inner_dim=cfg["inner_dim"],
            skip_connections=cfg.get("skip_connections", True),
            dropout_prob=float(cfg.get("dropout_prob", 0.2)),
            weight_norm=cfg.get("weight_norm", True),
        )

    decoder.load_state_dict(sd)
    decoder.eval()
    return decoder.cuda()


# ---------------------------------------------------------------------------
# Training latent statistics (used to initialise test-time latents)
# ---------------------------------------------------------------------------

def _empirical_stat(experiment_dir: str, checkpoint: str):
    """Return (mean, std) tensors computed from the saved training latent codes."""
    lat_path = os.path.join(experiment_dir, LATENT_CODES_SUBDIR, checkpoint + ".pth")
    if not os.path.isfile(lat_path):
        raise FileNotFoundError(f"Latent codes file not found: {lat_path}")
    data = torch.load(lat_path, map_location="cpu")
    # state_dict format: {"weight": (N, latent_size)}
    weight = data["latent_codes"]["weight"]  # (N, L)
    mean = weight.mean(dim=0)
    std = weight.std(dim=0).clamp(min=1e-6)
    return mean.cuda(), std.cuda()


# ---------------------------------------------------------------------------
# Per-shape latent optimisation (mirrors corepp reconstruct_deep_sdf.py)
# ---------------------------------------------------------------------------

def _optimise_latent(
    decoder: torch.nn.Module,
    pos_tensor: torch.Tensor,
    neg_tensor: torch.Tensor,
    latent_size: int,
    emp_mean: torch.Tensor,
    emp_std: torch.Tensor,
    clamp_dist: float,
    num_iterations: int = 800,
    num_samples: int = 32000,
    lr: float = 0.1,
    l2reg: bool = True,
) -> torch.Tensor:
    """
    Optimise a single latent code from SDF samples of one shape.

    Returns the optimised latent (1, latent_size) on CPU.
    """
    latent = torch.empty(1, latent_size).normal_(
        mean=0.0, std=1.0
    ).cuda()
    # initialise from empirical distribution of training codes
    latent = emp_mean.unsqueeze(0) + emp_std.unsqueeze(0) * latent
    latent.requires_grad_(True)

    optimizer = torch.optim.Adam([latent], lr=lr)
    loss_fn = torch.nn.MSELoss(reduction="sum")

    # LR step: halve at midpoint (mirrors corepp)
    decreased_by = 10
    adjust_lr_every = int(num_iterations / 2)

    all_samples = torch.cat([pos_tensor, neg_tensor], dim=0)  # (M, 4)

    for iteration in range(num_iterations):
        # LR schedule
        current_lr = lr * ((1.0 / decreased_by) ** (iteration // adjust_lr_every))
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        # subsample
        idx = torch.randperm(all_samples.shape[0])[:num_samples]
        batch = all_samples[idx].cuda().float()

        xyz = batch[:, :3]
        sdf_gt = batch[:, 3:4].clamp(-clamp_dist, clamp_dist)

        latent_exp = latent.expand(xyz.shape[0], -1)
        net_in = torch.cat([latent_exp, xyz], dim=1)

        decoder.eval()
        pred_sdf = decoder(net_in).clamp(-clamp_dist, clamp_dist)

        loss = loss_fn(pred_sdf, sdf_gt)
        if l2reg:
            loss = loss + 1e-4 * latent.pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return latent.detach().squeeze(0).cpu()  # (latent_size,)


# ---------------------------------------------------------------------------
# Chamfer distance (symmetric, L2)
# ---------------------------------------------------------------------------

def _chamfer(pred_pts: torch.Tensor, gt_pts: torch.Tensor) -> float:
    dists = torch.cdist(pred_pts.unsqueeze(0), gt_pts.unsqueeze(0)).squeeze(0)
    cd = dists.min(dim=1).values.mean() + dists.min(dim=0).values.mean()
    return cd.item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    with open(args.decoder_config) as f:
        cfg = yaml.safe_load(f)

    latent_size = int(cfg["latent_size"])
    clamp_dist = float(cfg.get("clamp_value", cfg.get("clamping_distance", 0.1)))

    # Resolve experiment_dir: if not given, fall back to cfg output_dir
    experiment_dir = args.experiment_dir
    if experiment_dir is None:
        experiment_dir = cfg["output_dir"]
        logging.info("--experiment_dir not set, using cfg output_dir: %s", experiment_dir)

    decoder = _load_decoder(experiment_dir, args.checkpoint, cfg)
    logging.info("Loaded decoder from checkpoint '%s'", args.checkpoint)

    emp_mean, emp_std = _empirical_stat(experiment_dir, args.checkpoint)
    logging.info(
        "Empirical latent stats — mean norm: %.4f, std norm: %.4f",
        emp_mean.norm().item(), emp_std.norm().item(),
    )

    # Resolve split labels from splits_csv
    splits_csv = args.splits_csv or cfg["splits_csv"]
    splits_df = pd.read_csv(splits_csv)
    requested = args.split.split(",")
    labels = sorted(
        splits_df[splits_df["split"].isin(requested)]["label"].astype(str).tolist()
    )
    if not labels:
        raise RuntimeError(f"No labels found for split={args.split!r} in {splits_csv}")
    logging.info("Reconstructing %d shapes (split=%s)", len(labels), args.split)

    # Load all SDF data into RAM
    sdf_data_dir = args.sdf_data_dir or cfg["sdf_data_dir"]
    ram: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    missing = []
    for label in labels:
        path = resolve_samples_npz(sdf_data_dir, label)
        if path is None:
            missing.append(label)
            continue
        raw = np.load(path)
        pos = torch.from_numpy(np.asarray(raw["pos"], dtype=np.float32))
        neg = torch.from_numpy(np.asarray(raw["neg"], dtype=np.float32))
        ram[label] = (pos, neg)
    if missing:
        logging.warning("%d label(s) have no samples.npz: %s", len(missing), missing[:10])
    labels = [l for l in labels if l in ram]

    # Output directory
    saved_epoch = args.checkpoint
    output_root = args.output_dir or experiment_dir
    codes_dir = os.path.join(
        output_root, RECONSTRUCTIONS_SUBDIR, saved_epoch, CODES_SUBDIR, args.split
    )
    os.makedirs(codes_dir, exist_ok=True)
    logging.info("Saving optimised latents to: %s", codes_dir)

    chamfer_vals = []

    for label in tqdm(labels, desc=f"Reconstructing [{args.split}]"):
        out_path = os.path.join(codes_dir, f"{label}.pth")
        if args.skip and os.path.isfile(out_path):
            logging.debug("Skipping %s (already exists)", label)
            continue

        pos_t, neg_t = ram[label]
        latent = _optimise_latent(
            decoder=decoder,
            pos_tensor=pos_t,
            neg_tensor=neg_t,
            latent_size=latent_size,
            emp_mean=emp_mean,
            emp_std=emp_std,
            clamp_dist=clamp_dist,
            num_iterations=args.iters,
            num_samples=args.num_samples,
            lr=args.lr,
            l2reg=True,
        )
        torch.save(latent, out_path)

        if args.chamfer:
            # Evaluate reconstruction quality using near-surface points as GT proxy
            all_pts = torch.cat([pos_t[:, :3], neg_t[:, :3]], dim=0)
            if all_pts.shape[0] > 4096:
                idx = torch.randperm(all_pts.shape[0])[:4096]
                all_pts = all_pts[idx]
            gt_pts = all_pts.cuda()

            # Reconstruct predicted surface points from SDF grid
            grid_res = 32
            bbox = clamp_dist * 1.5
            vals = torch.linspace(-bbox, bbox, grid_res)
            g = torch.meshgrid(vals, vals, vals, indexing="ij")
            grid = torch.stack([g_.ravel() for g_ in g], dim=1).cuda()

            with torch.no_grad():
                lat_exp = latent.cuda().unsqueeze(0).expand(grid.shape[0], -1)
                net_in = torch.cat([lat_exp, grid], dim=1)
                pred_sdf = decoder(net_in).squeeze(1)

            surface_mask = pred_sdf.abs() < (bbox / grid_res * 2)
            surface_pts = grid[surface_mask]
            if surface_pts.shape[0] >= 4:
                cd = _chamfer(surface_pts.float(), gt_pts.float())
                chamfer_vals.append(cd)
                logging.debug("  %s  CD=%.5f", label, cd)

    if args.chamfer and chamfer_vals:
        mean_cd = float(np.mean(chamfer_vals))
        print(
            f"\nCheckpoint {args.checkpoint} | split={args.split} | "
            f"mean Chamfer = {mean_cd * 1000:.3f} mm  (n={len(chamfer_vals)})"
        )
    elif args.chamfer:
        print(f"\nCheckpoint {args.checkpoint} | split={args.split} | no valid Chamfer values")

    print(f"\nDone. Latents written to:\n  {codes_dir}")
    print(
        f"\nTo use as Stage 2 targets, set in configs/train_encoder.yaml:\n"
        f"  latent_dir: {codes_dir}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Test-time latent optimisation for PointSDF_2 — "
            "checkpoint selection (val) and Stage 2 target generation (train)."
        )
    )
    parser.add_argument(
        "--decoder_config", "-c",
        required=True,
        help="Path to train_deepsdf.yaml (provides architecture + data paths).",
    )
    parser.add_argument(
        "--experiment_dir", "-e",
        default=None,
        help="Stage 1 output directory containing ModelParameters/ and LatentCodes/. "
             "Falls back to cfg['output_dir'] if omitted.",
    )
    parser.add_argument(
        "--checkpoint",
        default="latest",
        help="Decoder checkpoint name without .pth (e.g. 500, latest). Default: latest.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Which split(s) to reconstruct, comma-separated (e.g. train, val, train,val). "
             "Default: train.",
    )
    parser.add_argument(
        "--splits_csv",
        default=None,
        help="Override splits_csv from config.",
    )
    parser.add_argument(
        "--sdf_data_dir",
        default=None,
        help="Override sdf_data_dir from config.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Root dir for Reconstructions/ output. Defaults to experiment_dir.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=800,
        help="Latent optimisation iterations per shape (default 800, matching corepp).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=32000,
        help="SDF points sampled per optimisation step (default 32000).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Initial learning rate for latent optimisation (default 0.1, matching corepp).",
    )
    parser.add_argument(
        "--chamfer",
        action="store_true",
        help="Compute and report mean Chamfer distance after reconstruction.",
    )
    parser.add_argument(
        "--skip",
        action="store_true",
        help="Skip shapes whose .pth file already exists in the output directory.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()
    main(args)
