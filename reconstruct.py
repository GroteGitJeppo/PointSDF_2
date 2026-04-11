#!/usr/bin/env python3
"""
Test-time latent optimisation for PointSDF_2 — mirrors corepp/reconstruct_deep_sdf.py.

Three main uses:

1. Sweep all checkpoints on the val split to find the best epoch E*
   (replaces corepp/run_scripts_reconstruct.sh — loads SDF data into RAM only once):

       python reconstruct.py -c configs/train_deepsdf.yaml \\
           --experiment_dir weights/deepsdf/<run> \\
           --split val --all-checkpoints        # test every 10th epoch (corepp default)

       python reconstruct.py -c configs/train_deepsdf.yaml \\
           --experiment_dir weights/deepsdf/<run> \\
           --split val --all-checkpoints 50     # test every 50th epoch (faster sweep)

   Prints a sorted leaderboard and highlights the best checkpoint at the end.

2. Evaluate a single checkpoint (with Chamfer, for spot-checks):

       python reconstruct.py -c configs/train_deepsdf.yaml \\
           --experiment_dir weights/deepsdf/<run> \\
           --checkpoint 500 --split val --chamfer

3. Produce Stage 2 latent targets using the best checkpoint E*:

       python reconstruct.py -c configs/train_deepsdf.yaml \\
           --experiment_dir weights/deepsdf/<run> \\
           --checkpoint <E*> --split train

   Writes one <label>.pth per shape under
       <experiment_dir>/Reconstructions/<E*>/Codes/<split>/
   Point configs/train_encoder.yaml latent_dir at that folder.
"""

import argparse
import glob
import logging
import os

import numpy as np
import open3d as o3d
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from data.sdf_samples import resolve_samples_npz
from models.decoder import Decoder
from models import SDFDecoder
from utils import get_volume_coords, sdf2mesh
from metrics_3d.chamfer_distance import ChamferDistance

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

    Returns the optimised latent (latent_size,) on CPU.
    """
    latent = torch.empty(1, latent_size).normal_(mean=0.0, std=1.0).cuda()
    latent = emp_mean.unsqueeze(0) + emp_std.unsqueeze(0) * latent
    latent.requires_grad_(True)

    optimizer = torch.optim.Adam([latent], lr=lr)
    loss_fn = torch.nn.MSELoss(reduction="sum")

    # LR halved at midpoint (mirrors corepp)
    decreased_by = 10
    adjust_lr_every = int(num_iterations / 2)

    all_samples = torch.cat([pos_tensor, neg_tensor], dim=0)  # (M, 4)

    for iteration in range(num_iterations):
        current_lr = lr * ((1.0 / decreased_by) ** (iteration // adjust_lr_every))
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

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
# Chamfer distance — corepp-compatible
# ---------------------------------------------------------------------------
# Mirrors corepp/compute_reconstruction_metrics.py:
#   - GT  = complete laser/SfM PLY, centred (matches T.Center applied to partial scans)
#   - Pred = mesh extracted via sdf2mesh (convex hull of SDF < 0, same as test.py)
#   - Metric = metrics_3d.ChamferDistance: (mean(gt→pred) + mean(pred→gt)) / 2
# ---------------------------------------------------------------------------

_cd_metric = ChamferDistance()


def _load_gt_pcd_for_reconstruct(gt_pcd_dir: str, label: str, ply_pattern: str):
    """Load and centre the complete laser PLY for one shape. Returns None if not found."""
    matches = glob.glob(os.path.join(gt_pcd_dir, label, ply_pattern))
    if not matches:
        return None
    pcd = o3d.io.read_point_cloud(matches[0])
    pcd.translate(-pcd.get_center())
    return pcd


def _chamfer_for_latent(
    latent: torch.Tensor,
    decoder: torch.nn.Module,
    clamp_dist: float,
    gt_pcd_dir: str,
    label: str,
    ply_pattern: str,
    grid_resolution: int = 64,
) -> float | None:
    """
    Compute corepp-compatible Chamfer distance for one shape given its latent.

    Extracts a mesh from the SDF grid (same pipeline as test.py) and compares
    it against the centred GT laser PLY using metrics_3d.ChamferDistance.
    Returns None when the mesh cannot be extracted or the GT PLY is missing.
    """
    gt_pcd = _load_gt_pcd_for_reconstruct(gt_pcd_dir, label, ply_pattern)
    if gt_pcd is None:
        logging.debug("    %s: GT PLY not found, skipping Chamfer", label)
        return None

    bbox = clamp_dist * 1.5
    grid_coords = get_volume_coords(resolution=grid_resolution, bbox=bbox).cuda()

    with torch.no_grad():
        lat_exp = latent.cuda().unsqueeze(0).expand(grid_coords.shape[0], -1)
        net_in = torch.cat([lat_exp, grid_coords], dim=1)
        pred_sdf = decoder(net_in)

    try:
        mesh = sdf2mesh(pred_sdf, grid_coords)
    except (ValueError, RuntimeError) as e:
        logging.debug("    %s: mesh extraction failed: %s", label, e)
        return None

    _cd_metric.reset()
    _cd_metric.update(gt_pcd, mesh)
    return _cd_metric.compute(print_output=False)


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def _discover_checkpoints(experiment_dir: str, step: int) -> list[int]:
    """
    Find all numeric checkpoint files in ModelParameters/, filter by step.

    Only epochs divisible by `step` are returned — matching the stride logic
    from corepp/run_scripts_reconstruct.sh (which used step=10).

    Example: with step=50 and checkpoints [10,20,...,1000] on disk,
    returns [50, 100, 150, ..., 1000].
    """
    folder = os.path.join(experiment_dir, MODEL_PARAMS_SUBDIR)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"ModelParameters/ not found in: {experiment_dir}")

    epochs = []
    for fname in os.listdir(folder):
        if fname.endswith(".pth"):
            stem = fname[:-4]
            if stem.isdigit():
                epoch = int(stem)
                if epoch % step == 0:
                    epochs.append(epoch)
    return sorted(epochs)


# ---------------------------------------------------------------------------
# Per-checkpoint worker (shared between single-checkpoint and sweep modes)
# ---------------------------------------------------------------------------

def _run_checkpoint(
    checkpoint: str,
    experiment_dir: str,
    cfg: dict,
    labels: list[str],
    ram: dict,
    args,
    compute_chamfer: bool,
) -> float | None:
    """
    Run reconstruction for one checkpoint. Returns mean Chamfer (m) or None.

    Loads the decoder and latent stats fresh for this checkpoint, then
    optimises a latent per shape and optionally computes Chamfer distance.
    Latents are saved to disk; existing files are re-used when --skip is set.
    """
    latent_size = int(cfg["latent_size"])
    clamp_dist = float(cfg.get("clamp_value", cfg.get("clamping_distance", 0.1)))

    try:
        decoder = _load_decoder(experiment_dir, checkpoint, cfg)
        emp_mean, emp_std = _empirical_stat(experiment_dir, checkpoint)
    except FileNotFoundError as exc:
        logging.warning("Skipping checkpoint %s: %s", checkpoint, exc)
        return None

    output_root = args.output_dir or experiment_dir
    codes_dir = os.path.join(
        output_root, RECONSTRUCTIONS_SUBDIR, checkpoint, CODES_SUBDIR, args.split
    )
    os.makedirs(codes_dir, exist_ok=True)

    chamfer_vals = []

    for label in tqdm(labels, desc=f"  epoch {checkpoint}", leave=False):
        out_path = os.path.join(codes_dir, f"{label}.pth")

        if args.skip and os.path.isfile(out_path):
            latent = torch.load(out_path, map_location="cpu", weights_only=True)
        else:
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

        if compute_chamfer:
            gt_pcd_dir = cfg.get("gt_pcd_dir", None)
            ply_pattern = cfg.get("gt_ply_pattern", "*.ply")
            if gt_pcd_dir is None:
                logging.warning(
                    "gt_pcd_dir not set in config — Chamfer skipped. "
                    "Add gt_pcd_dir to train_deepsdf.yaml to enable corepp-compatible Chamfer."
                )
                compute_chamfer = False
            else:
                cd = _chamfer_for_latent(
                    latent=latent,
                    decoder=decoder,
                    clamp_dist=clamp_dist,
                    gt_pcd_dir=gt_pcd_dir,
                    label=label,
                    ply_pattern=ply_pattern,
                )
                if cd is not None:
                    chamfer_vals.append(cd)
                    logging.debug("    %s  CD=%.5f", label, cd)

    mean_cd = float(np.mean(chamfer_vals)) if chamfer_vals else None
    return mean_cd


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

    experiment_dir = args.experiment_dir
    if experiment_dir is None:
        experiment_dir = cfg["output_dir"]
        logging.info("--experiment_dir not set, using cfg output_dir: %s", experiment_dir)

    # Resolve labels and load all SDF data into RAM once (shared across checkpoints)
    splits_csv = args.splits_csv or cfg["splits_csv"]
    splits_df = pd.read_csv(splits_csv)
    requested = args.split.split(",")
    labels = sorted(
        splits_df[splits_df["split"].isin(requested)]["label"].astype(str).tolist()
    )
    if not labels:
        raise RuntimeError(f"No labels found for split={args.split!r} in {splits_csv}")

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
    logging.info("Loaded %d shapes (split=%s) into RAM", len(labels), args.split)

    # -----------------------------------------------------------------------
    # SWEEP MODE  (--all-checkpoints <step>)
    # -----------------------------------------------------------------------
    if args.all_checkpoints is not None:
        step = args.all_checkpoints
        checkpoints = _discover_checkpoints(experiment_dir, step)

        if not checkpoints:
            raise RuntimeError(
                f"No numeric checkpoints divisible by {step} found in "
                f"{os.path.join(experiment_dir, MODEL_PARAMS_SUBDIR)}"
            )

        print(
            f"\nSweeping {len(checkpoints)} checkpoint(s) on split={args.split!r} "
            f"(step={step}, --skip is always on during sweep)\n"
        )
        args.skip = True  # reuse already-computed latents during sweep

        results: list[tuple[int, float]] = []  # (epoch, mean_cd)

        for epoch in checkpoints:
            checkpoint_str = str(epoch)
            print(f"[{epoch}/{checkpoints[-1]}] Reconstructing checkpoint {checkpoint_str} …")
            mean_cd = _run_checkpoint(
                checkpoint=checkpoint_str,
                experiment_dir=experiment_dir,
                cfg=cfg,
                labels=labels,
                ram=ram,
                args=args,
                compute_chamfer=True,
            )
            if mean_cd is not None:
                results.append((epoch, mean_cd))
                print(f"  → mean Chamfer = {mean_cd * 1000:.3f} mm")
            else:
                print(f"  → no valid Chamfer values (skipping in leaderboard)")

        # Print sorted leaderboard
        if results:
            results_sorted = sorted(results, key=lambda x: x[1])
            best_epoch, best_cd = results_sorted[0]

            col_w = max(len(str(r[0])) for r in results_sorted)
            header = f"{'epoch':>{col_w}}  |  Chamfer (mm)"
            sep = "-" * (len(header) + 4)
            print(f"\n{'=' * len(sep)}")
            print(f"Checkpoint sweep results — split={args.split!r}, step={step}")
            print(sep)
            print(f"  {header}")
            print(sep)
            for epoch, cd in results_sorted:
                marker = "  ← BEST" if epoch == best_epoch else ""
                print(f"  {epoch:>{col_w}}  |  {cd * 1000:>8.3f}{marker}")
            print(sep)
            print(
                f"\nBest checkpoint: epoch {best_epoch}  "
                f"(Chamfer = {best_cd * 1000:.3f} mm)"
            )
            print(
                f"\nNext steps — update configs/train_encoder.yaml:\n"
                f"  decoder_weights: {os.path.join(experiment_dir, MODEL_PARAMS_SUBDIR, str(best_epoch))}.pth\n"
                f"  latent_dir:      (run with --checkpoint {best_epoch} --split train to generate)"
            )
        else:
            print("\nNo valid results — check that samples.npz files exist.")

        return

    # -----------------------------------------------------------------------
    # SINGLE-CHECKPOINT MODE  (--checkpoint <name>)
    # -----------------------------------------------------------------------
    compute_chamfer = args.chamfer

    print(f"\nReconstructing checkpoint '{args.checkpoint}' on split={args.split!r} …")
    mean_cd = _run_checkpoint(
        checkpoint=args.checkpoint,
        experiment_dir=experiment_dir,
        cfg=cfg,
        labels=labels,
        ram=ram,
        args=args,
        compute_chamfer=compute_chamfer,
    )

    if compute_chamfer:
        if mean_cd is not None:
            print(
                f"\nCheckpoint {args.checkpoint} | split={args.split} | "
                f"mean Chamfer = {mean_cd * 1000:.3f} mm  (n shapes={len(labels)})"
            )
        else:
            print(f"\nCheckpoint {args.checkpoint} | split={args.split} | no valid Chamfer values")

    output_root = args.output_dir or experiment_dir
    codes_dir = os.path.join(
        output_root, RECONSTRUCTIONS_SUBDIR, args.checkpoint, CODES_SUBDIR, args.split
    )
    print(f"\nDone. Latents written to:\n  {codes_dir}")
    print(
        f"\nTo use as Stage 2 targets, set in configs/train_encoder.yaml:\n"
        f"  latent_dir: {codes_dir}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Test-time latent optimisation for PointSDF_2.\n\n"
            "Sweep mode (--all-checkpoints): finds the best Stage 1 epoch by evaluating\n"
            "Chamfer distance on the val split for every N-th saved checkpoint.\n"
            "Single-checkpoint mode (--checkpoint): reconstructs one checkpoint and\n"
            "generates latent targets for Stage 2 training."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
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

    # ---- Checkpoint selection: sweep or single ----
    ckpt_group = parser.add_mutually_exclusive_group()
    ckpt_group.add_argument(
        "--all-checkpoints",
        dest="all_checkpoints",
        type=int,
        nargs="?",
        const=10,           # default step when flag is given without a value
        default=None,       # None means single-checkpoint mode
        metavar="STEP",
        help=(
            "Sweep all numeric checkpoints in ModelParameters/, testing every STEP-th epoch. "
            "Defaults to STEP=10 (matching corepp/run_scripts_reconstruct.sh) if no value is given. "
            "Implies --chamfer and --skip. Example: --all-checkpoints 50"
        ),
    )
    ckpt_group.add_argument(
        "--checkpoint",
        default="latest",
        help="Single checkpoint name without .pth (e.g. 500, latest). Default: latest. "
             "Cannot be combined with --all-checkpoints.",
    )

    parser.add_argument(
        "--split",
        default="train",
        help="Which split(s) to reconstruct, comma-separated (e.g. train, val). "
             "Use 'val' for checkpoint selection, 'train' for Stage 2 target generation. "
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
        help="Root directory for Reconstructions/ output. Defaults to experiment_dir.",
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
        help="Compute and report mean Chamfer distance (single-checkpoint mode only). "
             "Always enabled in --all-checkpoints mode.",
    )
    parser.add_argument(
        "--skip",
        action="store_true",
        help="Re-use latent .pth files that already exist instead of recomputing them. "
             "Always enabled in --all-checkpoints mode.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging (per-shape Chamfer distances etc.).",
    )
    args = parser.parse_args()
    main(args)
