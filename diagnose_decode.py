#!/usr/bin/env python3
"""
Decode-step diagnostic for PointSDF_2.

Bypasses the encoder entirely and feeds the corepp ground-truth latent codes
directly into the frozen decoder.  Measures volume RMSE / R² against GT.

If results are good (RMSE similar to corepp's reported ~22 mL on test):
    → decode step (grid_center, grid_bbox, grid_resolution) is correct.
    → the problem is entirely in the encoder; see encoder diagnostics next.

If results are bad (RMSE >> 30 mL or R² < 0):
    → decode step is broken regardless of encoder quality.
    → fix grid_center / grid_bbox first before retraining.

Usage (on server, from the PointSDF_2/ root):
    python misc/diagnose_decode.py --config configs/train_encoder.yaml [--split val]
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from tqdm import tqdm

from models import SDFDecoder
from utils import get_volume_coords, sdf2mesh

warnings.filterwarnings("ignore")


def _load_decoder(cfg: dict, decoder_cfg: dict, device: torch.device) -> SDFDecoder:
    decoder = SDFDecoder(
        latent_size=decoder_cfg["latent_size"],
        num_layers=decoder_cfg["num_layers"],
        inner_dim=decoder_cfg["inner_dim"],
        skip_connections=decoder_cfg["skip_connections"],
    ).float().to(device)

    ckpt = torch.load(cfg["decoder_weights"], map_location=device)
    sd = ckpt.get("model_state_dict", ckpt)
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    decoder.load_state_dict(sd)
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad_(False)
    return decoder


def main(cfg: dict, split: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(cfg["decoder_config"]) as f:
        decoder_cfg = yaml.safe_load(f)

    decoder = _load_decoder(cfg, decoder_cfg, device)
    print(f"Loaded decoder from {cfg['decoder_weights']}")

    grid_resolution = int(cfg.get("grid_resolution", 20))
    grid_bbox = float(cfg.get("grid_bbox", 0.15))
    grid_center = torch.tensor(
        cfg.get("grid_center", [0.0, 0.0, 0.0]), dtype=torch.float, device=device
    )
    grid_coords = get_volume_coords(resolution=grid_resolution, bbox=grid_bbox).to(device) + grid_center

    center_str = f"  center={grid_center.cpu().tolist()}" if float(grid_center.norm()) > 1e-6 else "  center=[0,0,0]"
    print(
        f"SDF grid: {grid_resolution}³ = {grid_coords.size(0):,} points"
        f"  bbox=±{grid_bbox} m{center_str}"
    )

    latent_dir = Path(cfg["latent_dir"])
    print(f"Latent dir: {latent_dir}")

    splits_df = pd.read_csv(cfg["splits_csv"])
    split_labels = set(splits_df.loc[splits_df["split"] == split, "label"].astype(str))
    print(f"Split '{split}': {len(split_labels)} unique labels")

    volume_col = cfg.get("volume_column", "volume (cm3)")
    gt_df = pd.read_csv(cfg["target_csv"]).set_index("label")
    if volume_col not in gt_df.columns:
        raise ValueError(f"Column '{volume_col}' not found in {cfg['target_csv']}. "
                         f"Available: {list(gt_df.columns)}")

    latent_files = sorted(latent_dir.glob("*.pth"))
    latent_files = [p for p in latent_files if p.stem in split_labels]
    print(f"Found {len(latent_files)} latent .pth files matching split '{split}'")

    if not latent_files:
        print(
            "\nERROR: No latent files found. Check that latent_dir contains "
            "<label>.pth files for the requested split.\n"
            f"  latent_dir : {latent_dir}\n"
            f"  split      : {split}\n"
            f"  example labels: {list(split_labels)[:5]}"
        )
        return

    gt_vols, pred_vols, n_failed = [], [], 0
    latent_stats: dict[str, list] = {"norm": [], "mean": []}

    with torch.no_grad():
        for pth in tqdm(latent_files, desc="Decoding GT latents"):
            label = pth.stem
            if label not in gt_df.index:
                continue

            latent = torch.load(pth, weights_only=True, map_location=device)
            # Normalise to (1, L) regardless of saved shape
            if latent.dim() == 1:
                latent = latent.unsqueeze(0)
            latent = latent.float()

            latent_stats["norm"].append(float(latent.norm()))
            latent_stats["mean"].append(float(latent.mean()))

            lat_tiled = latent.expand(grid_coords.size(0), -1)
            decoder_input = torch.cat([lat_tiled, grid_coords], dim=1)
            pred_sdf = decoder(decoder_input)

            try:
                mesh = sdf2mesh(pred_sdf, grid_coords)
                if mesh.is_watertight():
                    pred_vols.append(mesh.get_volume() * 1e6)
                    gt_vols.append(float(gt_df.loc[label, volume_col]))
                else:
                    n_failed += 1
            except (ValueError, RuntimeError) as e:
                print(f"  Mesh failed for {label}: {e}")
                n_failed += 1

    n_valid = len(gt_vols)
    print(f"\nResults ({n_valid} valid / {n_failed} failed watertight meshes):")

    if n_valid < 2:
        print("  Not enough valid meshes to compute metrics.")
        return

    gt_arr = np.array(gt_vols)
    pred_arr = np.array(pred_vols)
    rmse = root_mean_squared_error(gt_arr, pred_arr)
    mae = mean_absolute_error(gt_arr, pred_arr)
    r2 = r2_score(gt_arr, pred_arr)

    print(f"  RMSE : {rmse:.2f} mL")
    print(f"  MAE  : {mae:.2f} mL")
    print(f"  R²   : {r2:.4f}")
    print(f"\nGT volume  — mean={gt_arr.mean():.1f} mL  std={gt_arr.std():.1f} mL"
          f"  range=[{gt_arr.min():.1f}, {gt_arr.max():.1f}]")
    print(f"Pred volume — mean={pred_arr.mean():.1f} mL  std={pred_arr.std():.1f} mL"
          f"  range=[{pred_arr.min():.1f}, {pred_arr.max():.1f}]")

    norm_arr = np.array(latent_stats["norm"])
    mean_arr = np.array(latent_stats["mean"])
    print(f"\nLatent code stats (n={len(norm_arr)}):")
    print(f"  ||z|| — mean={norm_arr.mean():.3f}  std={norm_arr.std():.3f}"
          f"  range=[{norm_arr.min():.3f}, {norm_arr.max():.3f}]")
    print(f"  mean(z) — mean={mean_arr.mean():.4f}  std={mean_arr.std():.4f}")

    print("\n" + "=" * 60)
    if r2 > 0.5 and rmse < 50:
        print("DIAGNOSIS: Decode step looks CORRECT.")
        print("  The GT latents produce reasonable volumes.")
        print("  The problem is in the encoder — it is not predicting")
        print("  good enough latent codes from the partial point clouds.")
        print("  Next steps:")
        print("    1. Check the training MSE loss curve (TensorBoard).")
        print("    2. Enable augmentation (augmentation_enabled: true).")
        print("    3. Increase grid_resolution to ≥40 for select_checkpoint.py.")
    elif r2 > 0 and rmse < 80:
        print("DIAGNOSIS: Decode step is MARGINAL.")
        print("  GT latents give some correlation but volume errors are large.")
        print("  Possible causes:")
        print("    - grid_resolution=20 is too coarse (bump to 40-64).")
        print("    - grid_bbox might clip part of some potatoes.")
        print("    - Encoder will also need fixing (GT latents are the ceiling).")
    else:
        print("DIAGNOSIS: Decode step appears BROKEN.")
        print(f"  GT latents give R²={r2:.3f} — worse than predicting the mean.")
        print("  The encoder cannot fix this; the decode configuration is wrong.")
        print("  Likely causes:")
        print("    - grid_center is wrong: should be the mean centroid of the")
        print("      UNCENTERED full laser/SfM scans in the SCANNER frame.")
        print("      Verify that your EDA computed this from the full scans,")
        print("      not from the partial (camera-frame) RGBD scans.")
        print("    - The corepp decoder may have been trained on CENTERED scans")
        print("      (if the centering code in corepp was not commented out).")
        print("      In that case, set grid_center: [0.0, 0.0, 0.0].")
        print("    - grid_bbox=0.15 m may be too small for some potatoes.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diagnostic: decode GT corepp latents and measure volume accuracy."
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to configs/train_encoder.yaml",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
        help="Which split to evaluate (default: val).",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg, args.split)
