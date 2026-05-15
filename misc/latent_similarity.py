"""
Cosine similarity between encoder-predicted latents and Stage 1 ground-truth latents.

For every scan in the encoder's ``all_latents.pth`` dict the script:
  1. Extracts the unique_id (``2R3-1_pcd_095`` → ``2R3-1``).
  2. Loads the matching Stage 1 ground-truth latent from ``{label}.pth``.
  3. Computes cosine similarity between the two vectors.

Reports
-------
  - Per-scan cosine similarity, written to ``<output>/latent_similarity.csv``
  - Per-tuber mean ± std, printed to stdout
  - Overall stats (mean, median, std, min, max)
  - ``similarity_histogram.png`` — distribution of per-scan cosine similarity
  - ``similarity_per_tuber.png`` — bar chart sorted by mean similarity per tuber
  - ``pairwise_stage1.png``      — pairwise cosine similarity heatmap of the Stage 1
                                   latent space (shows how spread the space is)

Usage (run from PointSDF_2/):
    python misc/latent_similarity.py \\
        --stage1_latents weights/deepsdf/<run>/latent_codes \\
        --encoder_latents weights/encoder/<run>/latent_dir/test/all_latents.pth \\
        --output misc/results/similarity
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_stage1(latents_dir: str) -> dict[str, np.ndarray]:
    """Load Stage 1 per-label latents from a directory of ``{label}.pth`` files."""
    p = Path(latents_dir)
    pth_files = sorted(p.glob("*.pth"))
    if not pth_files:
        raise FileNotFoundError(f"No .pth files found in {p}")
    result = {}
    for f in pth_files:
        t = torch.load(f, map_location="cpu")
        if isinstance(t, dict):
            continue  # skip checkpoint files
        result[f.stem] = t.float().numpy().ravel()
    return result


def load_encoder(latents_path: str) -> dict[str, np.ndarray]:
    """Load Stage 2 encoder latents from ``all_latents.pth``."""
    data = torch.load(latents_path, map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError(
            f"{latents_path} does not contain a dict. "
            "Expected the all_latents.pth produced by test.py."
        )
    return {stem: t.float().numpy().ravel() for stem, t in data.items()}


def _stem_to_label(stem: str) -> str:
    """``2R3-1_pcd_095`` → ``2R3-1``."""
    if "_pcd_" in stem:
        return stem.split("_pcd_")[0]
    return stem


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_t = torch.tensor(a)
    b_t = torch.tensor(b)
    return float(F.cosine_similarity(a_t.unsqueeze(0), b_t.unsqueeze(0)).item())


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _histogram(sims: np.ndarray, output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(sims, bins=40, edgecolor="white", linewidth=0.4)
    ax.axvline(float(np.mean(sims)), color="red", linestyle="--",
               linewidth=1.2, label=f"mean={np.mean(sims):.3f}")
    ax.axvline(float(np.median(sims)), color="orange", linestyle=":",
               linewidth=1.2, label=f"median={np.median(sims):.3f}")
    ax.set_xlabel("Cosine similarity (encoder vs Stage 1)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of per-scan cosine similarity")
    ax.legend(fontsize=9)
    out = os.path.join(output_dir, "similarity_histogram.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def _per_tuber_bar(per_tuber: dict[str, list[float]], output_dir: str) -> None:
    labels = sorted(per_tuber.keys(), key=lambda k: np.mean(per_tuber[k]))
    means = np.array([np.mean(per_tuber[k]) for k in labels])
    stds = np.array([np.std(per_tuber[k]) for k in labels])

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.25), 5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=2, width=0.8,
           error_kw={"linewidth": 0.8, "elinewidth": 0.8})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_ylabel("Mean cosine similarity ± std")
    ax.set_title("Per-tuber cosine similarity (encoder vs Stage 1)")
    ax.set_ylim(0, 1.05)
    out = os.path.join(output_dir, "similarity_per_tuber.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def _pairwise_heatmap(stage1: dict[str, np.ndarray], output_dir: str) -> None:
    labels = sorted(stage1.keys())
    n = len(labels)
    mat = np.zeros((n, n), dtype=np.float32)
    vecs = np.stack([stage1[l] for l in labels])  # (n, D)
    # Normalise rows then dot product == cosine similarity
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    vecs_n = vecs / norms
    mat = vecs_n @ vecs_n.T  # (n, n)

    fig, ax = plt.subplots(figsize=(max(6, n * 0.18), max(5, n * 0.18)))
    im = ax.imshow(mat, vmin=0, vmax=1, aspect="auto", cmap="viridis")
    plt.colorbar(im, ax=ax, label="Cosine similarity", shrink=0.8)
    if n <= 80:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=90, fontsize=4)
        ax.set_yticklabels(labels, fontsize=4)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title(f"Pairwise cosine similarity — Stage 1 latents (n={n})")
    out = os.path.join(output_dir, "pairwise_stage1.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(stage1_dir: str, encoder_path: str, output_dir: str,
         pairwise: bool) -> None:
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading Stage 1 latents from: {stage1_dir}")
    stage1 = load_stage1(stage1_dir)
    print(f"  {len(stage1)} ground-truth latents loaded")

    print(f"Loading encoder latents from: {encoder_path}")
    encoder = load_encoder(encoder_path)
    print(f"  {len(encoder)} encoder latents loaded")

    # --- Match and compute similarity ---
    rows = []
    per_tuber: dict[str, list[float]] = {}
    skipped = 0

    for stem, enc_vec in sorted(encoder.items()):
        label = _stem_to_label(stem)
        if label not in stage1:
            skipped += 1
            continue
        sim = cosine_sim(enc_vec, stage1[label])
        rows.append({"stem": stem, "unique_id": label, "cosine_similarity": sim})
        per_tuber.setdefault(label, []).append(sim)

    if not rows:
        print(
            "ERROR: No encoder stems could be matched to Stage 1 labels. "
            "Check that stem format is '{label}_pcd_{frame}' and that "
            "--stage1_latents contains matching '{label}.pth' files."
        )
        return

    if skipped:
        print(f"  WARNING: {skipped} encoder stems had no matching Stage 1 latent (skipped)")

    df = pd.DataFrame(rows)
    csv_out = os.path.join(output_dir, "latent_similarity.csv")
    df.to_csv(csv_out, index=False)
    print(f"\nPer-scan similarity saved to: {csv_out}")

    sims = df["cosine_similarity"].to_numpy()
    print(f"\n=== Overall cosine similarity (n={len(sims)} scans, "
          f"{len(per_tuber)} tubers) ===")
    print(f"  Mean   : {np.mean(sims):.4f}")
    print(f"  Median : {np.median(sims):.4f}")
    print(f"  Std    : {np.std(sims):.4f}")
    print(f"  Min    : {np.min(sims):.4f}  ({df.loc[df['cosine_similarity'].idxmin(), 'stem']})")
    print(f"  Max    : {np.max(sims):.4f}  ({df.loc[df['cosine_similarity'].idxmax(), 'stem']})")

    print("\n=== Per-tuber mean cosine similarity ===")
    tuber_means = {lbl: np.mean(v) for lbl, v in per_tuber.items()}
    sorted_tubers = sorted(tuber_means.items(), key=lambda kv: kv[1])
    worst5 = sorted_tubers[:5]
    best5 = sorted_tubers[-5:][::-1]
    print("  Bottom 5:")
    for lbl, m in worst5:
        print(f"    {lbl:<20} mean={m:.4f}  n={len(per_tuber[lbl])}")
    print("  Top 5:")
    for lbl, m in best5:
        print(f"    {lbl:<20} mean={m:.4f}  n={len(per_tuber[lbl])}")

    # --- Figures ---
    print("\nGenerating figures ...")
    _histogram(sims, output_dir)
    _per_tuber_bar(per_tuber, output_dir)
    if pairwise:
        _pairwise_heatmap(stage1, output_dir)

    print(f"\nAll outputs written to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cosine similarity between encoder and Stage 1 latent codes"
    )
    parser.add_argument(
        "--stage1_latents", required=True,
        help="Directory of Stage 1 per-label .pth files (latent_codes/ from train_deepsdf.py)",
    )
    parser.add_argument(
        "--encoder_latents", required=True,
        help="Path to all_latents.pth produced by test.py",
    )
    parser.add_argument(
        "--output", default="misc/results/similarity",
        help="Output directory for CSV and figures (default: misc/results/similarity)",
    )
    parser.add_argument(
        "--pairwise", action="store_true",
        help="Also generate a pairwise cosine similarity heatmap of Stage 1 latents",
    )
    args = parser.parse_args()
    main(
        stage1_dir=args.stage1_latents,
        encoder_path=args.encoder_latents,
        output_dir=args.output,
        pairwise=args.pairwise,
    )
