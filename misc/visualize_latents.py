"""
Latent space visualisation — PCA and t-SNE.

Loads either:
  - Stage 1 autodecoder latents: a directory of ``{label}.pth`` tensors
    (one per unique tuber, produced by train_deepsdf.py → latent_codes/).
  - Stage 2 encoder latents: a single ``all_latents.pth`` dict file
    (produced by test.py → latent_dir/test/all_latents.pth),
    keyed by PLY stem (e.g. ``2R3-1_pcd_095``).

Produces three PNG figures per run:
  - pca_cultivar.png   — PCA 2-D, points coloured by cultivar
  - pca_volume.png     — PCA 2-D, points coloured by ground-truth volume
  - tsne_cultivar.png  — t-SNE 2-D, points coloured by cultivar

Usage (run from PointSDF_2/):
    # Stage 1 latents (directory)
    python misc/visualize_latents.py \\
        --latents weights/deepsdf/<run>/latent_codes \\
        --metadata data/3DPotatoTwin/ground_truth.csv \\
        --output misc/results/latents_stage1

    # Stage 2 encoder latents (single file)
    python misc/visualize_latents.py \\
        --latents weights/encoder/<run>/latent_dir/test/all_latents.pth \\
        --metadata data/3DPotatoTwin/ground_truth.csv \\
        --output misc/results/latents_stage2
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _stem_to_label(stem: str) -> str:
    """Extract unique_id from a PLY stem.

    ``2R3-1_pcd_095`` → ``2R3-1``
    Falls back to the full stem when '_pcd_' is absent (Stage 1 stems are
    already the label itself).
    """
    if "_pcd_" in stem:
        return stem.split("_pcd_")[0]
    return stem


def load_latents(path: str) -> tuple[np.ndarray, list[str]]:
    """Load latent vectors and return (matrix, labels).

    Accepts either:
      - A directory of individual ``{label}.pth`` files (Stage 1).
      - A single ``.pth`` file containing a ``dict[str, Tensor]`` (Stage 2).

    Returns
    -------
    latents : np.ndarray, shape (N, latent_size)
    labels  : list[str], length N — unique_id for each row
    """
    p = Path(path)

    if p.is_dir():
        pth_files = sorted(p.glob("*.pth"))
        if not pth_files:
            raise FileNotFoundError(f"No .pth files found in {p}")
        vecs, labels = [], []
        for f in pth_files:
            t = torch.load(f, map_location="cpu")
            if isinstance(t, dict):
                # checkpoint dict — skip (shouldn't happen in latent_codes/)
                continue
            vecs.append(t.float().numpy().ravel())
            labels.append(_stem_to_label(f.stem))
        return np.stack(vecs), labels

    if p.suffix == ".pth":
        data = torch.load(p, map_location="cpu")
        if isinstance(data, dict):
            vecs, labels = [], []
            for stem, t in data.items():
                vecs.append(t.float().numpy().ravel())
                labels.append(_stem_to_label(stem))
            return np.stack(vecs), labels
        # Single tensor — no label info
        return data.float().numpy().reshape(1, -1), ["unknown"]

    raise ValueError(f"--latents must be a directory or a .pth file, got: {path}")


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _scatter(
    ax: plt.Axes,
    xy: np.ndarray,
    colors,
    title: str,
    cmap=None,
    vmin=None,
    vmax=None,
    legend_handles=None,
    colorbar_label: str | None = None,
) -> None:
    sc = ax.scatter(
        xy[:, 0], xy[:, 1],
        c=colors, cmap=cmap, vmin=vmin, vmax=vmax,
        s=18, alpha=0.75, linewidths=0,
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.set_aspect("equal", adjustable="datalim")
    if legend_handles is not None:
        ax.legend(handles=legend_handles, fontsize=7, markerscale=1.2,
                  loc="best", framealpha=0.7)
    if colorbar_label is not None:
        plt.colorbar(sc, ax=ax, label=colorbar_label, shrink=0.8)


def _cultivar_colors(labels: list[str], meta: pd.DataFrame | None):
    """Return integer colour indices and a legend-ready list of patch handles."""
    if meta is not None and "cultivar" in meta.columns:
        cultivars = [
            str(meta.loc[lbl, "cultivar"]) if lbl in meta.index else "unknown"
            for lbl in labels
        ]
    else:
        cultivars = ["unknown"] * len(labels)

    unique_cult = sorted(set(cultivars))
    cmap = plt.get_cmap("tab10", len(unique_cult))
    cult_idx = {c: i for i, c in enumerate(unique_cult)}
    color_ints = np.array([cult_idx[c] for c in cultivars])

    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=cmap(cult_idx[c]), markersize=7, label=c
        )
        for c in unique_cult
    ]
    return color_ints, cmap, handles


def _volume_colors(labels: list[str], meta: pd.DataFrame | None, volume_col: str):
    if meta is None or volume_col not in meta.columns:
        return None, None, None
    vols = np.array([
        float(meta.loc[lbl, volume_col]) if lbl in meta.index else np.nan
        for lbl in labels
    ])
    return vols, "viridis", volume_col


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(latents_path: str, metadata_csv: str | None, output_dir: str,
         volume_col: str, tsne_perplexity: int, tsne_seed: int) -> None:
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading latents from: {latents_path}")
    Z, labels = load_latents(latents_path)
    print(f"  {Z.shape[0]} vectors, latent_size={Z.shape[1]}")

    # --- Metadata ---
    meta: pd.DataFrame | None = None
    if metadata_csv:
        meta = pd.read_csv(metadata_csv)
        if "label" in meta.columns:
            meta = meta.set_index("label")
        else:
            print("  WARNING: metadata CSV has no 'label' column — skipping metadata")
            meta = None

    # --- PCA ---
    print("Running PCA ...")
    pca = PCA(n_components=2, random_state=0)
    Z_pca = pca.fit_transform(Z)
    explained = pca.explained_variance_ratio_ * 100
    print(f"  Explained variance: PC1={explained[0]:.1f}%  PC2={explained[1]:.1f}%")

    # PCA — coloured by cultivar
    c_ints, c_cmap, c_handles = _cultivar_colors(labels, meta)
    fig, ax = plt.subplots(figsize=(7, 6))
    _scatter(
        ax, Z_pca, c_ints,
        title=f"PCA — cultivar  (PC1 {explained[0]:.1f}%, PC2 {explained[1]:.1f}%)",
        cmap=c_cmap, vmin=-0.5, vmax=len(c_handles) - 0.5,
        legend_handles=c_handles,
    )
    out_path = os.path.join(output_dir, "pca_cultivar.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")

    # PCA — coloured by volume
    v_vals, v_cmap, v_label = _volume_colors(labels, meta, volume_col)
    if v_vals is not None and not np.all(np.isnan(v_vals)):
        fig, ax = plt.subplots(figsize=(7, 6))
        _scatter(
            ax, Z_pca, v_vals,
            title=f"PCA — volume ({volume_col})"
                  f"  (PC1 {explained[0]:.1f}%, PC2 {explained[1]:.1f}%)",
            cmap=v_cmap,
            vmin=np.nanmin(v_vals), vmax=np.nanmax(v_vals),
            colorbar_label=f"{volume_col} (mL)",
        )
        out_path = os.path.join(output_dir, "pca_volume.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved {out_path}")
    else:
        print("  Skipping pca_volume.png (no volume data found)")

    # --- t-SNE ---
    n = Z.shape[0]
    perp = min(tsne_perplexity, max(5, n // 3))
    print(f"Running t-SNE (perplexity={perp}, n={n}) ...")
    tsne = TSNE(n_components=2, perplexity=perp, random_state=tsne_seed,
                n_iter=1000, init="pca")
    Z_tsne = tsne.fit_transform(Z)

    fig, ax = plt.subplots(figsize=(7, 6))
    _scatter(
        ax, Z_tsne, c_ints,
        title=f"t-SNE — cultivar  (perplexity={perp})",
        cmap=c_cmap, vmin=-0.5, vmax=len(c_handles) - 0.5,
        legend_handles=c_handles,
    )
    out_path = os.path.join(output_dir, "tsne_cultivar.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")

    print(f"\nAll figures written to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PCA / t-SNE visualisation of DeepSDF or encoder latent codes"
    )
    parser.add_argument(
        "--latents", required=True,
        help="Path to Stage 1 latent_codes/ directory OR Stage 2 all_latents.pth file",
    )
    parser.add_argument(
        "--metadata", default=None,
        help="CSV with 'label', 'cultivar', volume columns (e.g. ground_truth.csv)",
    )
    parser.add_argument(
        "--output", default="misc/results/latents",
        help="Output directory for PNG figures (default: misc/results/latents)",
    )
    parser.add_argument(
        "--volume_col", default="volume_ml",
        help="Column name for volume in metadata CSV (default: volume_ml)",
    )
    parser.add_argument(
        "--tsne_perplexity", type=int, default=30,
        help="t-SNE perplexity (capped at n//3; default: 30)",
    )
    parser.add_argument(
        "--tsne_seed", type=int, default=42,
        help="Random seed for t-SNE (default: 42)",
    )
    args = parser.parse_args()
    main(
        latents_path=args.latents,
        metadata_csv=args.metadata,
        output_dir=args.output,
        volume_col=args.volume_col,
        tsne_perplexity=args.tsne_perplexity,
        tsne_seed=args.tsne_seed,
    )
