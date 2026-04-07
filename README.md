# PointSDF: 3D Shape Completion for Potato Tuber Volume Estimation

## Summary

PointSDF is a two-stage encoder–decoder network that estimates the volume of potato tubers from partial point clouds captured on a harvester conveyor belt. Unlike regression approaches that predict weight directly, PointSDF completes the full 3D shape of each tuber using an implicit signed distance function (SDF) representation, then extracts volume from the reconstructed geometry.

The architecture is based on [PointRAFT](https://arxiv.org/abs/2512.24193) (encoder backbone) and [DeepSDF](https://arxiv.org/abs/1901.05103) (implicit decoder), with design choices validated against the [CoRe++](https://doi.org/10.1016/j.compag.2024.109673) pipeline.

```
Partial Point Cloud (.ply)
 │
 ▼
 PointNet++ Encoder          (SAModule × 2 + GlobalSA → latent z ∈ ℝ⁶⁴)
 │
 ▼
 Latent Code  z ∈ ℝ⁶⁴
 │
 ▼
 DeepSDF Decoder             (8-layer weight-normed MLP, skip at layer 3)
 concat(z, xyz) → SDF scalar
 │
 ▼
 Convex hull of SDF < 0 → Volume estimate (mL)
```

Training is split into two stages:

1. **Stage 1 — DeepSDF autodecoder** (`train_deepsdf.py`): trains an MLP decoder jointly with one latent code per potato on complete SDF samples extracted from the 3D laser scans. Saves per-shape latent codes to disk.
2. **Stage 2 — PointNet++ encoder** (`train.py`): freezes the Stage 1 decoder and trains a PointNet++ encoder to map partial point clouds to Stage 1 latent codes, supervised by MSE loss with an optional end-to-end SDF loss through the frozen decoder.

At inference, the encoder predicts a latent code from a single partial point cloud in ~10 ms. The decoder then evaluates the SDF on a 64³ grid, and volume is extracted via a GPU-accelerated convex hull of the predicted interior points.

---

## Installation

All code runs on a remote Debian server with an NVIDIA A40 GPU. The conda environment is defined in `environment.yaml`.

```bash
# Create and activate environment
conda env create -f environment.yaml
conda activate pointsdf

# Build and install the C++ FPS extension
cd pytorch_fpsample && pip install --no-deps --no-build-isolation . && cd ..

# Verify
python -c "import torch_fpsample; print('OK')"
```

Key dependencies: Python 3.12.3, PyTorch 2.8.0 (CUDA 12.8), PyTorch Geometric 2.7.0, Open3D 0.19.0.

---

## Dataset

The [3DPotatoTwin](https://huggingface.co/datasets/UTokyo-FieldPhenomics-Lab/3DPotatoTwin) dataset provides:

- **Partial point clouds** (RGB-D, from conveyor belt): `data/3DPotatoTwin/1_rgbd/2_pcd/<label>/<label>_pcd_*.ply`
- **Ground-truth volumes**: `data/3DPotatoTwin/mesh_traits_2023.csv` — column `volume (cm3)` (= mL) derived from SfM mesh reconstruction
- **Train/val/test split**: `data/3DPotatoTwin/splits.csv`

SDF sample files (`samples.npz`, one per potato label) must be generated from the complete SfM point clouds using the included preparation script:

```bash
python data/prepare_dataset.py sdf \
    --src  data/3DPotatoTwin/2_sfm/4_pcd \
    --out  data/3DPotatoTwin/sdfsamples/potato \
    --ply_pattern "*_20000.ply"
```

This writes `<out>/<label>/samples.npz` for each label. Each file contains `pos` and `neg` arrays of shape `(N, 4)` — columns `[x, y, z, sdf_value]`. Partial point clouds (if not already present) can be generated from the raw RGB-D images with:

```bash
python data/prepare_dataset.py pcd \
    --img_root   data/3DPotatoTwin/1_rgbd/1_image \
    --intrinsics data/3DPotatoTwin/1_rgbd/0_camera_intrinsics/realsense_d405_camera_intrinsic.json \
    --out        data/3DPotatoTwin/1_rgbd/2_pcd
```

---

## Usage

### Stage 1 — Train the SDF autodecoder

```bash
python train_deepsdf.py --config configs/train_deepsdf.yaml
```

Outputs are saved under `weights/deepsdf/<run_tag>/`:

- `decoder.pth` — trained decoder weights
- `latent_codes/<label>.pth` — one latent tensor per potato
- `snapshots/<epoch>/` — periodic checkpoints (every 10 epochs)

**Select the best Stage 1 checkpoint** by running `test.py` across snapshots and choosing the one with the lowest volume RMSE on the validation set. Update `latent_dir` and `decoder_weights` in `configs/train_encoder.yaml` to point to the chosen snapshot.

### Stage 2 — Train the PointNet++ encoder

```bash
python train.py --config configs/train_encoder.yaml
```

Outputs are saved under `weights/encoder/<run_tag>/`:

- `checkpoint.pth` — best checkpoint by validation loss (encoder + decoder)
- `snapshots/<epoch>/checkpoint.pth` — periodic checkpoints every 10 epochs
- `encoder.pth` — best encoder weights only

**Select the best Stage 2 checkpoint** by running `test.py` across snapshots and choosing the one with the lowest volume RMSE.

### Evaluation

```bash
python test.py \
  --config configs/train_encoder.yaml \
  --checkpoint weights/encoder/<run_tag>/checkpoint.pth
```

Reports MAE, RMSE, and R² on volume (mL), per-cultivar and per-season breakdowns, and Chamfer distance (if `sdf_data_dir` is set in the decoder config). Results are saved to `test_results.csv` alongside the checkpoint.

---

## Repository structure

```
PointSDF_2/
├── configs/
│   ├── train_deepsdf.yaml      # Stage 1 hyperparameters
│   └── train_encoder.yaml      # Stage 2 hyperparameters + data paths
├── data/
│   ├── 3DPotatoTwin/           # CSV files (splits, ground truth, mesh traits)
│   ├── encoder_dataset.py      # Stage 2 dataset: PLY + latent + optional SDF samples
│   ├── sdf_samples.py          # resolve_samples_npz() path helper
│   ├── sdf_scene_dataset.py    # Stage 1 dataset: SDF samples per shape
│   └── prepare_dataset.py      # data prep: pcd (RGB-D → PLY) and sdf (PLY → samples.npz)
├── models/
│   ├── encoder.py              # PointNet++ encoder → latent code
│   ├── decoder.py              # DeepSDF MLP decoder → SDF scalar
│   └── pointsdf.py             # combined PointSDF wrapper (for inference)
├── utils/
│   ├── sdf_helpers.py          # get_volume_coords, sdf2mesh, sdf_autodecoder_loss_chunk, chamfer_distance
│   └── visualize.py            # visualize_point_cloud(s) helpers
├── pytorch_fpsample/           # C++ farthest-point sampling extension
├── train_deepsdf.py            # Stage 1 training script
├── train.py                    # Stage 2 training script
├── test.py                     # evaluation script
└── environment.yaml            # conda environment definition
```

---

## Key design choices


| Choice                                                                 | Rationale                                                                                                                               |
| ---------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Convex hull (not marching cubes) for volume                            | Potatoes are roughly convex; convex hull is always watertight, GPU-accelerated, and avoids marching cubes grid resolution artefacts     |
| Latent size 64                                                         | Balances representational capacity and generalisation; CoRe++ found 32–64 optimal for potato tubers                                     |
| MSE loss for encoder (Stage 2)                                         | Penalises large latent deviations more than L1; validated in CoRe++ ablation study                                                      |
| No contrastive loss                                                    | CoRe++ ablation showed contrastive loss hurts for single top-view camera setups; only beneficial when multiple viewpoints are available |
| Augmentation: small pitch/roll (±2°), full yaw (±90°), left-right flip | Mirrors physical reality on the conveyor belt; vertical flip and shear removed as non-physical                                          |


---

## Citation

This work builds on the dataloader and encoder in PointRAFT and CoRe++

```bibtex
@misc{blok2025pointraft,
      title={PointRAFT: 3D deep learning for high-throughput prediction of potato tuber weight from partial point clouds},
      author={Pieter M. Blok and Haozhou Wang and Hyun Kwon Suh and Peicheng Wang and James Burridge and Wei Guo},
      year={2025},
      eprint={2512.24193},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2512.24193},
}

@article{blok2025corepp,
      title={High-throughput 3D shape completion of potato tubers on a harvester},
      author={Pieter M. Blok and Federico Magistri and Cyrill Stachniss and Haozhou Wang and James Burridge and Wei Guo},
      journal={Computers and Electronics in Agriculture},
      volume={228},
      year={2025},
      doi={10.1016/j.compag.2024.109673},
}
```

