# PointSDF_2 — 3D Shape Completion for Potato Tuber Volume Estimation

## What this does

PointSDF_2 estimates the **volume of a potato tuber** from a single partial point cloud (the kind you get from an RGB-D camera looking down at a conveyor belt). The potato is only visible from one side, so the shape has to be *completed* before you can measure its volume.

The approach: learn what a complete potato SDF (signed distance function) looks like, then train a neural network to predict that representation from a partial scan. Volume is extracted by finding all grid points inside the predicted surface and computing their convex hull.

```
Partial point cloud (.ply, from conveyor belt)
 │
 ▼
PointNet++ encoder    →  latent code  z ∈ ℝ³²
 │
 ▼
DeepSDF decoder       →  SDF value at any 3D point
(8-layer MLP, dim 512, skip connection at layer 4)
 │
 ▼
Convex hull of grid points where SDF < 0  →  Volume (mL)
```

Training is done in **two stages** with a **reconstruction step** in between:

1. **Stage 1** — teach the *decoder* what potato shapes look like, using complete 3D SDF samples from laser scans. Each potato gets its own latent code that is optimised jointly with the decoder.
2. **Reconstruct** — for each trained shape, run a fast per-shape optimisation using the frozen decoder to produce clean, decoder-consistent latent codes. Also used to pick the best Stage 1 checkpoint.
3. **Stage 2** — teach the *encoder* to predict those latent codes directly from a partial point cloud. The decoder is frozen; only the encoder trains.
4. **Evaluate** — run the full encoder → decoder → convex hull pipeline on the test set and report volume accuracy.

---

## Installation

All code runs on a remote Debian server with an NVIDIA A40 GPU.

```bash
# Create and activate the conda environment
conda env create -f environment.yaml
conda activate pointsdf

# Build and install the C++ farthest-point sampling (FPS) extension
cd pytorch_fpsample && pip install --no-deps --no-build-isolation . && cd ..

# Verify it worked
python -c "import torch_fpsample; print('OK')"
```

Key dependencies: Python 3.12.3, PyTorch 2.8.0 (CUDA 12.8), PyTorch Geometric 2.7.0, Open3D 0.19.0.

---

## Data preparation

The [3DPotatoTwin](https://huggingface.co/datasets/UTokyo-FieldPhenomics-Lab/3DPotatoTwin) dataset provides partial point clouds, SfM meshes, and ground-truth volumes.

**All commands below are run from inside `PointSDF_2/`.**

### Step 0a — Generate SDF samples from complete meshes (Stage 1 input)

Each potato needs a file `samples.npz` containing thousands of 3D points sampled near its complete surface, each labelled with its signed distance to the surface (negative = inside, positive = outside). These come from the SfM laser scans.

```bash
python data/prepare_dataset.py sdf \
    --src  data/3DPotatoTwin/2_sfm/4_pcd \
    --out  data/3DPotatoTwin/sdfsamples/potato \
    --ply_pattern "*_20000.ply"
```

This writes `data/3DPotatoTwin/sdfsamples/potato/<label>/samples.npz` for every potato label.

### Step 0b — Generate partial point clouds from RGB-D images (Stage 2 input)

```bash
python data/prepare_dataset.py pcd \
    --img_root   data/3DPotatoTwin/1_rgbd/1_image \
    --intrinsics data/3DPotatoTwin/1_rgbd/0_camera_intrinsics/realsense_d405_camera_intrinsic.json \
    --out        data/3DPotatoTwin/1_rgbd/2_pcd
```

This writes one `.ply` file per RGB-D frame under `data/3DPotatoTwin/1_rgbd/2_pcd/<label>/`.

---

## Full training pipeline

### Step 1 — Train the SDF decoder (Stage 1)

**What it does:** Trains an 8-layer MLP (the decoder) together with one latent code per potato on complete SDF samples. By the end, the decoder can turn any latent code into a full 3D shape, and each potato has a latent code that encodes its shape. Uses only the *train* split — the val split is held out for checkpoint selection.

**Config file:** `configs/train_deepsdf.yaml` — open it to check/adjust paths before running.

```bash
cd PointSDF_2/
python train_deepsdf.py --config configs/train_deepsdf.yaml
```

**Optional flags:**

```
--continue-from latest    Resume from the most recent saved checkpoint.
--continue-from 500       Resume from epoch 500 specifically.
--batch_split 2           Split each shape's samples into 2 gradient chunks
                          (reduces peak GPU memory; use if you get OOM errors).
--verbose                 Print debug-level logs.
```

**Output** — everything is written to `weights/deepsdf/<DD_MM_HHMMSS>/`:

```
weights/deepsdf/<run>/
├── ModelParameters/
│   ├── 0.pth          ← checkpoint at epoch 0 (initial weights)
│   ├── 10.pth         ← checkpoint every 10 epochs
│   ├── 500.pth        ← checkpoint at epoch 500
│   └── latest.pth     ← always overwritten with the most recent saved epoch
├── LatentCodes/
│   ├── 0.pth          ← latent codes at each checkpoint epoch
│   ├── 500.pth
│   └── latest.pth
├── OptimizerParameters/
│   └── ...            ← Adam optimiser state (needed to resume training)
├── Logs.pth           ← loss / lr / timing history
├── config.yaml        ← copy of the config used for this run
└── specs.json         ← corepp-compatible spec file
```

Training runs for **1001 epochs** by default. At epoch 500 an extra snapshot is also saved.

---

### Step 2 — Select the best checkpoint and generate latent targets (Reconstruct)

**What it does:** After Stage 1, you have ~100 checkpoints saved. This step finds the *best* one by sweeping the val split across all checkpoints in a single command, then generates the high-quality latent codes that Stage 2 will use as training targets.

The process for each potato shape: start with a random latent code, freeze the decoder, run 800 steps of gradient descent to make the decoder reproduce that potato's SDF as accurately as possible. The result is a *decoder-consistent* latent — the encoder will be trained to predict exactly these.

#### Step 2a — Find the best checkpoint (sweep the val split)

Run this once — it automatically discovers every checkpoint saved in `ModelParameters/`, tests every N-th epoch on the val split, then prints a sorted leaderboard so you can pick the best one.

```bash
python reconstruct.py \
    --decoder_config configs/train_deepsdf.yaml \
    --experiment_dir weights/deepsdf/<run> \
    --split val \
    --all-checkpoints        # tests every 10th epoch (corepp default)
```

**Replace `<run>` with the actual run folder name (e.g. `09_04_210939`).**

You can control how many epochs are skipped between tests:

```
--all-checkpoints        Tests every 10th epoch: 10, 20, 30, … (default stride)
--all-checkpoints 50     Tests every 50th epoch: 50, 100, 150, … (faster sweep)
--all-checkpoints 1      Tests every single saved checkpoint (thorough but slow)
--iters 800              Optimisation steps per shape (default 800, matching corepp)
--verbose                Show per-shape Chamfer distances during the sweep
```

Sweep mode always computes Chamfer distances and always reuses already-computed latents (`--skip`) if you interrupted a previous run. SDF data is loaded into RAM only once regardless of how many checkpoints are tested.

At the end the script prints a sorted leaderboard and tells you exactly which paths to use next:

```
Checkpoint sweep results — split='val', step=10
-------------------------------------------
  epoch  |  Chamfer (mm)
-------------------------------------------
    500  |     1.842  ← BEST
    600  |     1.953
    400  |     2.104
    300  |     2.871
    ...
-------------------------------------------

Best checkpoint: epoch 500  (Chamfer = 1.842 mm)

Next steps — update configs/train_encoder.yaml:
  decoder_weights: weights/deepsdf/<run>/ModelParameters/500.pth
  latent_dir:      (run with --checkpoint 500 --split train to generate)
```

The epoch with the lowest Chamfer distance on the val split is your best epoch `E*`.

**Output** (written during the sweep, reused on subsequent runs with `--skip`):

```
weights/deepsdf/<run>/Reconstructions/<E>/Codes/val/<label>.pth   ← optimised latent per val shape
```

#### Step 2b — Generate latent targets for Stage 2 (train split, best checkpoint)

Once you have identified `E*`, run reconstruction on the **train** split with that checkpoint:

```bash
python reconstruct.py \
    --decoder_config configs/train_deepsdf.yaml \
    --experiment_dir weights/deepsdf/<run> \
    --checkpoint <E*> \
    --split train
```

**Output:**

```
weights/deepsdf/<run>/Reconstructions/<E*>/Codes/train/<label>.pth   ← one file per train shape
```

The script prints exactly which path to put in the Stage 2 config, e.g.:

```
To use as Stage 2 targets, set in configs/train_encoder.yaml:
  latent_dir: weights/deepsdf/<run>/Reconstructions/<E*>/Codes/train
```

---

### Step 3 — Update the Stage 2 config

Before running Stage 2, open `configs/train_encoder.yaml` and update two lines with the paths from Step 2:

```yaml
latent_dir:      weights/deepsdf/<run>/Reconstructions/<E*>/Codes/train
decoder_weights: weights/deepsdf/<run>/ModelParameters/<E*>.pth
```

Everything else in the config can stay as-is for a standard run.

---

### Step 4 — Train the PointNet++ encoder (Stage 2)

**What it does:** Loads the best Stage 1 decoder and *freezes* it (its weights never change). Trains a PointNet++ encoder that takes a partial point cloud and predicts the latent code that the frozen decoder needs to reconstruct the full potato shape. Loss = MSE between predicted and target latents + contrastive loss (pulls latents of the same potato together, pushes different potatoes apart).

**Config file:** `configs/train_encoder.yaml` — must be updated with the paths from Step 3 above.

```bash
python train.py --config configs/train_encoder.yaml
```

No other flags are needed for a standard run.

**Output** — everything is written to `weights/encoder/<DD_MM_HHMMSS>/`:

```
weights/encoder/<run>/
├── encoder.pth             ← best encoder weights (lowest val loss so far)
├── checkpoint.pth          ← full checkpoint: encoder + decoder + optimiser state
├── snapshots/
│   ├── 0010/checkpoint.pth ← periodic snapshots every 10 epochs
│   ├── 0020/checkpoint.pth
│   └── ...
├── config.yaml             ← copy of the config
└── events.out.tfevents.*   ← TensorBoard logs
```

Training logs to TensorBoard. To monitor:

```bash
tensorboard --logdir weights/encoder/<run>
```

**Select the best Stage 2 checkpoint:** after training, run `test.py` (Step 5) on each snapshot and pick the one with the lowest volume RMSE on the val split. `snapshot_frequency` in the config controls how often snapshots are saved.

---

### Step 5 — Evaluate on the test set

**What it does:** Loads an encoder + decoder checkpoint, runs the full pipeline on every test-split potato (encode → decode SDF on a 64³ grid → convex hull → volume in mL), then compares predicted volumes to ground-truth volumes.

```bash
python test.py \
    --config     configs/train_encoder.yaml \
    --checkpoint weights/encoder/<run>/checkpoint.pth
```

```
--checkpoint    Path to any checkpoint.pth from Stage 2 training (best model or a specific snapshot).
```

**Output — printed to console:**

```
Test results (49/51 with valid meshes):
  MAE volume:  18.4 mL
  RMSE volume: 22.6 mL
  R²:          0.912
  Chamfer dist: 2.8 mm  (if sdf_data_dir is set in decoder config)
  Avg exec:    9.9 ms

=== Per cultivar ===
  Corolle:  n=9  | MAE=17.6 mL | R²=0.934
  Sayaka:   n=28 | MAE=24.8 mL | R²=0.891
  Kitahime: n=12 | MAE=19.3 mL | R²=0.921
```

Results are also saved to `weights/encoder/<run>/test_results.csv`.

---

## Quick-reference command summary

Run all commands from inside `PointSDF_2/`. Replace `<run>` and `<E*>` with your actual folder/epoch.

```bash
# 0. Prepare data (one-time)
python data/prepare_dataset.py sdf --src data/3DPotatoTwin/2_sfm/4_pcd \
    --out data/3DPotatoTwin/sdfsamples/potato --ply_pattern "*_20000.ply"
python data/prepare_dataset.py pcd --img_root data/3DPotatoTwin/1_rgbd/1_image \
    --intrinsics data/3DPotatoTwin/1_rgbd/0_camera_intrinsics/realsense_d405_camera_intrinsic.json \
    --out data/3DPotatoTwin/1_rgbd/2_pcd

# 1. Train decoder (Stage 1) — ~1001 epochs, several hours on A40
python train_deepsdf.py --config configs/train_deepsdf.yaml

# 2a. Find best checkpoint — sweep val split, every 10th epoch (prints a sorted leaderboard)
python reconstruct.py -c configs/train_deepsdf.yaml \
    --experiment_dir weights/deepsdf/<run> --split val --all-checkpoints

# 2b. Generate latent targets for Stage 2 using the best epoch E*
python reconstruct.py -c configs/train_deepsdf.yaml \
    --experiment_dir weights/deepsdf/<run> --checkpoint <E*> --split train

# 3. Update configs/train_encoder.yaml with latent_dir and decoder_weights paths (manual edit)

# 4. Train encoder (Stage 2) — 500 epochs
python train.py --config configs/train_encoder.yaml

# 5. Evaluate on test set
python test.py --config configs/train_encoder.yaml \
    --checkpoint weights/encoder/<run>/checkpoint.pth
```

---

## Repository structure

```
PointSDF_2/
├── configs/
│   ├── train_deepsdf.yaml      # Stage 1: decoder training hyperparameters
│   └── train_encoder.yaml      # Stage 2: encoder training hyperparameters + data paths
├── data/
│   ├── encoder_dataset.py      # Stage 2 dataset: partial PLY + latent code + optional SDF
│   ├── sdf_samples.py          # resolve_samples_npz() — finds samples.npz for a label
│   ├── sdf_scene_dataset.py    # Stage 1 dataset: SDF samples loaded into RAM per shape
│   └── prepare_dataset.py      # data prep: pcd (RGB-D → PLY) and sdf (PLY → samples.npz)
├── models/
│   ├── encoder.py              # PointNet++ encoder: point cloud → latent code
│   ├── decoder.py              # DeepSDF MLP decoder: (latent, xyz) → SDF scalar
│   └── pointsdf.py             # combined PointSDF wrapper (for inference)
├── utils/
│   ├── sdf_helpers.py          # get_volume_coords, sdf2mesh, chamfer_distance
│   └── visualize.py            # visualize_point_cloud helpers
├── pytorch_fpsample/           # C++ farthest-point sampling extension (must be compiled)
├── train_deepsdf.py            # Stage 1: train decoder + latent codes jointly
├── reconstruct.py              # Between stages: test-time latent optimisation + checkpoint selection
├── train.py                    # Stage 2: train encoder with frozen decoder
├── test.py                     # Evaluate encoder → decoder → volume on test split
└── environment.yaml            # conda environment definition
```

---

## Config file reference

### `configs/train_deepsdf.yaml` (Stage 1)


| Key                          | Default                               | What it controls                                                 |
| ---------------------------- | ------------------------------------- | ---------------------------------------------------------------- |
| `sdf_data_dir`               | `data/3DPotatoTwin/sdfsamples/potato` | Where to find `samples.npz` files                                |
| `splits_csv`                 | `data/3DPotatoTwin/splits.csv`        | Which labels belong to which split                               |
| `stage1_splits`              | `[train]`                             | Which splits to include in decoder training                      |
| `latent_size`                | `32`                                  | Dimensionality of each shape's latent code                       |
| `inner_dim`                  | `512`                                 | Width of each hidden layer in the decoder MLP                    |
| `num_layers`                 | `8`                                   | Number of hidden layers                                          |
| `skip_connections`           | `true`                                | Whether to concatenate input at layer 4 (the DeepSDF skip)       |
| `epochs`                     | `1001`                                | Total training epochs                                            |
| `clamp_value`                | `0.1`                                 | SDF values are clamped to ±0.1 m (focus on near-surface region)  |
| `code_bound`                 | `1.0`                                 | Maximum L2 norm of any latent code (prevents blow-up)            |
| `code_regularization_lambda` | `0.0001`                              | Weight of the latent L2 regularisation loss                      |
| `reg_ramp_epochs`            | `100`                                 | Latent reg is ramped from 0 to full weight over this many epochs |
| `snapshot_frequency`         | `10`                                  | Save a checkpoint every N epochs                                 |
| `additional_snapshots`       | `[0, 500]`                            | Always save at these specific epochs too                         |
| `seed`                       | `42`                                  | Random seed for reproducibility                                  |


### `configs/train_encoder.yaml` (Stage 2)


| Key                  | Default                          | What it controls                                                               |
| -------------------- | -------------------------------- | ------------------------------------------------------------------------------ |
| `data_root`          | `data/3DPotatoTwin/1_rgbd/2_pcd` | Root folder of partial point cloud `.ply` files                                |
| `splits_csv`         | `data/3DPotatoTwin/splits.csv`   | Train/val/test label assignments                                               |
| `latent_dir`         | *(must be set after Step 2b)*    | Folder of `<label>.pth` latent target files                                    |
| `decoder_weights`    | *(must be set after Step 2a)*    | Path to the best Stage 1 `ModelParameters/<E*>.pth`                            |
| `decoder_config`     | `configs/train_deepsdf.yaml`     | Used to read the decoder architecture                                          |
| `num_points`         | `1024`                           | Number of points to sample from each partial point cloud                       |
| `epochs`             | `500`                            | Total training epochs                                                          |
| `batch_size`         | `16`                             | Shapes per batch                                                               |
| `lr`                 | `0.0001`                         | Encoder learning rate                                                          |
| `lr_gamma`           | `0.995`                          | Exponential LR decay per epoch                                                 |
| `sigma_regulariser`  | `0.01`                           | Weight of the latent L2 regularisation on encoder outputs                      |
| `contrastive_loss`   | `true`                           | Enable AttRepLoss (attract same-tuber latents, repel different)                |
| `lambda_attraction`  | `0.05`                           | Weight of the contrastive loss term                                            |
| `delta_rep`          | `0.5`                            | Repulsion margin — latents of different tubers must be at least this far apart |
| `sdf_loss_weight`    | `0.1`                            | Weight of the optional end-to-end SDF loss through the frozen decoder          |
| `snapshot_frequency` | `10`                             | Save a snapshot every N epochs for post-hoc best-checkpoint selection          |
| `grid_resolution`    | `64`                             | SDF query grid resolution for inference (64³ = 262 144 points)                 |
| `grid_bbox`          | `0.15`                           | Half-size of the query bounding box in metres (±0.15 m cube)                   |


---

## Key design choices


| Choice                                                | Rationale                                                                                                                                                                                                              |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Two-stage training (decoder first, then encoder)      | The decoder learns a general shape space from complete SDF data; the encoder then maps partial observations into that space. Training them separately is simpler and avoids the encoder interfering with SDF learning. |
| Test-time latent optimisation for Stage 2 targets     | After Stage 1, each training latent is further refined with the *fixed* final decoder. This gives the encoder cleaner, more consistent targets than the raw autodecoder embeddings.                                    |
| Checkpoint selection by Chamfer distance on val split | The decoder's reconstruction quality on unseen shapes (val) peaks before the decoder memorises the training shapes. Chamfer distance on the val split is the right signal.                                             |
| Convex hull (not marching cubes) for volume           | Potatoes are roughly convex; convex hull is always watertight, GPU-accelerated via Open3D, and avoids marching cubes grid resolution artefacts.                                                                        |
| Latent size 32                                        | CoRe++ systematic study found latent size 32 optimal for potato tubers, balancing shape representational capacity and generalisation.                                                                                  |
| Decoder width 512                                     | Matches the original DeepSDF and CoRe++ configuration.                                                                                                                                                                 |
| AttRepLoss (contrastive) in Stage 2                   | Pulls latent codes of multiple images of the *same* potato together and pushes codes of *different* potatoes apart. Makes the latent space more structured, which helps the encoder generalise.                        |
| MSE loss for encoder                                  | Penalises large latent deviations more than L1; validated by CoRe++ ablation.                                                                                                                                          |
| PointNet++ encoder (not RGB-D CNN)                    | Input is a partial 3D point cloud rather than an RGB-D image, making the approach agnostic to camera calibration and depth noise characteristics.                                                                      |


---

## Citation

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

