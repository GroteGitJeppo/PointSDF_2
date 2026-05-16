import os
from pathlib import Path

import math
import numpy as np
import open3d as o3d
import pandas as pd
import torch
import torch_fpsample
from torch.utils.data import Dataset, Sampler
from torch_geometric.data import Data
from tqdm import tqdm

from data.sdf_samples import resolve_samples_npz


def _remove_nans(t: torch.Tensor) -> torch.Tensor:
    return t[~torch.isnan(t[:, 3])]


class PointCloudLatentDataset(Dataset):
    """
    Dataset for Stage 2 (encoder training).

    At construction time, all partial point clouds are loaded from disk once,
    centred, normalised, and FPS-sampled into RAM (mirrors corepp DeepSDF
    ``SDFSamples(load_ram=True)``).  Latent codes and optional SDF samples are
    also pre-loaded.  ``__getitem__`` only clones the cached cloud and applies
    augmentation.

    Latent codes must be saved as individual tensors: <latent_dir>/<label>.pth

    Optionally loads SDF samples from samples.npz for each label so that the
    training loop can also compute an end-to-end SDF loss through the frozen
    decoder.  When sdf_data_dir is provided, only labels whose samples.npz
    exists are kept (a warning is printed for excluded labels).

    Each __getitem__ returns a PyG Data object with:
        data.pos        — (num_points, 3) centred, normalised, FPS-sampled point cloud
        data.scale      — (1,) ratio original_half_extent / normalize_half_extent;
                          multiply normalised coords by this to recover metric coords
        data.latent     — (1, latent_size) ground-truth latent code
        data.sdf_xyz    — (1, sdf_samples_per_shape, 3) SDF query points
                          (only when sdf_data_dir is configured)
        data.sdf_gt     — (1, sdf_samples_per_shape, 1) ground-truth SDF values
                          (only when sdf_data_dir is configured)
    """

    def __init__(
        self,
        data_root: str,
        splits_csv: str,
        latent_dir: str,
        split: str = 'train',
        num_points: int = 1024,
        apply_augmentation: bool = True,
        augmentation_cfg: dict | None = None,
        sdf_data_dir: str | None = None,
        sdf_samples_per_shape: int = 1024,
        sdf_clamp_value: float | None = None,
        normalize_half_extent: float = 0.05,
    ):
        self.latent_dir = latent_dir
        self.split = split
        self.num_points = num_points
        self.apply_augmentation = apply_augmentation
        self._sdf_samples_per_shape = sdf_samples_per_shape
        self._sdf_clamp = sdf_clamp_value
        self.normalize_half_extent = normalize_half_extent
        self.augmentation_cfg = self._parse_augmentation_cfg(augmentation_cfg)

        splits_df = pd.read_csv(splits_csv)
        labels = set(splits_df[splits_df['split'] == split]['label'].astype(str))

        # Build candidate sample list — (ply_path, label, latent_path)
        candidates: list[tuple[str, str, str]] = []
        for ply_file in Path(data_root).rglob('*.ply'):
            label = ply_file.parent.name
            if label not in labels:
                continue
            latent_path = os.path.join(latent_dir, f'{label}.pth')
            if not os.path.exists(latent_path):
                continue
            candidates.append((str(ply_file), label, latent_path))

        if not candidates:
            raise RuntimeError(
                f"No (ply, latent) pairs found for split='{split}'. "
                f"Check that '{latent_dir}' contains <label>.pth files "
                f"produced by train_deepsdf.py."
            )

        # Load SDF data (optional) — pre-loaded into RAM, keyed by label
        self._sdf_ram: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None
        if sdf_data_dir is not None:
            self._sdf_ram = {}
            unique_labels = {lbl for _, lbl, _ in candidates}
            missing = []
            for label in sorted(unique_labels):
                path = resolve_samples_npz(sdf_data_dir, label)
                if path is None:
                    missing.append(label)
                    continue
                raw = np.load(path)
                pos = _remove_nans(torch.from_numpy(np.asarray(raw['pos'], dtype=np.float32)))
                neg = _remove_nans(torch.from_numpy(np.asarray(raw['neg'], dtype=np.float32)))
                self._sdf_ram[label] = (pos, neg)

            if missing:
                print(
                    f"PointCloudLatentDataset [{split}]: WARNING — "
                    f"{len(missing)} label(s) have no samples.npz and will be excluded "
                    f"from training: {missing[:5]}{'...' if len(missing) > 5 else ''}"
                )
                # Remove those labels so every item in the batch has SDF data
                candidates = [(p, l, lp) for p, l, lp in candidates if l not in missing]

            print(
                f"PointCloudLatentDataset [{split}]: SDF data loaded for "
                f"{len(self._sdf_ram)} labels, {sdf_samples_per_shape} samples/shape"
            )

        self.samples: list[tuple[str, str, str]] = candidates

        if not self.samples:
            raise RuntimeError(
                f"No samples remaining for split='{split}' after SDF filtering."
            )

        # Eagerly load all latent codes into RAM so __getitem__ never hits the
        # filesystem per step (important on networked / HPC storage).
        self.latents_dict: dict[str, torch.Tensor] = self._load_latents_dict(
            {lbl: lp for _, lbl, lp in self.samples}
        )

        self._pc_cache: list[tuple[torch.Tensor, float]] = self._preload_point_clouds()

        print(f"PointCloudLatentDataset [{split}]: {len(self.samples)} samples")

    # ------------------------------------------------------------------

    @staticmethod
    def _load_latents_dict(label_to_path: dict[str, str]) -> dict[str, torch.Tensor]:
        """Load all per-fruit latent codes into RAM once at construction time.

        Mirrors corepp's ``get_latents_dict`` but uses ``weights_only=True``
        for safety and detaches the tensors so they are never part of any
        computation graph.
        """
        latents: dict[str, torch.Tensor] = {}
        for label, path in label_to_path.items():
            latents[label] = torch.load(path, weights_only=True, map_location='cpu').detach().reshape(-1)
        return latents

    def _load_processed_point_cloud(self, ply_path: str) -> tuple[torch.Tensor, float]:
        """Load a PLY, centre, normalise, and FPS to num_points."""
        pcd = o3d.io.read_point_cloud(ply_path)
        points = torch.tensor(np.asarray(pcd.points), dtype=torch.float32)
        points = self._center_points(points)
        points, scale = self._normalize_points(points)
        points = self._enforce_num_points(points)
        return points, scale

    def _preload_point_clouds(self) -> list[tuple[torch.Tensor, float]]:
        """Load every scan into RAM once (centred, normalised, FPS-sampled)."""
        cache: list[tuple[torch.Tensor, float]] = []
        for ply_path, _, _ in tqdm(
            self.samples,
            desc=f"Preloading point clouds [{self.split}]",
            unit="scan",
        ):
            cache.append(self._load_processed_point_cloud(ply_path))
        return cache

    def _parse_augmentation_cfg(self, cfg: dict | None) -> dict:
        # Backward-compatible defaults approximate previous behaviour.
        defaults = {
            "jitter_std": 5e-4,
            "jitter_clip": 1e-3,
            "rotate_x_deg": 2.0,
            "rotate_y_deg": 2.0,
            "rotate_z_deg": 90.0,
            "max_shear": 0.0,
            "shear_prob": 1.0,
            "flip_x_prob": 0.5,
            "scale_min": 1.0,
            "scale_max": 1.0,
            "point_dropout_prob": 0.0,
            "point_dropout_min": 0.0,
            "point_dropout_max": 0.0,
            "occlusion_prob": 0.0,
            "occlusion_ratio_min": 0.0,
            "occlusion_ratio_max": 0.0,
        }
        if cfg is None:
            return defaults
        out = defaults.copy()
        out.update(cfg)
        return out

    def _center_points(self, points: torch.Tensor) -> torch.Tensor:
        center = points.mean(dim=0, keepdim=True)
        return points - center

    def _normalize_points(self, points: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Scale points isotropically so the max absolute coordinate equals
        normalize_half_extent.  Returns the normalised points and the scale
        ratio (original_half_extent / normalize_half_extent) needed to recover
        metric coordinates: metric_coord = normalised_coord * scale.
        """
        max_half_extent = points.abs().max().item()
        if max_half_extent < 1e-6:
            return points, 1.0
        scale = max_half_extent / self.normalize_half_extent
        return points / scale, scale

    def _enforce_num_points(self, points: torch.Tensor) -> torch.Tensor:
        n = points.size(0)
        if n == self.num_points:
            return points
        if n > self.num_points:
            sampled, _ = torch_fpsample.sample(points, self.num_points)
            return sampled
        if n == 0:
            return torch.zeros((self.num_points, 3), dtype=torch.float32)
        extra_idx = torch.randint(0, n, (self.num_points - n,))
        extras = points[extra_idx]
        return torch.cat([points, extras], dim=0)

    @staticmethod
    def _rotation_matrix_xyz(rx: float, ry: float, rz: float) -> torch.Tensor:
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)
        rot_x = torch.tensor([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
        rot_y = torch.tensor([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
        rot_z = torch.tensor([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
        return rot_z @ rot_y @ rot_x

    @staticmethod
    def _apply_shear_x(points: torch.Tensor, max_shear: float) -> torch.Tensor:
        """X-direction shear matching corepp/data_preparation/augment.py.

        x' = x + s0 * y + s1 * z,  with s0, s1 ~ Uniform(-max_shear, max_shear).
        """
        if max_shear <= 0.0:
            return points
        shear = np.random.uniform(-max_shear, max_shear, size=(2,))
        s0, s1 = float(shear[0]), float(shear[1])
        shear_mat = torch.tensor(
            [[1.0, s0, s1], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=points.dtype,
            device=points.device,
        )
        return points @ shear_mat.T

    def _dropout_points(self, points: torch.Tensor, drop_ratio: float) -> torch.Tensor:
        n = points.size(0)
        if n < 2 or drop_ratio <= 0.0:
            return points
        drop_n = int(n * drop_ratio)
        if drop_n <= 0:
            return points
        keep_n = max(n - drop_n, 1)
        keep_idx = torch.randperm(n)[:keep_n]
        kept = points[keep_idx]
        if keep_n == n:
            return kept
        refill_idx = torch.randint(0, keep_n, (n - keep_n,))
        refill = kept[refill_idx]
        return torch.cat([kept, refill], dim=0)

    def _occlude_points(self, points: torch.Tensor, occ_ratio: float) -> torch.Tensor:
        n = points.size(0)
        if n < 2 or occ_ratio <= 0.0:
            return points
        occ_n = int(n * occ_ratio)
        if occ_n <= 0:
            return points
        center_idx = torch.randint(0, n, (1,)).item()
        center = points[center_idx : center_idx + 1]
        dist = ((points - center) ** 2).sum(dim=1)
        remove_idx = torch.topk(dist, k=min(occ_n, n - 1), largest=False).indices
        keep_mask = torch.ones(n, dtype=torch.bool)
        keep_mask[remove_idx] = False
        kept = points[keep_mask]
        keep_n = kept.size(0)
        refill_idx = torch.randint(0, keep_n, (n - keep_n,))
        refill = kept[refill_idx]
        return torch.cat([kept, refill], dim=0)

    def _augment_points(self, points: torch.Tensor) -> torch.Tensor:
        cfg = self.augmentation_cfg

        # Random rotations in degrees around xyz.
        rx = math.radians(np.random.uniform(-cfg["rotate_x_deg"], cfg["rotate_x_deg"]))
        ry = math.radians(np.random.uniform(-cfg["rotate_y_deg"], cfg["rotate_y_deg"]))
        rz = math.radians(np.random.uniform(-cfg["rotate_z_deg"], cfg["rotate_z_deg"]))
        rot = self._rotation_matrix_xyz(rx, ry, rz).to(points.dtype)
        points = points @ rot.T

        # X-direction shear (CoRe++ augment.py: T[0,1], T[0,2]).
        if np.random.rand() < float(cfg["shear_prob"]):
            points = self._apply_shear_x(points, float(cfg["max_shear"]))

        # Left-right flip around x (physically plausible for conveyor setup).
        if np.random.rand() < float(cfg["flip_x_prob"]):
            points[:, 0] = -points[:, 0]

        # Isotropic scaling.
        s_min = float(cfg["scale_min"])
        s_max = float(cfg["scale_max"])
        if s_max < s_min:
            s_min, s_max = s_max, s_min
        if s_max > 0:
            scale = float(np.random.uniform(s_min, s_max))
            points = points * scale

        # Gaussian jitter with clipping.
        std = float(cfg["jitter_std"])
        if std > 0.0:
            noise = torch.randn_like(points) * std
            clip = float(cfg["jitter_clip"])
            if clip > 0.0:
                noise = torch.clamp(noise, -clip, clip)
            points = points + noise

        # Random point dropout.
        if np.random.rand() < float(cfg["point_dropout_prob"]):
            dr_min = float(cfg["point_dropout_min"])
            dr_max = float(cfg["point_dropout_max"])
            if dr_max < dr_min:
                dr_min, dr_max = dr_max, dr_min
            drop_ratio = float(np.random.uniform(dr_min, dr_max))
            points = self._dropout_points(points, max(0.0, min(drop_ratio, 0.95)))

        # Occlusion-like local removal/refill.
        if np.random.rand() < float(cfg["occlusion_prob"]):
            or_min = float(cfg["occlusion_ratio_min"])
            or_max = float(cfg["occlusion_ratio_max"])
            if or_max < or_min:
                or_min, or_max = or_max, or_min
            occ_ratio = float(np.random.uniform(or_min, or_max))
            points = self._occlude_points(points, max(0.0, min(occ_ratio, 0.95)))

        return points

    def get_label_to_indices(self) -> dict[str, list[int]]:
        """Return a mapping from label → list of dataset indices for that label.

        Used by TuberBatchSampler to guarantee same-label pairs per batch.
        """
        mapping: dict[str, list[int]] = {}
        for i, (_, label, _) in enumerate(self.samples):
            mapping.setdefault(label, []).append(i)
        return mapping

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Data:
        _, label, _ = self.samples[idx]

        points, scale = self._pc_cache[idx]
        points = points.clone()
        if self.apply_augmentation:
            points = self._augment_points(points)
        data = Data(pos=points)
        data.scale = torch.tensor([scale], dtype=torch.float)

        latent = self.latents_dict[label]  # (latent_size,) — pre-loaded at init
        # Shape (1, latent_size) so PyG's Batch concatenates to (B, latent_size)
        data.latent = latent.unsqueeze(0)

        # Expose the tuber label so the training loop can form contrastive pairs
        # (mirrors corepp's fruit_id). PyG Batch collates strings as a plain list.
        data.label = label

        # SDF samples — shape (1, N_sdf, 3/1) so Batch gives (B, N_sdf, 3/1)
        if self._sdf_ram is not None and label in self._sdf_ram:
            pos_t, neg_t = self._sdf_ram[label]
            half = self._sdf_samples_per_shape // 2

            pos_idx = torch.randint(0, pos_t.size(0), (half,))
            neg_idx = torch.randint(0, neg_t.size(0), (half,))
            sdf_samples = torch.cat([pos_t[pos_idx], neg_t[neg_idx]], dim=0)  # (N_sdf, 4)

            if self._sdf_clamp is not None:
                sdf_samples = sdf_samples.clone()
                sdf_samples[:, 3] = torch.clamp(
                    sdf_samples[:, 3], -self._sdf_clamp, self._sdf_clamp
                )

            data.sdf_xyz = sdf_samples[:, :3].unsqueeze(0)   # (1, N_sdf, 3)
            data.sdf_gt  = sdf_samples[:, 3:4].unsqueeze(0)  # (1, N_sdf, 1)

        return data


class TuberBatchSampler(Sampler):
    """Batch sampler that guarantees same-label (same-tuber) pairs in every batch.

    Each batch contains exactly ``k_scans`` random scans from each of
    ``n_labels`` randomly chosen tubers, giving an effective batch size of
    ``n_labels * k_scans``.  This mirrors corepp's natural same-label
    collision rate and makes the attraction term of AttRepLoss actually fire
    every step rather than ~30% of the time under uniform random sampling.

    Args:
        label_to_indices: dict mapping label → list of dataset indices,
                          as returned by PointCloudLatentDataset.get_label_to_indices().
        n_labels:         number of distinct tubers per batch.
        k_scans:          number of scans drawn per tuber per batch.
        drop_last:        if True, drop the last incomplete batch.
    """

    def __init__(
        self,
        label_to_indices: dict[str, list[int]],
        n_labels: int,
        k_scans: int,
        drop_last: bool = False,
    ):
        self.n_labels = n_labels
        self.k_scans = k_scans
        self.drop_last = drop_last

        # Only use labels that have at least k_scans scans available
        self.label_to_indices = {
            lbl: idxs for lbl, idxs in label_to_indices.items() if len(idxs) >= k_scans
        }
        skipped = len(label_to_indices) - len(self.label_to_indices)
        if skipped:
            print(
                f'TuberBatchSampler: skipped {skipped} label(s) with < {k_scans} scans. '
                f'{len(self.label_to_indices)} labels available.'
            )
        self._labels = sorted(self.label_to_indices.keys())

    @property
    def batch_size(self) -> int:
        return self.n_labels * self.k_scans

    def __iter__(self):
        labels = self._labels.copy()
        np.random.shuffle(labels)

        n_full = len(labels) // self.n_labels
        for i in range(n_full):
            chunk = labels[i * self.n_labels: (i + 1) * self.n_labels]
            batch = []
            for lbl in chunk:
                idxs = self.label_to_indices[lbl]
                chosen = np.random.choice(idxs, size=self.k_scans, replace=False)
                batch.extend(chosen.tolist())
            yield batch

        remainder = labels[n_full * self.n_labels:]
        if remainder and not self.drop_last:
            batch = []
            for lbl in remainder:
                idxs = self.label_to_indices[lbl]
                chosen = np.random.choice(idxs, size=self.k_scans, replace=False)
                batch.extend(chosen.tolist())
            yield batch

    def __len__(self) -> int:
        n = len(self._labels)
        if self.drop_last:
            return n // self.n_labels
        return (n + self.n_labels - 1) // self.n_labels
