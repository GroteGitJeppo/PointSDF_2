import os
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd
import torch
import torch_fpsample
import torch_geometric.transforms as T
from torch.utils.data import Dataset
from torch_geometric.data import Data

from data.sdf_samples import resolve_samples_npz


def _remove_nans(t: torch.Tensor) -> torch.Tensor:
    return t[~torch.isnan(t[:, 3])]


class PointCloudLatentDataset(Dataset):
    """
    On-the-fly dataset for Stage 2 (encoder training).

    Loads partial point clouds (.ply) and their corresponding Stage 1 latent
    codes (.pth) on demand.  Latent codes must be saved by train_deepsdf.py
    as individual tensors: <latent_dir>/<label>.pth

    Optionally loads SDF samples from samples.npz for each label so that the
    training loop can also compute an end-to-end SDF loss through the frozen
    decoder.  When sdf_data_dir is provided, only labels whose samples.npz
    exists are kept (a warning is printed for excluded labels).

    Each __getitem__ returns a PyG Data object with:
        data.pos        — (num_points, 3) centred and FPS-sampled point cloud
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
        sdf_data_dir: str | None = None,
        sdf_samples_per_shape: int = 1024,
        sdf_clamp_value: float | None = None,
    ):
        self.latent_dir = latent_dir
        self.num_points = num_points
        self.apply_augmentation = apply_augmentation
        self.pre_transform = T.Center()
        self._sdf_samples_per_shape = sdf_samples_per_shape
        self._sdf_clamp = sdf_clamp_value

        if apply_augmentation:
            self.transform = T.Compose([
                T.RandomJitter(0.0005),
                T.RandomRotate(2, axis=0),    # small tilt (realistic)
                T.RandomRotate(2, axis=1),    # small tilt (realistic)
                T.RandomRotate(90, axis=2),   # full yaw — unconstrained on conveyor belt
                T.RandomFlip(axis=0, p=0.5),  # left-right flip — physically plausible
            ])
        else:
            self.transform = None

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

        print(f"PointCloudLatentDataset [{split}]: {len(self.samples)} samples")

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Data:
        ply_path, label, latent_path = self.samples[idx]

        pcd = o3d.io.read_point_cloud(ply_path)
        points = torch.tensor(np.asarray(pcd.points), dtype=torch.float)

        data = Data(pos=points)
        data = self.pre_transform(data)
        points = data.pos

        if points.size(0) > self.num_points:
            points, _ = torch_fpsample.sample(points, self.num_points)

        data = Data(pos=points)

        if self.apply_augmentation and self.transform is not None:
            data = self.transform(data)

        latent = torch.load(latent_path, weights_only=True, map_location='cpu').detach()  # (latent_size,)
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
