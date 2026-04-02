"""
Per-shape SDF sample dataset for Stage 1 (corepp-style).

One __getitem__ per potato: loads samples.npz from disk and returns a random
subsample of `samples_per_scene` points (half pos / half neg), matching
corepp's unpack_sdf_samples behaviour.
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.sdf_samples import resolve_samples_npz


def _remove_nans(tensor: torch.Tensor) -> torch.Tensor:
    mask = ~torch.isnan(tensor[:, 3])
    return tensor[mask]


class SDFSceneDataset(Dataset):
    """
    Args:
        sdf_data_dir: Root with one folder per label.
        splits_csv: CSV with columns label, split.
        split: Single split name, or list of names (e.g. ['train', 'val']). Test excluded by config.
        samples_per_scene: Total random SDF points per shape per __getitem__.
        clamp_value: If set, clamp sdf targets to [-v, v].
    """

    def __init__(
        self,
        sdf_data_dir: str,
        splits_csv: str,
        split: str | list[str],
        samples_per_scene: int,
        clamp_value: float | None = None,
    ):
        splits_df = pd.read_csv(splits_csv)

        if isinstance(split, list):
            labels = set(splits_df[splits_df['split'].isin(split)]['label'].astype(str))
        else:
            labels = set(splits_df[splits_df['split'] == split]['label'].astype(str))

        self.sdf_data_dir = sdf_data_dir
        self.samples_per_scene = samples_per_scene
        self.clamp_value = clamp_value

        self.labels: list[str] = []
        self.npz_paths: list[str] = []
        self.label_to_idx: dict[str, int] = {}

        for label in sorted(labels):
            path = resolve_samples_npz(sdf_data_dir, label)
            if path is None:
                continue
            idx = len(self.labels)
            self.label_to_idx[label] = idx
            self.labels.append(label)
            self.npz_paths.append(path)

        if not self.labels:
            raise RuntimeError(
                f"No samples.npz found under '{sdf_data_dir}' for split={split!r}."
            )
        if samples_per_scene < 2:
            raise ValueError('samples_per_scene must be at least 2 (pos/neg halves).')

        print(
            f'SDFSceneDataset [{split}]: {len(self.labels)} shapes, '
            f'{samples_per_scene} samples/scene per step'
        )

    def __len__(self) -> int:
        return len(self.labels)

    def _unpack_subsample(self, pos_tensor: torch.Tensor, neg_tensor: torch.Tensor) -> torch.Tensor:
        subsample = self.samples_per_scene
        half = int(subsample / 2)

        pos_tensor = _remove_nans(pos_tensor)
        neg_tensor = _remove_nans(neg_tensor)

        if pos_tensor.shape[0] == 0 or neg_tensor.shape[0] == 0:
            raise RuntimeError('pos or neg has no valid samples after NaN removal')

        if pos_tensor.shape[0] < half:
            random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
            sample_pos = torch.index_select(pos_tensor, 0, random_pos)
        else:
            pos_start = torch.randint(0, pos_tensor.shape[0] - half + 1, (1,)).item()
            sample_pos = pos_tensor[pos_start : pos_start + half]

        if neg_tensor.shape[0] < half:
            random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
            sample_neg = torch.index_select(neg_tensor, 0, random_neg)
        else:
            neg_start = torch.randint(0, neg_tensor.shape[0] - half + 1, (1,)).item()
            sample_neg = neg_tensor[neg_start : neg_start + half]

        samples = torch.cat([sample_pos, sample_neg], dim=0)
        if samples.shape[0] < subsample:
            pool = torch.cat([pos_tensor, neg_tensor], dim=0)
            pad_n = subsample - samples.shape[0]
            r = torch.randint(0, pool.shape[0], (pad_n,))
            samples = torch.cat([samples, pool[r]], dim=0)
        elif samples.shape[0] > subsample:
            samples = samples[:subsample]

        if self.clamp_value is not None:
            samples = samples.clone()
            samples[:, 3] = torch.clamp(
                samples[:, 3], -self.clamp_value, self.clamp_value
            )

        return samples.float()

    def __getitem__(self, idx: int) -> dict:
        data = np.load(self.npz_paths[idx])
        pos_tensor = torch.from_numpy(np.asarray(data['pos'], dtype=np.float32))
        neg_tensor = torch.from_numpy(np.asarray(data['neg'], dtype=np.float32))
        sdf_data = self._unpack_subsample(pos_tensor, neg_tensor)
        return {
            'sdf_data': sdf_data,
            'latent_idx': idx,
            'label': self.labels[idx],
        }

    @property
    def num_shapes(self) -> int:
        return len(self.labels)
