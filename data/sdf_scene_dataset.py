"""
Per-shape SDF sample dataset for Stage 1 (train_deepsdf.py).

All samples.npz files for the split are loaded into RAM at init (no disk read
per epoch). Each __getitem__ returns a random subsample of `samples_per_scene`
points (half positive / half negative SDF samples).

Augmented shapes (from augmented_sdf_data_dir) are always added to the training
set regardless of splits_csv.  Their labels are tracked in `augmented_labels` so
the caller can skip them when saving Stage 2 latent targets.
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


def _load_npz(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    data = np.load(path)
    pos = _remove_nans(torch.from_numpy(np.asarray(data['pos'], dtype=np.float32)))
    neg = _remove_nans(torch.from_numpy(np.asarray(data['neg'], dtype=np.float32)))
    if pos.shape[0] > 0:
        pos = pos[torch.randperm(pos.shape[0])]
    if neg.shape[0] > 0:
        neg = neg[torch.randperm(neg.shape[0])]
    return pos, neg


class SDFSceneDataset(Dataset):
    """
    Args:
        sdf_data_dir: Root with one folder per label.
        splits_csv: CSV with columns label, split.
        split: Single split name, or list of names (e.g. ['train', 'val']). Test excluded by config.
        samples_per_scene: Total random SDF points per shape per __getitem__.
        clamp_value: If set, clamp sdf targets to [-v, v].
        augmented_sdf_data_dir: Optional second root whose sub-folders are augmented
            variants (e.g. '<label>_00' … '<label>_09').  Every sub-folder that
            contains a samples.npz is added to the training set as an additional
            shape with its own latent code.  These labels are recorded in
            ``self.augmented_labels`` so train_deepsdf.py can exclude them when
            writing Stage 2 per-label latent files.
    """

    def __init__(
        self,
        sdf_data_dir: str,
        splits_csv: str,
        split: str | list[str],
        samples_per_scene: int,
        clamp_value: float | None = None,
        augmented_sdf_data_dir: str | None = None,
    ):
        splits_df = pd.read_csv(splits_csv)

        if isinstance(split, list):
            labels = set(splits_df[splits_df['split'].isin(split)]['label'].astype(str))
        else:
            labels = set(splits_df[splits_df['split'] == split]['label'].astype(str))

        self.samples_per_scene = samples_per_scene
        self.clamp_value = clamp_value

        self.labels: list[str] = []
        self.label_to_idx: dict[str, int] = {}
        self._ram_pos_neg: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.augmented_labels: set[str] = set()

        for label in sorted(labels):
            path = resolve_samples_npz(sdf_data_dir, label)
            if path is None:
                continue
            idx = len(self.labels)
            self.label_to_idx[label] = idx
            self.labels.append(label)
            self._ram_pos_neg.append(_load_npz(path))

        if not self.labels:
            raise RuntimeError(
                f"No samples.npz found under '{sdf_data_dir}' for split={split!r}."
            )

        # -- Augmented shapes (always train-only, bypass splits CSV) ----------
        n_aug = 0
        if augmented_sdf_data_dir and os.path.isdir(augmented_sdf_data_dir):
            for aug_label in sorted(os.listdir(augmented_sdf_data_dir)):
                aug_dir = os.path.join(augmented_sdf_data_dir, aug_label)
                npz_path = os.path.join(aug_dir, 'samples.npz')
                if not os.path.isfile(npz_path):
                    continue
                idx = len(self.labels)
                self.label_to_idx[aug_label] = idx
                self.labels.append(aug_label)
                self.augmented_labels.add(aug_label)
                self._ram_pos_neg.append(_load_npz(npz_path))
                n_aug += 1
            if n_aug:
                print(
                    f'SDFSceneDataset: added {n_aug} augmented shapes '
                    f'from {augmented_sdf_data_dir}'
                )
            else:
                print(
                    f'SDFSceneDataset: WARNING — augmented_sdf_data_dir set but no '
                    f'samples.npz found under {augmented_sdf_data_dir}'
                )

        if samples_per_scene < 2:
            raise ValueError('samples_per_scene must be at least 2 (pos/neg halves).')

        n_orig = len(self.labels) - n_aug
        total_floats = sum(
            p.numel() + n.numel() for p, n in self._ram_pos_neg
        )
        approx_gib = total_floats * 4 / (1024 ** 3)

        print(
            f'SDFSceneDataset [{split}]: {n_orig} original + {n_aug} augmented = '
            f'{len(self.labels)} shapes, {samples_per_scene} samples/scene per step'
        )
        print(
            f'SDFSceneDataset: loaded all samples.npz into RAM '
            f'(~{approx_gib:.2f} GiB float32, {total_floats:,} values).'
        )

    def __len__(self) -> int:
        return len(self.labels)

    def _unpack_subsample(self, pos_tensor: torch.Tensor, neg_tensor: torch.Tensor) -> torch.Tensor:
        subsample = self.samples_per_scene
        half = int(subsample / 2)

        if pos_tensor.shape[0] == 0 or neg_tensor.shape[0] == 0:
            raise RuntimeError('pos or neg has no valid samples (empty after init load)')

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
        pos_tensor, neg_tensor = self._ram_pos_neg[idx]
        sdf_data = self._unpack_subsample(pos_tensor, neg_tensor)
        return {
            'sdf_data': sdf_data,
            'latent_idx': idx,
            'label': self.labels[idx],
        }

    @property
    def num_shapes(self) -> int:
        return len(self.labels)
