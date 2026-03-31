import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SDFSamplesDataset(Dataset):
    """
    Dataset of SDF samples for Stage 1 (DeepSDF autodecoder training).

    Expected directory layout on the server:
        sdf_data_dir/
          <label>/
            samples.npz   # 'pos': (N, 4) — [x, y, z, sdf>0]
                          # 'neg': (M, 4) — [x, y, z, sdf<0]

    Each potato is assigned a contiguous integer index used to address the
    learnable latent code table in train_deepsdf.py.

    Each __getitem__ returns:
        xyz_latent: (4,) float tensor — [latent_idx, x, y, z]
        sdf:        (1,) float tensor — signed distance value
    """

    def __init__(self, sdf_data_dir: str, splits_csv: str, split='train', clamp_value=None):
        """
        Args:
            sdf_data_dir:  Root directory containing one sub-folder per potato label.
            splits_csv:    Path to splits.csv (columns: label, split).
            split:         One of 'train', 'val', 'test', 'all', or a list of split names.
            clamp_value:   If set, SDF values are clamped to [-clamp_value, +clamp_value].
        """
        splits_df = pd.read_csv(splits_csv)

        if split == 'all':
            labels = set(splits_df['label'].astype(str))
        elif isinstance(split, list):
            labels = set(splits_df[splits_df['split'].isin(split)]['label'].astype(str))
        else:
            labels = set(splits_df[splits_df['split'] == split]['label'].astype(str))

        self.label_to_idx: dict[str, int] = {}
        all_xyz_latent = []
        all_sdf = []

        for label in sorted(labels):
            npz_path = os.path.join(sdf_data_dir, label, 'samples.npz')
            if not os.path.exists(npz_path):
                continue

            data = np.load(npz_path)
            pos_samples = data['pos']  # (N, 4)
            neg_samples = data['neg']  # (M, 4)
            samples = np.concatenate([pos_samples, neg_samples], axis=0)  # (N+M, 4)

            idx = len(self.label_to_idx)
            self.label_to_idx[label] = idx

            latent_col = np.full((len(samples), 1), idx, dtype=np.float32)
            all_xyz_latent.append(np.concatenate([latent_col, samples[:, :3]], axis=1))
            all_sdf.append(samples[:, 3:4])

        if not all_xyz_latent:
            raise RuntimeError(
                f"No samples.npz files found in '{sdf_data_dir}' for split='{split}'. "
                "Run the SDF extraction script first."
            )

        xyz_latent = np.concatenate(all_xyz_latent, axis=0).astype(np.float32)
        sdf = np.concatenate(all_sdf, axis=0).astype(np.float32)

        self.xyz_latent = torch.from_numpy(xyz_latent)
        self.sdf = torch.from_numpy(sdf)

        if clamp_value is not None:
            self.sdf = torch.clamp(self.sdf, -clamp_value, clamp_value)

        self.num_shapes = len(self.label_to_idx)
        print(
            f"SDFSamplesDataset [{split}]: "
            f"{self.num_shapes} shapes, {len(self):,} SDF samples"
        )

    def __len__(self) -> int:
        return self.sdf.shape[0]

    def __getitem__(self, idx):
        return self.xyz_latent[idx], self.sdf[idx]
