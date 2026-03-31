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


class PointCloudLatentDataset(Dataset):
    """
    On-the-fly dataset for Stage 2 (encoder training).

    Loads partial point clouds (.ply) and their corresponding Stage 1 latent
    codes (.pth) on demand.  Latent codes must be saved by train_deepsdf.py
    as individual tensors: <latent_dir>/<label>.pth

    Each __getitem__ returns a PyG Data object with:
        data.pos    — (num_points, 3) centred and FPS-sampled point cloud
        data.latent — (1, latent_size) ground-truth latent code for this potato
        data.batch  — not set here; populated by PyG DataLoader at collation time
    """

    def __init__(
        self,
        data_root: str,
        splits_csv: str,
        latent_dir: str,
        split: str = 'train',
        num_points: int = 1024,
        apply_augmentation: bool = True,
    ):
        self.latent_dir = latent_dir
        self.num_points = num_points
        self.apply_augmentation = apply_augmentation
        self.pre_transform = T.Center()

        if apply_augmentation:
            self.transform = T.Compose([
                T.RandomJitter(0.0005),
                T.RandomRotate(2, axis=0),
                T.RandomRotate(2, axis=1),
                T.RandomRotate(2, axis=2),
                T.RandomFlip(axis=0, p=0.5),
                T.RandomFlip(axis=1, p=0.5),
                T.RandomShear(0.2),
            ])
        else:
            self.transform = None

        splits_df = pd.read_csv(splits_csv)
        labels = set(splits_df[splits_df['split'] == split]['label'].astype(str))

        self.samples: list[tuple[str, str, str]] = []
        for ply_file in Path(data_root).rglob('*.ply'):
            label = ply_file.parent.name
            if label not in labels:
                continue
            latent_path = os.path.join(latent_dir, f'{label}.pth')
            if not os.path.exists(latent_path):
                continue
            self.samples.append((str(ply_file), label, latent_path))

        if not self.samples:
            raise RuntimeError(
                f"No (ply, latent) pairs found for split='{split}'. "
                f"Check that '{latent_dir}' contains <label>.pth files "
                f"produced by train_deepsdf.py."
            )

        print(f"PointCloudLatentDataset [{split}]: {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Data:
        ply_path, _label, latent_path = self.samples[idx]

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

        latent = torch.load(latent_path, weights_only=True)  # (latent_size,)
        # Shape (1, latent_size) so PyG's Batch concatenates correctly to (B, latent_size)
        data.latent = latent.unsqueeze(0)

        return data
