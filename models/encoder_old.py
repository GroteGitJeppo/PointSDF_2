import torch
import torch.nn as nn
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class PointNetEncoder(torch.nn.Module):
    """
    PointNet++ encoder: partial point cloud → latent code of size `latent_size`.
    Reduced-capacity architecture to prevent overfitting on simple convex geometries.
    """

    def __init__(self, latent_size: int = 32):
        super().__init__()
        self.latent_size = latent_size
        self.sa1_module = SAModule(ratio=0.25, r=0.2, nn=MLP([3, 32, 32, 64], norm="batch_norm"))
        self.sa2_module = GlobalSAModule(MLP([64 + 3, 64, 128, 256], norm="batch_norm"))
        self.latent_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(128, latent_size),
        )

    def forward(self, data):
        """
        Args:
            data: PyG Data with data.pos (N_total, 3) and data.batch (N_total,)
        Returns:
            latent: (B, latent_size)
        """
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        x, _, _ = sa2_out
        return self.latent_head(x)
