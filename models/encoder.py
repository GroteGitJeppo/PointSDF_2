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
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
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

    Three-level hierarchy tuned for potato geometry normalised to a half-extent
    of 0.05 m (coordinates in [-0.05, 0.05]^3 after centering + normalisation):
      SA1  r=0.01 m — small local surface patch
      SA2  r=0.02 m — medium neighbourhood covering most of the potato
      SA3  GlobalSA — full-shape aggregation

    The per-cloud scale ratio (original_half_extent / 0.05) is concatenated to
    the 1024-dim global feature before the latent head so that the encoder can
    recover metric size information lost during normalisation.
    """

    def __init__(self, latent_size: int = 32):
        super().__init__()
        self.latent_size = latent_size

        self.sa1_module = SAModule(0.5, 0.01, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.02, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.latent_head = nn.Sequential(
            nn.Linear(1025, 512),   # 1024 global features + 1 scale scalar
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, latent_size),
        )

    def forward(self, data):
        """
        Args:
            data: PyG Data/Batch with:
                data.pos   (N_total, 3)  — normalised, centred point coordinates
                data.batch (N_total,)    — batch index per point
                data.scale (B, 1) or (B,) — per-cloud scale ratio
        Returns:
            latent: (B, latent_size)
        """
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, _, _ = sa3_out                          # (B, 1024)
        scale = data.scale.view(-1, 1)             # (B, 1)
        x = torch.cat([x, scale], dim=1)           # (B, 1025)
        return self.latent_head(x)
