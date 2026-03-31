import torch
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
    No height embedding — pure geometry only.

    Architecture mirrors the original PointRAFT backbone but replaces the
    regression head (scalar output) with a latent projection head.
    """

    def __init__(self, latent_size: int = 64):
        super().__init__()
        self.latent_size = latent_size
        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))
        self.latent_head = MLP([1024, 512, 256, latent_size], dropout=0.5, norm=None)

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
        sa3_out = self.sa3_module(*sa2_out)
        x, _, _ = sa3_out
        return self.latent_head(x)
