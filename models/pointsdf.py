import torch
import torch.nn as nn

from .encoder import PointNetEncoder
from .decoder import SDFDecoder


class PointSDF(nn.Module):
    """
    Full PointSDF encoder-decoder model.

    Stage 2 forward pass:
        partial point cloud  →  PointNetEncoder  →  latent z
        (latent z, query xyz) →  SDFDecoder       →  SDF scalar

    The decoder is kept as a module attribute so checkpoints save both
    encoder and decoder weights together for inference.
    """

    def __init__(
        self,
        latent_size: int = 64,
        num_layers: int = 8,
        inner_dim: int = 256,
        skip_connections: bool = True,
    ):
        super().__init__()
        self.latent_size = latent_size
        self.encoder = PointNetEncoder(latent_size=latent_size)
        self.decoder = SDFDecoder(
            latent_size=latent_size,
            num_layers=num_layers,
            inner_dim=inner_dim,
            skip_connections=skip_connections,
        )

    def encode(self, data) -> torch.Tensor:
        """Returns (B, latent_size) latent codes from a PyG Data batch."""
        return self.encoder(data)

    def decode(self, latent: torch.Tensor, query_xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent:    (B, latent_size) — one code per sample in the batch
            query_xyz: (M, 3)           — SDF query points (B=1 assumed or pre-tiled)
        Returns:
            sdf: (M, 1)
        """
        latent_tiled = latent.expand(query_xyz.size(0), -1)
        return self.decoder(torch.cat([latent_tiled, query_xyz], dim=1))

    def forward(self, data, query_xyz: torch.Tensor):
        """
        Args:
            data:      PyG Data batch (B partial point clouds)
            query_xyz: (M, 3) SDF query points
        Returns:
            sdf:    (M, 1)
            latent: (B, latent_size)
        """
        latent = self.encode(data)
        sdf = self.decode(latent, query_xyz)
        return sdf, latent
