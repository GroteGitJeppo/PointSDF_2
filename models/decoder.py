import torch
import torch.nn as nn


class SDFDecoder(nn.Module):
    """
    DeepSDF-style weight-normed MLP decoder.

    Input:  concat(latent_code, xyz)  — shape (N, latent_size + 3)
    Output: SDF scalar                — shape (N, 1)

    A skip connection re-injects the full input after the first 3 layers,
    following the architecture in Park et al. (CVPR 2019).
    """

    def __init__(
        self,
        latent_size: int = 64,
        num_layers: int = 8,
        inner_dim: int = 256,
        skip_connections: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.skip_connections = skip_connections
        self.latent_size = latent_size

        input_dim = latent_size + 3
        self.input_dim = input_dim

        # 2 layers are outside the sequential stack (skip_layer + final_layer)
        # when skip_connections is active and num_layers >= 8.
        num_extra_layers = 2 if (skip_connections and num_layers >= 8) else 1

        layers = []
        for _ in range(num_layers - num_extra_layers):
            layers.append(
                nn.Sequential(
                    nn.utils.parametrizations.weight_norm(
                        nn.Linear(input_dim, inner_dim)
                    ),
                    nn.ReLU(),
                )
            )
            input_dim = inner_dim

        self.net = nn.Sequential(*layers)
        self.final_layer = nn.Sequential(nn.Linear(inner_dim, 1), nn.Tanh())
        # After the skip, the channel count changes: inner_dim - input_dim channels
        # from skip_layer are concatenated with the original input_dim channels.
        self.skip_layer = nn.Sequential(
            nn.Linear(inner_dim, inner_dim - self.input_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, latent_size + 3) — latent code concatenated with xyz query
        Returns:
            sdf: (N, 1)
        """
        skip_input = x  # preserve for skip re-injection (full gradient flow)

        if self.skip_connections and self.num_layers >= 5:
            h = x
            for i in range(3):
                h = self.net[i](h)
            h = self.skip_layer(h)
            h = torch.cat([h, skip_input], dim=1)
            for i in range(self.num_layers - 5):
                h = self.net[3 + i](h)
            return self.final_layer(h)
        else:
            return self.final_layer(self.net(x))
