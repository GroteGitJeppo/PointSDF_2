"""Coarse-to-fine SDF grid decoding for faster decoder inference."""

from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.nn.functional as F

from .sdf_helpers import get_volume_coords

logger = logging.getLogger(__name__)

_DEFAULT_CHUNK = 131072


def _decode_in_chunks(
    latent: torch.Tensor,
    decoder: torch.nn.Module,
    coords: torch.Tensor,
    chunk: int,
    clamp_dist: float | None,
) -> torch.Tensor:
    """Decode SDF at coords; latent is (1, L), coords (N, 3)."""
    outs = []
    n = coords.shape[0]
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        lat_t = latent.expand(e - s, -1)
        inp = torch.cat([lat_t, coords[s:e]], dim=1)
        pred = decoder(inp)
        if clamp_dist is not None:
            pred = pred.clamp(-clamp_dist, clamp_dist)
        outs.append(pred)
    return torch.cat(outs, dim=0)


def _cell_surface_mask_vectorized(sdf_vol: torch.Tensor) -> torch.Tensor:
    """Vectorised eight-corner sign straddle test."""
    rc = sdf_vol.shape[0]
    if rc < 2:
        return torch.zeros((0, 0, 0), dtype=torch.bool, device=sdf_vol.device)
    # Stack 8 corner tensors shaped (Rc-1, Rc-1, Rc-1)
    corners = []
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                corners.append(
                    sdf_vol[di : rc - 1 + di, dj : rc - 1 + dj, dk : rc - 1 + dk]
                )
    stacked = torch.stack(corners, dim=0)  # (8, Rc-1, Rc-1, Rc-1)
    lo = stacked.min(dim=0).values
    hi = stacked.max(dim=0).values
    return (lo < 0) & (hi > 0)


def _dilate_mask(mask: torch.Tensor, iterations: int) -> torch.Tensor:
    """Binary 3D dilation (max over 3^3 neighbourhood)."""
    if iterations <= 0:
        return mask
    x = mask.unsqueeze(0).unsqueeze(0).float()
    for _ in range(iterations):
        x = F.max_pool3d(x, kernel_size=3, stride=1, padding=1)
    return x.squeeze(0).squeeze(0) > 0.5


def decode_sdf_hierarchical(
    latent: torch.Tensor,
    decoder: torch.nn.Module,
    bbox: float,
    R_coarse: int,
    subdiv: int,
    surface_dilation: int,
    device: torch.device,
    clamp_dist: float | None = None,
    max_fine_queries: int | None = None,
    decode_chunk: int = _DEFAULT_CHUNK,
    warn_fn: Callable[[str], None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Two-pass SDF decode: coarse grid, surface cells + dilation, then fine vertices
    only inside active coarse cells (embedded lattice, no duplicate xyz).

    Args:
        latent: (1, latent_size) on device
        decoder: frozen SDF MLP
        bbox: half-extent of the axis-aligned cube (metres)
        R_coarse: coarse grid resolution (vertices per axis), >= 2
        subdiv: integer >= 1; fine vertices per axis:
            R_fine = (R_coarse - 1) * subdiv + 1
        surface_dilation: binary 3D dilation iterations on the cell mask
        device: torch device for tensors
        clamp_dist: if set, clamp decoder outputs to [-clamp, clamp]
        max_fine_queries: if set, raise RuntimeError when fine point count exceeds this
        decode_chunk: max rows per decoder forward
        warn_fn: optional callback for fallback messages (e.g. test print)

    Returns:
        (grid_coords, pred_sdf) both on device, shapes (N, 3) and (N, 1)
    """
    if R_coarse < 2:
        raise ValueError(f"R_coarse must be >= 2, got {R_coarse}")
    if subdiv < 1:
        raise ValueError(f"subdiv must be >= 1, got {subdiv}")

    def _warn(msg: str) -> None:
        if warn_fn is not None:
            warn_fn(msg)
        else:
            logger.warning(msg)

    coarse_xyz = get_volume_coords(resolution=R_coarse, bbox=bbox).to(device)
    pred_coarse = _decode_in_chunks(
        latent, decoder, coarse_xyz, decode_chunk, clamp_dist
    )
    sdf_vol = pred_coarse.view(R_coarse, R_coarse, R_coarse)

    cell_mask = _cell_surface_mask_vectorized(sdf_vol)
    dilated = _dilate_mask(cell_mask, surface_dilation)

    if not dilated.any():
        _warn(
            "hierarchical_decode: no surface cells on coarse grid; "
            "falling back to coarse SDF samples for meshing."
        )
        return coarse_xyz, pred_coarse

    R_fine = (R_coarse - 1) * subdiv + 1
    fine_mask = torch.zeros(
        (R_fine, R_fine, R_fine), dtype=torch.bool, device=device
    )
    act = dilated.nonzero(as_tuple=False)
    if act.numel() > 0:
        dv = torch.arange(subdiv + 1, device=device, dtype=torch.long)
        # Per active cell (c0,c1,c2), mark fine vertices [c*subdiv, (c+1)*subdiv] on each axis.
        # Use (N,1,1,1) so (N,1) does not left-pad to (1,1,N,1) and break j_idx broadcast.
        ci = act[:, 0].view(-1, 1, 1, 1) * subdiv
        cj = act[:, 1].view(-1, 1, 1, 1) * subdiv
        ck = act[:, 2].view(-1, 1, 1, 1) * subdiv
        i_idx = ci + dv.view(1, -1, 1, 1)
        j_idx = cj + dv.view(1, 1, -1, 1)
        k_idx = ck + dv.view(1, 1, 1, -1)
        i_idx = i_idx.expand(-1, subdiv + 1, subdiv + 1, subdiv + 1).reshape(-1)
        j_idx = j_idx.expand(-1, subdiv + 1, subdiv + 1, subdiv + 1).reshape(-1)
        k_idx = k_idx.expand(-1, subdiv + 1, subdiv + 1, subdiv + 1).reshape(-1)
        fine_mask[i_idx, j_idx, k_idx] = True

    idx = fine_mask.nonzero(as_tuple=False)
    n_fine = idx.shape[0]
    if n_fine == 0:
        _warn(
            "hierarchical_decode: empty fine mask; falling back to coarse SDF samples."
        )
        return coarse_xyz, pred_coarse

    if max_fine_queries is not None and n_fine > max_fine_queries:
        raise RuntimeError(
            f"hierarchical_decode: fine query count {n_fine} exceeds "
            f"max_fine_queries={max_fine_queries}. Increase max_fine_queries, "
            f"coarsen R_coarse/subdiv, or reduce surface_dilation."
        )

    vals = torch.linspace(-bbox, bbox, R_fine, device=device)
    ii, jj, kk = idx[:, 0], idx[:, 1], idx[:, 2]
    fine_xyz = torch.stack([vals[ii], vals[jj], vals[kk]], dim=1)
    pred_fine = _decode_in_chunks(
        latent, decoder, fine_xyz, decode_chunk, clamp_dist
    )
    return fine_xyz, pred_fine
