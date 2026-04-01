"""SDF grid, mesh extraction, and DeepSDF loss helpers."""

import torch
import torch.utils.dlpack
import open3d as o3d
import open3d.core as o3c


# ---------------------------------------------------------------------------
# Volume / mesh extraction
# ---------------------------------------------------------------------------

def get_volume_coords(resolution: int = 64, bbox: float = 0.15) -> torch.Tensor:
    """
    Generate a uniform 3D grid of query points in [-bbox, bbox]^3.

    Args:
        resolution: number of grid steps per axis (resolution^3 total points)
        bbox:       half-extent of the bounding box in metres

    Returns:
        coords: (resolution^3, 3) on CPU
    """
    vals = torch.linspace(-bbox, bbox, resolution)
    grid = torch.meshgrid(vals, vals, vals, indexing='ij')
    coords = torch.stack([g.ravel() for g in grid], dim=1)
    return coords


def sdf2mesh(pred_sdf: torch.Tensor, grid_points: torch.Tensor, t: float = 0.0):
    """
    Extract a watertight mesh from SDF predictions using convex hull.
    Adapted from corepp/utils.py (sdf2mesh_cuda).

    Strategy: keep all grid points where SDF < t (predicted interior), then
    compute the convex hull.  If the result is not watertight, iteratively
    voxel-downsample until it is.

    Args:
        pred_sdf:    (N,) or (N, 1) SDF values on CUDA
        grid_points: (N, 3) corresponding 3D positions on CUDA
        t:           threshold (default 0.0)

    Returns:
        mesh: open3d.geometry.TriangleMesh (watertight, on CPU)

    Raises:
        ValueError  if fewer than 4 interior points are found
        RuntimeError if a watertight mesh cannot be produced
    """
    pred_sdf = pred_sdf.squeeze()
    keep_idx = torch.lt(pred_sdf, t)
    keep_points = grid_points[keep_idx].contiguous()

    if keep_points.shape[0] < 4:
        raise ValueError(
            f"Only {keep_points.shape[0]} interior SDF points — "
            "convex hull requires at least 4. "
            "Try lowering the SDF threshold or increasing grid resolution."
        )

    o3d_t = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(keep_points))
    pcd_gpu = o3d.t.geometry.PointCloud(o3d_t)

    voxel_size = 0.0
    hull_gpu = pcd_gpu.compute_convex_hull()
    mesh = _clean_mesh(hull_gpu.to_legacy())

    while not mesh.is_watertight():
        voxel_size += 0.001
        if voxel_size > 0.05:
            raise RuntimeError(
                "Could not produce a watertight mesh after progressive "
                "voxel downsampling."
            )
        down_pcd = pcd_gpu.voxel_down_sample(voxel_size=voxel_size)
        hull_gpu = down_pcd.compute_convex_hull()
        mesh = _clean_mesh(hull_gpu.to_legacy())

    return mesh


def _clean_mesh(mesh):
    mesh = mesh.subdivide_loop(number_of_iterations=1)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    return mesh


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def sdf_loss(
    pred_sdf: torch.Tensor,
    target_sdf: torch.Tensor,
    latent: torch.Tensor,
    sigma: float = 0.01,
):
    """
    DeepSDF loss: L1 reconstruction + L2 latent regulariser.

    Args:
        pred_sdf:   (N, 1) predicted SDF values
        target_sdf: (N, 1) ground-truth SDF values
        latent:     (N, latent_size) latent codes for each sample in the batch
        sigma:      regularisation weight

    Returns:
        (total_loss, l1_loss, l2_loss)
    """
    l1 = torch.mean(torch.abs(pred_sdf - target_sdf))
    l2 = sigma ** 2 * torch.mean(torch.linalg.norm(latent, dim=1, ord=2))
    return l1 + l2, l1, l2
