"""SDF grid, mesh extraction, and DeepSDF loss helpers."""

import torch
import torch.nn.functional as F
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
    Convex-hull mesh extraction used in this repo for volume estimation.

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


def sdf_autodecoder_loss_chunk(
    pred_sdf: torch.Tensor,
    target_sdf: torch.Tensor,
    latent_vecs: torch.Tensor,
    num_sdf_samples_scene: int,
    epoch_1based: int,
    code_reg_lambda: float,
    reg_ramp_epochs: int,
    do_code_regularization: bool,
    do_code_regularization_sphere: bool,
):
    """
    Loss for one gradient chunk in Stage 1: L1 sum over the chunk divided by
    total samples in the scene, plus optional ramped latent regularisers
    (same denominator), as used in train_deepsdf.py.
    """
    loss_l1 = F.l1_loss(pred_sdf, target_sdf, reduction='sum') / num_sdf_samples_scene
    chunk_loss = loss_l1
    reg_l2 = torch.zeros((), device=pred_sdf.device, dtype=pred_sdf.dtype)
    reg_sphere = torch.zeros((), device=pred_sdf.device, dtype=pred_sdf.dtype)
    ramp = min(1.0, float(epoch_1based) / float(reg_ramp_epochs))

    if do_code_regularization:
        l2_size_loss = torch.sum(torch.norm(latent_vecs, dim=1))
        reg = code_reg_lambda * ramp * l2_size_loss / num_sdf_samples_scene
        chunk_loss = chunk_loss + reg
        reg_l2 = reg.detach()

    if do_code_regularization_sphere:
        sphere_loss = torch.abs(1.0 - torch.norm(latent_vecs, dim=1)).sum()
        reg = code_reg_lambda * ramp * sphere_loss / num_sdf_samples_scene
        chunk_loss = chunk_loss + reg
        reg_sphere = reg.detach()

    return chunk_loss, loss_l1.detach(), reg_l2, reg_sphere


# ---------------------------------------------------------------------------
# Chamfer distance
# ---------------------------------------------------------------------------

def chamfer_distance(pred_pts: torch.Tensor, gt_pts: torch.Tensor) -> float:
    """
    Symmetric Chamfer distance (L1) between two point sets.

    CD(A, B) = mean_a( min_b ||a - b|| ) + mean_b( min_a ||a - b|| )

    Both tensors must be on the same device.

    Args:
        pred_pts: (M, 3) predicted surface point cloud
        gt_pts:   (N, 3) ground-truth surface point cloud
    Returns:
        Scalar Chamfer distance (same units as the point coordinates).
    """
    dists = torch.cdist(pred_pts, gt_pts)            # (M, N)
    cd = dists.min(dim=1).values.mean() + dists.min(dim=0).values.mean()
    return cd.item()
