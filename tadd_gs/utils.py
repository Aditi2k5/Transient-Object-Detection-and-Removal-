"""
TADD-GS Complete Utilities
Combines all TADD-GS functionality
"""

import torch
import torch.nn.functional as F

def project_gaussians_to_camera(gaussians, viewpoint):
    """
    Project 3D Gaussians to 2D camera space
    
    Args:
        gaussians: GaussianModel
        viewpoint: Camera viewpoint
    
    Returns:
        xy: 2D projected positions [N, 2]
    """
    xyz = gaussians.get_xyz  # [N, 3]
    
    # Get camera matrices
    world_view_transform = viewpoint.world_view_transform.transpose(0, 1)
    projection_matrix = viewpoint.projection_matrix.transpose(0, 1)
    full_proj_transform = (world_view_transform @ projection_matrix).transpose(0, 1)
    
    # Project to clip space
    xyz_h = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=-1)  # [N, 4]
    xyz_clip = xyz_h @ full_proj_transform.T  # [N, 4]
    
    # Perspective divide
    xyz_ndc = xyz_clip[:, :3] / (xyz_clip[:, 3:4] + 1e-7)  # [N, 3]
    
    # Take only x, y
    xy = xyz_ndc[:, :2]  # [N, 2]
    
    return xy


def gaussians_to_mask(gaussians, viewpoint, H, W, threshold=0.5):
    """
    Generate distractor mask for visualization
    
    Args:
        gaussians: GaussianModel
        viewpoint: Camera viewpoint
        H, W: Image dimensions
        threshold: Distractor score threshold
    
    Returns:
        mask: [H, W] binary mask (1 = distractor, 0 = static)
    """
    if not hasattr(gaussians, 'motion_variance') or gaussians.motion_variance is None:
        return torch.zeros(H, W, device='cuda')
    
    # Get distractor scores
    scores = gaussians.get_distractor_score(threshold=threshold)
    
    # Project to 2D
    xy = project_gaussians_to_camera(gaussians, viewpoint)
    
    # Convert NDC [-1, 1] to pixel coordinates
    xy_pixel = (xy + 1.0) / 2.0 * torch.tensor([W, H], device=xy.device)
    xy_pixel = xy_pixel.long()
    
    # Clamp to image bounds
    xy_pixel[:, 0] = torch.clamp(xy_pixel[:, 0], 0, W - 1)
    xy_pixel[:, 1] = torch.clamp(xy_pixel[:, 1], 0, H - 1)
    
    # Create mask
    mask = torch.zeros(H, W, device='cuda')
    
    # Rasterize distractor scores
    for i in range(xy_pixel.shape[0]):
        if scores[i] > 0.5:  # High confidence distractor
            x, y = xy_pixel[i]
            mask[y, x] = scores[i].item()
    
    return mask


def compute_camera_normalized_motion(xy_prev, xy_curr):
    """
    Compute motion normalized by camera movement
    
    Args:
        xy_prev: 2D positions at t-1 [N, 2]
        xy_curr: 2D positions at t [N, 2]
    
    Returns:
        outlier_motion: Camera-normalized motion [N]
    """
    # Raw motion
    motion = (xy_prev - xy_curr).norm(dim=-1)
    
    # Normalize by median to filter out camera motion
    median_motion = motion.median()
    normalized_motion = motion / (median_motion + 1e-6)
    
    # Only keep outlier motion (above median)
    outlier_motion = torch.clamp(normalized_motion - 1.0, min=0.0)
    
    return outlier_motion


__all__ = [
    'project_gaussians_to_camera',
    'gaussians_to_mask',
    'compute_camera_normalized_motion'
]