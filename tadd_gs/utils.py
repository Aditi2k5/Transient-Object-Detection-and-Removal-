import torch
import torch.nn.functional as F

def project_gaussians_to_camera(gaussians, camera):
    xyz = gaussians.get_xyz
    xyz_h = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=-1)
    view_h = xyz_h @ camera.world_view_transform.T
    proj_h = view_h @ camera.full_proj_transform.T
    proj_h = proj_h[:, :3] / (proj_h[:, 3:4] + 1e-8)
    return proj_h[:, :2]

def gaussians_to_mask(gaussians, camera, H, W):
    xy = project_gaussians_to_camera(gaussians, camera)
    xyn = torch.clamp(xy / torch.tensor([W-1, H-1], device=xy.device), -1, 1)
    xyn = xyn * torch.tensor([-1, 1], device=xy.device)
    score = gaussians.get_distractor_score().unsqueeze(-1)
    grid = xyn.unsqueeze(0).unsqueeze(0)
    mask = F.grid_sample(score.view(1,1,-1,1), grid, mode='bilinear', align_corners=True)
    return (mask.squeeze() > 0.5).float()
    