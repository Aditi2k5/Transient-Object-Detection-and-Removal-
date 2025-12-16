import torch
import random
from tqdm import tqdm
from .gaussian_model import GaussianModel
from .renderer import render
from .losses import *
from .utils import project_gaussians_to_camera, gaussians_to_mask

def train_one_scene(source_path, model_path, iterations=15000):
    from gaussian_splatting.scene import Scene
    import argparse

    args = argparse.Namespace(
        source_path=source_path,
        model_path=model_path,
        images="images",
        eval=False,
        resolution=1,
        white_background=False,
        data_device="cuda",
        quiet=True
    )

    # 1. Create model first
    gaussians = GaussianModel(sh_degree=3)
    scene = Scene(args, gaussians=gaussians, shuffle=True)

    gaussians.initialize_motion_tracking()
    gaussians.training_setup()  # original optimizer setup

    bg = torch.tensor([1,1,1], device="cuda", dtype=torch.float32)
    pipe = argparse.Namespace(debug=False)

    prev_cam = None

    for itr in tqdm(range(1, iterations + 1)):
        gaussians.update_learning_rate(itr)

        cam = random.choice(scene.getTrainCameras())

        render_pkg = render(cam, gaussians, pipe, bg)
        image = render_pkg["render"]
        gt = cam.original_image.cuda()

        loss = 0.8 * l1_loss(image, image, gt) + 0.2 * ssim_loss(image, gt)

        # Temporal contrastive + motion tracking
        if prev_cam is not None:
            xy_prev = project_gaussians_to_camera(gaussians, prev_cam)
            xy_curr = project_gaussians_to_camera(gaussians, cam)
            motion = (xy_prev - xy_curr).norm(dim=-1)

            gaussians.update_motion_variance_ema(motion, alpha=0.9)

            feat = torch.cat([gaussians.get_xyz,
                              gaussians.get_scaling,
                              gaussians.get_opacity], dim=-1)
            loss += 0.10 * temporal_contrastive_loss(feat, feat)

            loss += 0.01 * distractor_reg_loss(gaussians.motion_variance)

        # Constant-mover opacity suppression
        if itr % 5 == 0:
            score = gaussians.get_distractor_score(threshold=0.30)
            gaussians._opacity = gaussians._opacity - 0.02 * score
            gaussians._opacity = torch.clamp(gaussians._opacity, max=0.0)

        # Save mask every 500 iter
        if itr % 500 == 0:
            mask = gaussians_to_mask(gaussians, cam, gt.shape[1], gt.shape[2])
            torch.save(mask.cpu(), f"{model_path}/masks/mask_{itr:05d}.pt")

        loss.backward()
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)

        if itr % 100 == 0:
            psnr_val = -10 * torch.log10(((image - gt)**2).mean())
            print(f"Iter {itr} PSNR {psnr_val.item():.2f}")

        if itr % 5000 == 0:
            gaussians.save_ply(f"{model_path}/point_cloud/iteration_{itr}/point_cloud.ply")

        prev_cam = cam

    gaussians.save_ply(f"{model_path}/point_cloud/final/point_cloud.ply")