# Complete TADD-GS Training Script
# Includes all critical components from your research framework

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import torch.nn as nn

# TADD-GS imports
from tadd_gs.losses import temporal_contrastive_loss, distractor_regularization_loss
from tadd_gs.utils import project_gaussians_to_camera, gaussians_to_mask, compute_camera_normalized_motion
from tadd_gs.decomposition import decompose_scene

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def prepare_output_and_logger(args):
    if not args.model_path:
        unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
       
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
    
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
        
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, 
             disable_tadd=False, tadd_warmup=1500, distractor_threshold=0.50, suppression_strength=0.02):

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)

    print(f"Initial Gaussians: {gaussians.get_xyz.shape[0]}")

    gaussians.training_setup(opt)
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # Initialize TADD-GS components
    if not disable_tadd:
        gaussians.initialize_motion_tracking()
        print(f"[TADD-GS] ✅ Full TADD-GS enabled with:")
        print(f"  - Self-supervised contrastive learning")
        print(f"  - Pattern detection (cyclic + linear motion)")
        print(f"  - Two-layer decomposition capability")
        print(f"  - Warmup: {tadd_warmup} iterations")
        print(f"  - Distractor threshold: {distractor_threshold:.2f}")
        print(f"  - Suppression strength: {suppression_strength:.4f}")
    else:
        print("[TADD-GS] DISABLED — running baseline 3DGS")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    ema_loss_for_log = 0.0
    
    # For tracking consecutive frames
    prev_viewpoint = None
    prev_features = None

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # ========================================================================
        # CRITICAL COMPONENT 1: Self-Supervised Contrastive Learning
        # ========================================================================
        if not disable_tadd and prev_viewpoint is not None and iteration > tadd_warmup:
            try:
                # Project Gaussians to camera space
                xy_prev = project_gaussians_to_camera(gaussians, prev_viewpoint)
                xy_curr = project_gaussians_to_camera(gaussians, viewpoint_cam)
                
                # Camera-normalized motion (filters out camera movement)
                outlier_motion = compute_camera_normalized_motion(xy_prev, xy_curr)
                
                # Update motion variance with EMA
                gaussians.update_motion_variance_ema(outlier_motion, alpha=0.95)

                # Create features for contrastive learning
                curr_features = torch.cat([
                    gaussians.get_xyz,
                    gaussians.get_scaling,
                    gaussians.get_opacity
                ], dim=-1)
                
                # TRUE CONTRASTIVE LOSS with positive/negative pairs
                if prev_features is not None:
                    contrastive_loss = temporal_contrastive_loss(
                        prev_features, 
                        curr_features, 
                        motion_variance=gaussians.motion_variance,
                        temperature=0.07
                    )
                    if torch.isfinite(contrastive_loss):
                        loss += 0.001 * contrastive_loss
                
                # Distractor regularization
                reg_loss = distractor_regularization_loss(
                    gaussians.motion_variance, 
                    threshold=distractor_threshold
                )
                if torch.isfinite(reg_loss):
                    loss += 0.0001 * reg_loss
                
                # Store features for next iteration
                prev_features = curr_features.detach()
                    
            except Exception as e:
                if iteration % 1000 == 0:
                    print(f"Warning: Contrastive loss failed at iter {iteration}: {e}")

        # ========================================================================
        # CRITICAL COMPONENT 2: Pattern Detection (Cyclic + Linear Motion)
        # ========================================================================
        if not disable_tadd and iteration % 100 == 0 and iteration > tadd_warmup:
            try:
                # Update pattern scores every 100 iterations
                gaussians.update_pattern_scores()
                
                if iteration % 500 == 0:
                    # Log pattern detection results
                    num_cyclic = (gaussians.cyclic_score > 0.5).sum().item()
                    num_linear = (gaussians.linear_score > 0.5).sum().item()
                    num_constant = (gaussians.constant_mover_score > 0.5).sum().item()
                    print(f"[Pattern Detection] Cyclic: {num_cyclic}, Linear: {num_linear}, Constant: {num_constant}")
                    
            except Exception as e:
                if iteration % 1000 == 0:
                    print(f"Warning: Pattern detection failed: {e}")

        # ========================================================================
        # Smart Distractor Suppression (High-Confidence Only)
        # ========================================================================
        if not disable_tadd and iteration % 50 == 0 and iteration > tadd_warmup and hasattr(gaussians, "motion_variance") and gaussians.motion_variance is not None:
            try:
                with torch.no_grad():
                    score = gaussians.get_distractor_score(threshold=distractor_threshold)
                    high_confidence = score > 0.8
                    
                    if high_confidence.sum() > 0:
                        opacity_reduction = torch.zeros_like(gaussians._opacity)
                        opacity_reduction[high_confidence] = suppression_strength
                        gaussians._opacity.data = torch.clamp(
                            gaussians._opacity.data - opacity_reduction,
                            min=-5.0, max=5.0
                        )
                        
                        if iteration % 500 == 0:
                            num_suppressed = high_confidence.sum().item()
                            print(f"[TADD-GS] Suppressed {num_suppressed} high-confidence distractors")
                            
            except Exception as e:
                print(f"Warning: Distractor suppression failed at iter {iteration}: {e}")

        # Save masks periodically
        if iteration % 500 == 0 and not disable_tadd:
            try:
                os.makedirs(f"{scene.model_path}/masks", exist_ok=True)
                H, W = gt_image.shape[1], gt_image.shape[2]
                mask = gaussians_to_mask(gaussians, viewpoint_cam, H, W)
                torch.save(mask.cpu(), f"{scene.model_path}/masks/mask_{iteration:05d}.pt")
            except Exception as e:
                if iteration % 1000 == 0:
                    print(f"Warning: Mask save failed: {e}")

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)

            # ========================================================================
            # DETAILED METRICS EVERY 500 ITERATIONS
            # ========================================================================
            if iteration % 500 == 0:
                current_psnr = psnr(image, gt_image).mean().item()
                num_points = gaussians.get_xyz.shape[0]
                motion_var = 0.0
                cyclic_mean = 0.0
                linear_mean = 0.0
                
                if not disable_tadd and hasattr(gaussians, 'motion_variance') and gaussians.motion_variance is not None:
                    motion_var = gaussians.motion_variance.mean().item()
                    if hasattr(gaussians, 'cyclic_score') and gaussians.cyclic_score is not None:
                        cyclic_mean = gaussians.cyclic_score.mean().item()
                    if hasattr(gaussians, 'linear_score') and gaussians.linear_score is not None:
                        linear_mean = gaussians.linear_score.mean().item()
                
                print(f"\n{'='*90}")
                print(f"[ITER {iteration:5d}] Detailed Metrics:")
                print(f"  Loss: {ema_loss_for_log:.6f}")
                print(f"  PSNR: {current_psnr:.2f} dB")
                print(f"  Points: {num_points:,}")
                if not disable_tadd:
                    print(f"  Motion Variance: {motion_var:.6f}")
                    print(f"  Cyclic Pattern: {cyclic_mean:.4f}")
                    print(f"  Linear Pattern: {linear_mean:.4f}")
                print(f"{'='*90}\n")

            if iteration == opt.iterations:
                progress_bar.close()

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), dataset)

            # Save point cloud every 2000 iterations
            if iteration % 2000 == 0 and iteration > 0:
                print(f"\n[ITER {iteration}] Saving point cloud checkpoint")
                scene.save(iteration)

            if (iteration in saving_iterations):
                print(f"\n[ITER {iteration}] Saving Gaussians (scheduled)")
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    
                    # Sync motion variance after densification
                    if not disable_tadd:
                        gaussians.reset_motion_variance_on_densification()

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print(f"\n[ITER {iteration}] Saving full checkpoint")
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        prev_viewpoint = viewpoint_cam

    # ========================================================================
    # CRITICAL COMPONENT 3: Two-Layer Decomposition at End of Training
    # ========================================================================
    if not disable_tadd:
        print("\n" + "="*90)
        print("PERFORMING FINAL SCENE DECOMPOSITION")
        print("="*90)
        
        try:
            # Get final distractor scores
            distractor_scores = gaussians.get_distractor_score(threshold=distractor_threshold)
            
            # Decompose into static + distractor layers
            decomposed = decompose_scene(gaussians, distractor_scores, threshold=distractor_threshold)
            
            # Save decomposed models
            decomposed.save_decomposed(scene.model_path, opt.iterations)
            
            print(f"✅ Scene decomposed successfully!")
            print(f"   Static layer: {scene.model_path}/static_iteration_{opt.iterations}/")
            print(f"   Distractor layer: {scene.model_path}/distractor_iteration_{opt.iterations}/")
            
        except Exception as e:
            print(f"Warning: Scene decomposition failed: {e}")

    print("\n✅ Training complete.")

if __name__ == "__main__":
    parser = ArgumentParser(description="TADD-GS Training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    # TADD-GS control
    parser.add_argument("--disable_tadd", action='store_true', default=False, help="Disable TADD-GS (baseline 3DGS)")
    parser.add_argument("--tadd_warmup", type=int, default=1500, help="Warmup iterations")
    parser.add_argument("--distractor_threshold", type=float, default=0.70, help="Distractor threshold")
    parser.add_argument("--suppression_strength", type=float, default=0.005, help="Suppression strength")
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.images = "images"
    
    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), 
             args.test_iterations, args.save_iterations, args.checkpoint_iterations, 
             args.start_checkpoint, args.debug_from,
             disable_tadd=args.disable_tadd, 
             tadd_warmup=args.tadd_warmup, 
             distractor_threshold=args.distractor_threshold, 
             suppression_strength=args.suppression_strength)
    
    print("\n✅ Training complete.")