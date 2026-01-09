# Complete TADD-GS Training Script
# WITH: Simplified progress, Early stopping, Explosion detection

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
             disable_tadd=False, tadd_warmup=1500, distractor_threshold=0.85, suppression_strength=0.001,
             early_stop_patience=20, early_stop_psnr_threshold=12.0, max_points=200000):

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)

    print(f"Initial Gaussians: {gaussians.get_xyz.shape[0]}")

    with torch.no_grad():
    # Check scales
        scales = gaussians.get_scaling()
        if scales.max() > 10.0 or scales.min() < 0.0001:
            print(f"⚠️  Invalid scales detected: {scales.min():.6f} - {scales.max():.6f}")
            print(f"   Clamping to safe range [0.001, 1.0]")
            gaussians._scaling.data = torch.clamp(gaussians._scaling.data, min=-6.9, max=0.0)
            scales = gaussians.get_scaling()
            print(f"   Fixed scales: {scales.min():.6f} - {scales.max():.6f}")
        
        # Check rotations
        rots = gaussians.get_rotation()
        norms = rots.norm(dim=-1)
        if (norms - 1.0).abs().max() > 0.1:
            print(f"⚠️  Invalid rotations detected")
            print(f"   Normalizing quaternions")
            gaussians._rotation.data = rots / (norms.unsqueeze(-1) + 1e-8)
        
        # Check for NaN/Inf
        for name, param in [('xyz', gaussians._xyz), ('scaling', gaussians._scaling), 
                            ('rotation', gaussians._rotation), ('opacity', gaussians._opacity)]:
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"⚠️  NaN/Inf in {name} - fixing")
                param.data = torch.nan_to_num(param, nan=0.0, posinf=1.0, neginf=-1.0)
        
        print(f"✅ Gaussian validation complete")
    gaussians.training_setup(opt)
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # Initialize TADD-GS components
    if not disable_tadd:
        gaussians.initialize_motion_tracking()
        gaussians.initialize_semantic_tracking()
        print(f"\n{'='*70}")
        print(f"[TADD-GS] ENABLED")
        print(f"{'='*70}")
        print(f"  Warmup: {tadd_warmup} iterations")
        print(f"  Distractor threshold: {distractor_threshold:.2f}")
        print(f"  Suppression strength: {suppression_strength:.4f}")
        print(f"  Early stop patience: {early_stop_patience} checks ({early_stop_patience * 500} iters)")
        print(f"  Min PSNR threshold: {early_stop_psnr_threshold} dB")
        print(f"  Max points: {max_points:,}")
        print(f"{'='*70}\n")
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
    
    # Early stopping tracking
    best_psnr = 0.0
    psnr_no_improve_count = 0
    psnr_history = []
    loss_history = []
    point_history = []

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
        # CRITICAL DIAGNOSTIC: Check state right at warmup point
        # ========================================================================
        if not disable_tadd and iteration == tadd_warmup + 1:
            print("\n" + "🔍"*35)
            print("WARMUP JUST ENDED - DIAGNOSING STATE")
            print("🔍"*35)

            try:
                gaussians.diagnose_hybrid_detection()
            except Exception as e:
                print(f"Could not show diagnostics: {e}")
            
            print(f"\n{'🔍'*35}\n")
            
            if gaussians.motion_variance is not None:
                mv = gaussians.motion_variance
                print(f"\n📊 Motion Variance Distribution:")
                print(f"  Min:    {mv.min().item():.6f}")
                print(f"  Max:    {mv.max().item():.6f}")
                print(f"  Mean:   {mv.mean().item():.6f}")
                print(f"  Median: {mv.median().item():.6f}")
                
                total = mv.shape[0]
                low = (mv < 0.1).sum().item()
                med = ((mv >= 0.1) & (mv < 0.5)).sum().item()
                high = (mv >= 0.5).sum().item()
                
                print(f"\n📊 Distribution:")
                print(f"  < 0.1:   {low:,} ({100*low/total:.1f}%) - Likely static")
                print(f"  0.1-0.5: {med:,} ({100*med/total:.1f}%) - Moderate motion")
                print(f"  > 0.5:   {high:,} ({100*high/total:.1f}%) - High motion")
                
                # Show what will be suppressed
                scores = gaussians.get_distractor_score(threshold=distractor_threshold)
                will_suppress = (scores > 0.8).sum().item()
                print(f"\n⚠️  SUPPRESSION PREVIEW:")
                print(f"  Will suppress: {will_suppress:,} ({100*will_suppress/total:.1f}%)")
                print(f"  Suppression strength: {suppression_strength:.4f}")
                print(f"  Every 50 iterations")
                
                if will_suppress > total * 0.3:
                    print(f"\n🚨 WARNING: Suppressing {100*will_suppress/total:.1f}% of scene!")
                    print(f"   This is TOO MUCH - training may collapse!")
                    print(f"   Consider: Higher threshold or lower strength")
            
            print(f"\n📊 Current State:")
            print(f"  Loss: {loss.item():.6f}")
            print(f"  PSNR: {psnr(image, gt_image).mean().item():.2f} dB")
            print(f"  Points: {gaussians.get_xyz.shape[0]:,}")
            print("🔍"*35 + "\n")

        # ========================================================================
        # TADD-GS: Motion Tracking (START EARLY, even during warmup!)
        # ========================================================================
        motion_tracking_start = 500  # Start tracking motion early
        
        if not disable_tadd and prev_viewpoint is not None and iteration > motion_tracking_start:
            try:
                # Project Gaussians to camera space
                xy_prev = project_gaussians_to_camera(gaussians, prev_viewpoint)
                xy_curr = project_gaussians_to_camera(gaussians, viewpoint_cam)
                
                # USE RAW MOTION (no camera normalization - that's the bug!)
                motion = (xy_prev - xy_curr).norm(dim=-1)
                
                # 🔥 CRITICAL FIX: Adaptive percentile normalization
                # Problem: Raw motion is 5000+, clamping at 10 makes everything 1.0!
                # Solution: Normalize by 95th percentile (adaptive to distribution)
                
                motion_95th = torch.quantile(motion, 0.95)
                
                if motion_95th > 1e-6:
                    # Normalize by 95th percentile
                    motion = motion / (motion_95th + 1e-6)
                else:
                    # No motion detected, keep zeros
                    motion = motion * 0.0
                
                # Allow some values above reference (up to 2.0)
                motion = torch.clamp(motion, 0.0, 2.0)
                
                # Debug: Check normalization produces range
                if iteration == 501:
                    print(f"\n[Adaptive Normalization]")
                    print(f"  Raw 95th percentile: {motion_95th:.1f}")
                    print(f"  Normalized - Min: {motion.min():.4f}, Max: {motion.max():.4f}, Mean: {motion.mean():.4f}")
                    print(f"  Should have RANGE, not all 1.0!\n")
                
                # Update motion variance with EMA
                gaussians.update_motion_variance_ema(motion, alpha=0.95)

                # ONLY apply contrastive learning AFTER warmup
                if iteration > tadd_warmup:
                    # Create features for contrastive learning
                    curr_features = torch.cat([
                        gaussians.get_xyz,
                        gaussians.get_scaling,
                        gaussians.get_opacity
                    ], dim=-1)
                    
                    # Contrastive loss
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
                    print(f"⚠️  Contrastive loss failed: {e}")


        semantic_update_freq = 50
        if not disable_tadd and iteration > motion_tracking_start:
            if iteration % semantic_update_freq == 0:
                try:
                    with torch.no_grad():
                        # Update semantic scores based on rendered image
                        gaussians.update_semantic_scores(image, viewpoint_cam)
                        
                        # Log occasionally
                        if iteration % 500 == 0:
                            num_high_semantic = (gaussians.semantic_scores > 0.6).sum().item()
                            print(f"[Semantic] Updated at iter {iteration} | High scores: {num_high_semantic}")
                            
                except Exception as e:
                    if iteration % 1000 == 0:
                        print(f"⚠️  Semantic update: {e}")

        # ========================================================================
        # Pattern Detection (every 100 iterations)
        # ========================================================================
        if not disable_tadd and iteration % 100 == 0 and iteration > tadd_warmup:
            try:
                gaussians.update_pattern_scores()
            except Exception as e:
                if iteration % 1000 == 0:
                    print(f"⚠️  Pattern detection failed: {e}")

        # ========================================================================
        # GENTLER Distractor Suppression (Less Frequent)
        # ========================================================================
        # ========== HYBRID SUPPRESSION (UPDATED) ==========
        if not disable_tadd and iteration % 500 == 0 and iteration > tadd_warmup:
            try:
                with torch.no_grad():
                    # Get HYBRID distractor scores (motion + semantic)
                    distractor_scores = gaussians.get_distractor_score_hybrid(
                        threshold=distractor_threshold
                    )
                    
                    # Very high confidence distractors only
                    high_confidence = distractor_scores > 0.90
                    
                    num_suppress = high_confidence.sum().item()
                    
                    if num_suppress > 50:  # At least 50 points
                        # Gentle opacity suppression
                        gaussians._opacity[high_confidence] -= suppression_strength
                        
                        # Clamp to valid range
                        gaussians._opacity.data = torch.clamp(gaussians._opacity.data, min=0.0)
                        
                        if iteration % 2000 == 0:
                            print(f"\n[Hybrid Suppression] Iter {iteration}")
                            print(f"  Suppressed: {num_suppress:,} Gaussians")
                            
                            # Show diagnostic info
                            gaussians.diagnose_hybrid_detection()
                    
            except Exception as e:
                if iteration % 1000 == 0:
                    print(f"⚠️  Suppression: {e}")

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            current_psnr = psnr(image, gt_image).mean().item()
            num_points = gaussians.get_xyz.shape[0]

            # ========================================================================
            # SIMPLIFIED PROGRESS BAR: Only Loss and PSNR
            # ========================================================================
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.4f}",
                    "PSNR": f"{current_psnr:.2f}dB"
                })
                progress_bar.update(10)

            # ========================================================================
            # DETAILED METRICS EVERY 500 ITERATIONS
            # ========================================================================
            if iteration % 500 == 0:
                motion_var = 0.0
                num_suppressed = 0
                
                if not disable_tadd and hasattr(gaussians, 'motion_variance') and gaussians.motion_variance is not None:
                    motion_var = gaussians.motion_variance.mean().item()
                    scores = gaussians.get_distractor_score(threshold=distractor_threshold)
                    num_suppressed = (scores > 0.9).sum().item()
                
                print(f"\n{'─'*70}")
                print(f"[ITER {iteration:5d}] Loss: {ema_loss_for_log:.4f} | PSNR: {current_psnr:.2f} dB | Points: {num_points:,}")
                if not disable_tadd:
                    print(f"              Motion: {motion_var:.4f} | Suppressed: {num_suppressed:,}")
                print(f"{'─'*70}")
                
                # Track for early stopping
                psnr_history.append(current_psnr)
                loss_history.append(ema_loss_for_log)
                point_history.append(num_points)

            # ========================================================================
            # EARLY STOPPING: Explosion Detection
            # ========================================================================
            if iteration % 500 == 0 and iteration > tadd_warmup + 500:
                
                # Check 1: Point explosion
                if num_points > max_points:
                    print(f"\n{'🚨'*35}")
                    print(f"EARLY STOP: Point explosion detected!")
                    print(f"  Current: {num_points:,} points")
                    print(f"  Maximum: {max_points:,} points")
                    print(f"  Scene has collapsed - stopping training")
                    print(f"{'🚨'*35}\n")
                    break
                
                # Check 2: PSNR collapse
                if current_psnr < early_stop_psnr_threshold:
                    print(f"\n{'🚨'*35}")
                    print(f"EARLY STOP: PSNR collapse detected!")
                    print(f"  Current PSNR: {current_psnr:.2f} dB")
                    print(f"  Threshold: {early_stop_psnr_threshold} dB")
                    print(f"  Quality too low - stopping training")
                    print(f"{'🚨'*35}\n")
                    break
                
                # Check 3: PSNR not improving (VERY PATIENT)
                if current_psnr > best_psnr:
                    best_psnr = current_psnr
                    psnr_no_improve_count = 0
                else:
                    psnr_no_improve_count += 1
                
                # Only stop if NO improvement for very long time (10,000 iterations default)
                if psnr_no_improve_count >= early_stop_patience:
                    print(f"\n{'⚠️ '*35}")
                    print(f"EARLY STOP: PSNR plateau detected")
                    print(f"  No improvement for {psnr_no_improve_count * 500} iterations")
                    print(f"  Best PSNR: {best_psnr:.2f} dB")
                    print(f"  Current: {current_psnr:.2f} dB")
                    print(f"  This is likely converged, stopping to save time")
                    print(f"{'⚠️ '*35}\n")
                    break
                
                # Check 4: Loss explosion
                if len(loss_history) >= 6:  # Need at least 6 entries
                    recent_losses = loss_history[-5:]
                    if all(l > loss_history[-6] for l in recent_losses):
                        print(f"\n{'🚨'*35}")
                        print(f"EARLY STOP: Loss increasing consistently")
                        print(f"  Last 5 checks: Loss going up")
                        print(f"  Training diverging - stopping")
                        print(f"{'🚨'*35}\n")
                        break

            if iteration == opt.iterations:
                progress_bar.close()

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), dataset)

            if (iteration in saving_iterations):
                print(f"\n[Saving checkpoint at iter {iteration}]")
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
                        # CRITICAL: Reset prev_features to avoid shape mismatch!
                        prev_features = None  # Will be rebuilt next iteration

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print(f"\n[Saving full checkpoint at iter {iteration}]")
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        prev_viewpoint = viewpoint_cam

    # ========================================================================
    # Final Scene Decomposition
    # ========================================================================
    if not disable_tadd and gaussians.motion_variance is not None:
        print("\n" + "="*70)
        print("FINAL SCENE DECOMPOSITION")
        print("="*70)
        
        try:
            distractor_scores = gaussians.get_distractor_score(threshold=distractor_threshold)
            decomposed = decompose_scene(gaussians, distractor_scores, threshold=distractor_threshold)
            decomposed.save_decomposed(scene.model_path, iteration)
            
            print(f"✅ Decomposition complete")
            
        except Exception as e:
            print(f"⚠️  Decomposition failed: {e}")

    print("\n✅ Training complete")
    print(f"   Best PSNR: {best_psnr:.2f} dB")
    print(f"   Final points: {gaussians.get_xyz.shape[0]:,}")

if __name__ == "__main__":
    parser = ArgumentParser(description="TADD-GS Training with Early Stopping")
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
    parser.add_argument("--disable_tadd", action='store_true', default=False)
    parser.add_argument("--tadd_warmup", type=int, default=1500)
    parser.add_argument("--distractor_threshold", type=float, default=0.85)
    parser.add_argument("--suppression_strength", type=float, default=0.001)
    
    # Early stopping
    parser.add_argument("--early_stop_patience", type=int, default=20, help="Stop if no PSNR improvement for N checks (N × 500 iters)")
    parser.add_argument("--early_stop_psnr_threshold", type=float, default=12.0, help="Stop if PSNR drops below this (catastrophic)")
    parser.add_argument("--max_points", type=int, default=200000, help="Stop if points exceed this (explosion)")
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.images = "images"
    
    print("\n" + "="*70)
    print("TADD-GS Training with Early Stopping")
    print("="*70)
    print(f"Dataset: {args.source_path}")
    print(f"Output: {args.model_path}")
    print(f"Iterations: {args.iterations}")
    if not args.disable_tadd:
        print(f"TADD warmup: {args.tadd_warmup}")
        print(f"Early stop patience: {args.early_stop_patience}")
        print(f"Max points: {args.max_points:,}")
    print("="*70 + "\n")
    
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
             suppression_strength=args.suppression_strength,
             early_stop_patience=args.early_stop_patience,
             early_stop_psnr_threshold=args.early_stop_psnr_threshold,
             max_points=args.max_points)
    
    print("\n✅ All done!")