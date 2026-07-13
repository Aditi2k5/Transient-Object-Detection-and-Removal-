import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import lpips

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# Global LPIPS model
lpips_model = None

def get_lpips_model():
    """Lazy initialization of LPIPS model"""
    global lpips_model
    if lpips_model is None:
        lpips_model = lpips.LPIPS(net='alex').cuda()
        print("[Metrics] ✅ LPIPS model loaded (AlexNet)")
    return lpips_model


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, 
                    testing_iterations, scene, renderFunc, renderArgs):
    """
    Training report with PSNR, SSIM, and LPIPS metrics
    """
    
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Full evaluation at test iterations
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        
        lpips_fn = get_lpips_model()
        
        validation_configs = (
            {'name': 'test', 'cameras': scene.getTestCameras()}, 
            {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] 
                                         for idx in range(5, 30, 5)]}
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                # Initialize metrics
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                
                for idx, viewpoint in enumerate(config['cameras']):
                    # Render
                    render_output = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_output["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    
                    # Tensorboard images (first 5 only)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            config['name'] + "_view_{}/render".format(viewpoint.image_name), 
                            image[None], global_step=iteration
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), 
                                gt_image[None], global_step=iteration
                            )
                    
                    # Compute metrics
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    
                    # LPIPS (expects [-1, 1] range)
                    image_lpips = image.unsqueeze(0) * 2.0 - 1.0
                    gt_lpips = gt_image.unsqueeze(0) * 2.0 - 1.0
                    with torch.no_grad():
                        lpips_val = lpips_fn(image_lpips, gt_lpips)
                    lpips_test += lpips_val.item()
                
                # Average over all views
                num_views = len(config['cameras'])
                l1_test /= num_views
                psnr_test /= num_views
                ssim_test /= num_views
                lpips_test /= num_views
                
                # Print results
                print("\n" + "="*70)
                print(f"[ITER {iteration}] Evaluating {config['name'].upper()} set:")
                print(f"  L1:    {l1_test:.6f}")
                print(f"  PSNR:  {psnr_test:.2f} dB")
                print(f"  SSIM:  {ssim_test:.4f}")
                print(f"  LPIPS: {lpips_test:.4f}")
                print("="*70 + "\n")
                
                # Tensorboard logging
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/metrics/l1', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/metrics/psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/metrics/ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/metrics/lpips', lpips_test, iteration)
                
                # Save to CSV file
                metrics_file = os.path.join(scene.model_path, f"metrics_{config['name']}.txt")
                with open(metrics_file, 'a') as f:
                    f.write(f"{iteration},{l1_test:.6f},{psnr_test:.4f},{ssim_test:.4f},{lpips_test:.4f}\n")

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        
        torch.cuda.empty_cache()


def training(dataset, opt, pipe, testing_iterations, saving_iterations, 
             checkpoint_iterations, checkpoint, debug_from, enable_clip=False):
    """
    Main training function with CLIP integration
    """
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    
    # ========== ENABLE CLIP ==========
    if enable_clip:
        gaussians.enable_clip_removal(warmup_iter=500)
    # =================================
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    for iteration in range(first_iter, opt.iterations + 1):        
        
        # Network GUI
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 iterations
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
            
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg["viewspace_points"], \
            render_pkg["visibility_filter"], render_pkg["radii"]

        # ========== CLIP: UPDATE TRACKING ==========
        if enable_clip and iteration >= 500:
            gaussians.update_clip_tracking(iteration, viewpoint_cam, image, visibility_filter)
        # ===========================================

        # Compute base loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # ========== CLIP: ADD OPACITY PENALTY ==========
        if enable_clip and iteration >= 2000:
            clip_penalty = gaussians.get_clip_opacity_penalty(iteration, lambda_penalty=0.01, start_iter=2000)
            loss = loss + clip_penalty
            
            # Log penalty periodically
            if iteration % 500 == 0:
                stats = gaussians.get_clip_statistics()
                print(f"[CLIP Iter {iteration}] Penalty: {clip_penalty.item():.6f} | Stats: {stats}")
        # ================================================
        
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Testing and logging
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, 
                          iter_start.elapsed_time(iter_end), testing_iterations, 
                          scene, render, (pipe, background))
            
            # Save Gaussians at save iterations
            if (iteration in saving_iterations):
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, 
                                               scene.cameras_extent, size_threshold)
                    
                    # ========== CLIP: PRUNE DISTRACTORS ==========
                    if enable_clip and iteration >= 5000 and iteration % 1000 == 0:
                        num_pruned = gaussians.prune_clip_distractors(
                            iteration, 
                            score_threshold=0.3,  # Tune this (0.2-0.4)
                            min_views=10           # Tune this (5-20)
                        )
                    # =============================================
                
                if iteration % opt.opacity_reset_interval == 0 or \
                   (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            # ========== CHECKPOINTING (every 5k, 10k) ==========
            if (iteration in checkpoint_iterations):
                print(f"\n[ITER {iteration}] 💾 Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), 
                          scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            # ===================================================


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Tensorboard
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    
    # Create CSV headers for metrics
    for split in ['test', 'train']:
        metrics_file = os.path.join(args.model_path, f"metrics_{split}.txt")
        with open(metrics_file, 'w') as f:
            f.write("iteration,l1,psnr,ssim,lpips\n")
    
    return tb_writer


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 7_000, 10_000, 15_000, 20_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[5_000, 10_000, 15_000, 20_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[5_000, 10_000, 15_000, 20_000])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    
    # ========== CLIP FLAG ==========
    parser.add_argument("--enable_clip", action='store_true', default=False,
                       help="Enable CLIP-based distractor removal")
    # ===============================
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("\n" + "="*70)
    print(f"🚀 Optimizing: {args.model_path}")
    print("="*70)
    
    if args.enable_clip:
        print("\n" + "="*70)
        print("✅ CLIP DISTRACTOR REMOVAL ENABLED")
        print("="*70 + "\n")

    safe_state(args.quiet)
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training(lp.extract(args), op.extract(args), pp.extract(args), 
            args.test_iterations, args.save_iterations, args.checkpoint_iterations, 
            args.start_checkpoint, args.debug_from, enable_clip=args.enable_clip)

    print("\n" + "="*70)
    print("✅ Training complete!")
    print(f"📊 Metrics: {args.model_path}/metrics_*.txt")
    print(f"💾 Checkpoints: {args.model_path}/chkpnt*.pth")
    print("="*70 + "\n")