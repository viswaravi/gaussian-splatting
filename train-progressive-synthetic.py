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
from random import randint
import matplotlib.pyplot as plt
from utils.graphics_utils import BasicPointCloud
from train_utils import showTensorDepthImage, showTensorImage, prepare_output_and_logger, training_report, showGSRender, loadPLY, saveTensorAsPLY, renderGS, getGaussianDepthMap, saveTensorAsPLY, showGSRender, showTensorDepthImage, saveCurrentRender

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    

def training(dataset, opt, pipeline, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    # Initialize Gaussians with Zero Tensor
    gaussians = GaussianModel(dataset.sh_degree)

    # Load Actual Gaussians, Camera from PCD
    scene = Scene(dataset, gaussians, shuffle=False)

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    saveCurrentRender(viewpoint_stack[2], gaussians, pipeline, scene.model_path, 0)

    # viewpoint_stack = None
    # ema_loss_for_log = 0.0
    # progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    # first_iter += 1

    # for iteration in range(first_iter, opt.iterations + 1):        
    #     if network_gui.conn == None:
    #         network_gui.try_connect()
    #     while network_gui.conn != None:
    #         try:
    #             net_image_bytes = None
    #             custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
    #             if custom_cam != None:
    #                 net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
    #                 net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
    #             network_gui.send(net_image_bytes, dataset.source_path)
    #             if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
    #                 break
    #         except Exception as e:
    #             network_gui.conn = None

    #     iter_start.record()

    #     gaussians.update_learning_rate(iteration)

    #     # Every 1000 its we increase the levels of SH up to a maximum degree
    #     if iteration % 1000 == 0:
    #         gaussians.oneupSHdegree()

    #     # Pick a random Camera
    #     if not viewpoint_stack:
    #         viewpoint_stack = scene.getTrainCameras().copy()
    #     viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

    #     # Render
    #     if (iteration - 1) == debug_from:
    #         pipe.debug = True

    #     bg = torch.rand((3), device="cuda") if opt.random_background else background

    #     render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
    #     image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

    #     # Loss
    #     gt_image = viewpoint_cam.original_image.cuda()
    #     Ll1 = l1_loss(image, gt_image)
    #     loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
    #     loss.backward()

    #     iter_end.record()

    #     with torch.no_grad():
    #         # Progress bar
    #         ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
    #         if iteration % 10 == 0:
    #             progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
    #             progress_bar.update(10)
    #         if iteration == opt.iterations:
    #             progress_bar.close()

    #         # Log and save
    #         training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
    #         if (iteration in saving_iterations):
    #             print("\n[ITER {}] Saving Gaussians".format(iteration))
    #             scene.save(iteration)

    #         # Densification
    #         if iteration < opt.densify_until_iter:
    #             # Keep track of max radii in image-space for pruning
    #             gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
    #             gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

    #             if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
    #                 size_threshold = 20 if iteration > opt.opacity_reset_interval else None
    #                 gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
    #             if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
    #                 gaussians.reset_opacity()

    #         # Optimizer step
    #         if iteration < opt.iterations:
    #             gaussians.optimizer.step()
    #             gaussians.optimizer.zero_grad(set_to_none = True)

    #         if (iteration in checkpoint_iterations):
    #             print("\n[ITER {}] Saving Checkpoint".format(iteration))
    #             torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
     

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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[500, 1000, 2000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1000, 2000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[1000, 2000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing :" + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
