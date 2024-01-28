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
    
def rasterizePoints(viewpoint_cam, points, colors=None, sub_pixel_level=1):
    intrinsics = viewpoint_cam.intrinsics
    c_x = intrinsics[0,2]
    c_y = intrinsics[1,2]
    f_x = intrinsics[0,0]
    f_y = intrinsics[1,1]
    W2C = viewpoint_cam.world_view_transform
    full_proj_transform = viewpoint_cam.full_proj_transform
    img_width, img_height = viewpoint_cam.image_width, viewpoint_cam.image_height

    # print('W2C:',W2C.transpose(0,1))
    # print('FP:',full_proj_transform.transpose(0,1))

    # Project points using projection matrix P in torch cuda and get result in NDC
    points_homogeneous = torch.cat((points, torch.ones(points.shape[0], 1, device=points.device)), dim=1)
    # print(points_homogeneous.shape, colors.shape, W2C)
    
    points_camera_homogeneous = torch.matmul(points_homogeneous, W2C) # Clip space coordinates
    projected_points_homogeneous = torch.matmul(points_homogeneous, full_proj_transform)

    assert projected_points_homogeneous.shape[1] == 4
    # Extract x, y, z, w from the tensor
    x, y, z, w = projected_points_homogeneous[:, 0], projected_points_homogeneous[:, 1], projected_points_homogeneous[:, 2], projected_points_homogeneous[:, 3]
    x_ndc = x / w
    y_ndc = y / w
    z_ndc = z / w
    projected_points_NDC = torch.stack((x_ndc, y_ndc, z_ndc), dim=1)
    projected_points_IS = (projected_points_NDC + 1) / 2

    u = projected_points_IS[:,0] * img_width
    v = projected_points_IS[:,1] * img_height
    Z = points_camera_homogeneous[:, 2]

    u = u.to(torch.int16)
    v = v.to(torch.int16)

    # Filter points outside image space
    image_space_mask = (u >= 0) & (u < img_width) & (v >= 0) & (v < img_height )
    u = u[image_space_mask]
    v = v[image_space_mask]
    points_filtered = points[image_space_mask]
    depth_filtered = Z[image_space_mask]
    # raster = showRasterizedImage(u,v, colors_filtered, img_height, img_width)

    try:
        # print(u.shape, v.shape, colors_filtered.shape)
        raster_color = torch.zeros((img_height, img_width, 3), dtype=torch.uint8 , device=points.device)
        raster_depth_map = torch.zeros((img_height, img_width), dtype=Z.dtype, device=points.device)

        IS_point_positions = torch.zeros((img_height, img_width, 3), dtype=torch.float16 , device=points.device)
        IS_point_colors = torch.zeros((img_height, img_width, 3), dtype=torch.float16 , device=points.device)

        # Create Indices
        u_long = u.to(torch.long)
        v_long = v.to(torch.long)
        # Store Point Positions
        raster_depth_map[v_long, u_long] = depth_filtered
        IS_point_positions[v_long, u_long] = points_filtered.to(torch.float16)
        
        # Store Point Colors
        if colors is not None:
            colors_filtered = colors[image_space_mask]
            raster_color[v_long, u_long] = (colors_filtered * 255).to(torch.uint8)
            IS_point_colors[v_long, u_long] = colors_filtered.to(torch.float16)
        else:
            colors_filtered = None


    except Exception as e:
        print(f"Error: {e}")

    # pixel_indices_long = torch.clamp(pixel_indices_long, 0, img_width - 1)
    # pixel_indices_long = torch.cat((pixel_indices_long[:, 1:], pixel_indices_long[:, :1]), dim=1)  # Swap x and y for indexing
    # colors_uint8 = (colors * 255).to(torch.uint8)
    # img[pixel_indices_long[:, 1], pixel_indices_long[:, 0]] = colors_uint8

    pixel_indices = torch.stack((u,v), dim=1)
    # print(pixel_indices.shape)
    # # print(pixel_indices.shape, pixel_indices.min(), pixel_indices.max())
    return points_filtered,colors_filtered,raster_depth_map, pixel_indices, raster_color, IS_point_positions, IS_point_colors


def training(dataset, opt, pipeline, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    # Initialize Profiler
    prof = torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=0),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],)

    prof.start()

    # Progressive Fusion
    first_viewpoint = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)
    gaussians.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    ply_path = os.path.join(dataset.source_path, "ply")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ema_loss_for_log = 0.0

    viewpoint_stack = scene.getTrainCameras().copy()
    num_viewpoints = 6 # len(viewpoint_stack)
    progress_bar = tqdm(range(first_viewpoint, num_viewpoints), desc="Training progress")
    first_viewpoint += 1

    # Scene is Already loaded with First Frame
    frame_batch_size = 5
    frame_step = 2
    visibility_threshold = 0.5
    batch_iter = 10

    # Densification
    density_every = 1

    assert viewpoint_stack != None, "Viewpoint Stack is None"

    for viewpoint_index in range(first_viewpoint, num_viewpoints, frame_step):
        # Fuse K frames into Global Field
        # Gaussian Densification / Fusion
        start_cam_index = viewpoint_index
        end_cam_index = min(num_viewpoints, viewpoint_index + frame_step)

        print("Fusing Frames: ", list(range(start_cam_index, end_cam_index)))
        for K_i in range(start_cam_index, end_cam_index):
            viewpoint_cam = viewpoint_stack[K_i]
            ply_file_name = viewpoint_cam.colmap_id + ".ply"        

            ply_file_path = os.path.join(ply_path, ply_file_name)
            assert os.path.exists(ply_file_path), "PLY File does not exist"

            pcd_points, pcd_colors = loadPLY(ply_file_path)
            filtered_points, filtered_colors, current_depth_map, indices, current_raster, raster_point_positions, raster_point_colors = rasterizePoints(viewpoint_cam, pcd_points, pcd_colors)
            global_depth_map, global_visibility_map = getGaussianDepthMap(viewpoint_cam, gaussians, pipeline)
            
            # Identify Free Pixels
            free_pixel_mask = (global_visibility_map < visibility_threshold)
            mask = free_pixel_mask.int()

            # Apply Mask to point positions
            raster_point_positions_masked = raster_point_positions.view(-1, 3) * mask.view(-1, 1)
            current_raster_masked = raster_point_colors.view(-1, 3) * mask.view(-1, 1)

            # showTensorImage(current_raster_masked.view(current_raster.shape)*255)

            # Filter points which are at origin
            invalid_points_mask = torch.all(raster_point_positions_masked == 0, dim=1, keepdim=True)
            raster_point_positions_filtered = raster_point_positions_masked[~invalid_points_mask.view(-1)]
            current_raster_filtered = current_raster_masked[~invalid_points_mask.view(-1)]
            
            # Fuse points into global field
            # Initialize 3D Gaussians with new points and add to scene
            scene.extendGaussians(BasicPointCloud(points=raster_point_positions_filtered.cpu().numpy(), colors=current_raster_filtered.cpu().numpy(), normals=None))
    
        # Clear Memory
        current_raster_filtered = None
        raster_point_positions_filtered = None
        raster_point_positions_masked = None
        current_raster_masked = None
        raster_point_positions = None
        raster_point_colors = None
        current_raster = None
        # torch.cuda.empty_cache()

        # Optimize over K frames
        # Gaussian Optimization
        for iter in range(batch_iter):
            for cam_index in range(start_cam_index, end_cam_index):
                # Pick the current Camera
                viewpoint_cam = viewpoint_stack[cam_index]
                
                # Render
                bg = torch.rand((3), device="cuda") if opt.random_background else background

                render_pkg = render(viewpoint_cam, gaussians, pipeline, bg)
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

                # Loss
                gt_image = viewpoint_cam.original_image.cuda()
                Ll1 = l1_loss(image, gt_image)
                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
                loss.backward()
    
            # Post loss accumulation of K frames
            with torch.no_grad():
                # Update Gaussians
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})

        # Densify the Scene after each batch optimization
        # Keep track of max radii in image-space for pruning
        gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
        # print(viewspace_point_tensor.shape, visibility_filter.shape)
        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
        size_threshold = 20
        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
        
        # Post Fusion and Optimization of K frames
        with torch.no_grad():
            # Update Progress Bar
            progress_bar.update(frame_step)


            # Log and save
            if (end_cam_index in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(end_cam_index))
                torch.save((gaussians.capture(), end_cam_index), scene.model_path + "/chkpnt" + str(end_cam_index) + ".pth")

            if (end_cam_index in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(end_cam_index))
                scene.save(end_cam_index)
            
        prof.step()

    # Finished Training
    prof.stop()
    progress_bar.close()
    torch.cuda.empty_cache()

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

    # In Progressive iteration numbers is used like frames
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[20, 30, 60])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[20, 30, 60])
    
    parser.add_argument("--quiet", action="store_true")
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
