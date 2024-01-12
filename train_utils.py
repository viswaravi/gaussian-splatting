import torch
import numpy as np
import matplotlib.pyplot as plt
import uuid
import os
from argparse import Namespace
import open3d as o3d
from gaussian_renderer import render, network_gui

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def showTensorDepthImage(tensor):
    image_array = tensor.cpu().numpy()
    plt.imshow(image_array, cmap='viridis')
    plt.colorbar()
    plt.show()

def showTensorImage(image):
    raster = image.cpu().numpy()
    plt.imshow(raster.astype(int))
    plt.show()

def showGSRender(image):
    image_numpy = image.cpu().detach().numpy()
    # Transpose the dimensions to [height, width, channels]
    image_numpy = np.transpose(image_numpy, (1, 2, 0))

    # Display the image using Matplotlib
    plt.imshow(image_numpy)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()


def loadPLY(ply_file):
    pcd = o3d.io.read_point_cloud(ply_file)
    # o3d.geometry.estimate_normals(pcd, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    # normals = np.asarray(pcd.normals)

    points = torch.from_numpy(points).float().cuda()
    colors = torch.from_numpy(colors).float().cuda()
    return points, colors

def saveTensorAsPLY(points,colors, file_name):
    point_cloud_np = points.cpu().detach().numpy() 
    # Create an Open3D point cloud
    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud_np)
    point_cloud_o3d.colors = o3d.utility.Vector3dVector(colors.cpu().detach().numpy())
    # Save the point cloud as a PLY file
    o3d.io.write_point_cloud(file_name, point_cloud_o3d)


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene , renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
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