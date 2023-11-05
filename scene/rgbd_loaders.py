import os 
import open3d as o3d
import numpy as np

def readRGBDConfig(config_file):
    # Define dictionaries to hold camera parameters
    rgb_camera_params = {}
    depth_camera_params = {}
    relative_positions = {}
    
    with open(config_file, 'r') as file:
        data = file.read().split('\n\n')
    
        # Read RGB camera parameters
        rgb_data = data[0].split('\n')
        for line in rgb_data[1:4]:
            key, value = line.split('=')
            if ',' in value:
                value = tuple(map(float, value.split(',')))
            else:
                value = tuple(map(int, value.split('x')))
            rgb_camera_params[key] = value

        vFOV, hFOV = rgb_data[4].split(',')
        key, value = vFOV.split('=')
        rgb_camera_params[key] = float(value.strip('°'))
        key, value = hFOV.split('=')
        rgb_camera_params[key.strip(' ')] = float(value.strip('°'))

        # Read Depth camera parameters
        depth_data = data[1].split('\n')
        for line in depth_data[1:4]:
            key, value = line.split('=')
            if ',' in value:
                value = tuple(map(float, value.split(',')))
            else:
                value = tuple(map(int, value.split('x')))
            depth_camera_params[key] = value

        vFOV, hFOV = depth_data[4].split(',')
        key, value = vFOV.split('=')
        depth_camera_params[key] = float(value.strip('°'))
        key, value = hFOV.split('=')
        depth_camera_params[key.strip(' ')] = float(value.strip('°'))

    
        # Read relative positions of camera components
        rel_pos_data = data[2].split('\n')
        for line in rel_pos_data[1:]:
            key, value = line.split(': ')
            value = tuple(map(float, value.strip('(').strip(')').split(',')))
            relative_positions[key] = value

        return rgb_camera_params, depth_camera_params, relative_positions


def loadPlyfromRGBD(path, frameid, ply_path, camera_params, camera_pose , save = False):
    color_file = os.path.join(path, "rgb-" + frameid + ".png")
    depth_file = os.path.join(path, "gt-rgb-depth-" + frameid + ".png")

    # print(color_file)
    # print(depth_file)

    # Read the depth and color images
    depth_image = o3d.io.read_image(depth_file)
    color_image = o3d.io.read_image(color_file)
    
    # Convert images to numpy arrays
    depth_array = np.asarray(depth_image)
    color_array = np.asarray(color_image)

    width = depth_array.shape[1]
    height = depth_array.shape[0]

    # Intrinsic parameters of the camera
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    cx = camera_params["rgb"]['cx,cy,fx,fy'][0]
    cy = camera_params["rgb"]['cx,cy,fx,fy'][1]
    fx = camera_params["rgb"]['cx,cy,fx,fy'][2] 
    fy = camera_params["rgb"]['cx,cy,fx,fy'][3]
    intrinsic.set_intrinsics(width=width, height=height, cx=cx, cy=cy, fx=fx, fy=fy)
    
    # Extrinsics of the camera
    extrinsic = camera_pose

    # Create a point cloud from the depth and color information
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_image, depth_trunc=10.0, convert_rgb_to_intensity=False)
    o3d_pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, extrinsic)

    # Store the point cloud as a ply file
    if save:
        o3d.io.write_point_cloud(ply_path, o3d_pc)
    return o3d_pc