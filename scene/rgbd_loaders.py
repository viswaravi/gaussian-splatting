import os 
import open3d as o3d
import numpy as np
# from scene.dataset_readers import CameraInfo
from PIL import Image
from scipy.spatial.transform import Rotation as ScipyRotation

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

def readRGBDCamInfo(path):
    cam_infos = []
    frame_ids = []
    camera_params = {} # Camera Intrinsic parameters
    camera_poses = {} # Camera Extrinsic Matrix
    configs_path = os.path.join(path, "config")
    images_path = os.path.join(path, "rgb")
    config_files = os.listdir(configs_path)
    config_files.pop() # Remove configuration.txt
    config_files.sort()
    frame_step = 5

    # Read Config
    rgb_camera_params, depth_camera_params, relative_positions = readRGBDConfig(os.path.join(configs_path, "configuration.txt"))
    camera_params["rgb"] = rgb_camera_params
    camera_params["depth"] = depth_camera_params
    camera_params["relative"] = relative_positions

    # Read the camera extrinsics and intrinsics
    # for i in range(20): # Process first 20 frames
    for i in range(0, len(config_files), frame_step):
        file = config_files[i]
        if file.startswith("campose-rgb-"):
            frame_id = file.split('-')[2].split('.')[0]
            config_file = os.path.join(configs_path, file)

            with open(config_file, 'r') as file:
                lines = file.readlines()

                # Extracting position
                position_str = lines[0].replace('position=', '').split('\n')[0]
                position = np.array([float(i) for i in position_str.strip('()').split(',')])

                # Extracting rotation as a quaternion
                rotation_str = lines[1].replace('rotation_as_quaternion=', '').split('\n')[0]
                rotation = np.array([float(i) for i in rotation_str.strip('()').split(',')])

                # Extracting the 4x4 pose matrix
                pose_str = lines[3:]
                pose = np.array([[float(i) for i in row.strip('(').split(')')[0].split(',')] for row in pose_str if row != ''])    

                # print(position)
                # print(rotation)
                # print(pose)

                # Extracting the camera extrinsics
                T = np.array(position)
                R = ScipyRotation.from_quat(rotation).as_matrix().transpose()
                # R = np.transpose(qvec2rotmat(rotation))

                # Extracting the camera image
                image_name = 'rgb-' + frame_id + '.png'
                image_path = os.path.join(images_path, image_name)
                # Load GT Image for Loss Calculation
                image = Image.open(image_path)
                # image = None

                # Extracting the camera intrinsics
                FovY = rgb_camera_params['vFov']
                FovX = rgb_camera_params['hFov']

                frame_ids.append(frame_id)
                camera_poses[frame_id] = pose
                cam_infos.append(CameraInfo(uid=frame_id, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
    return cam_infos, frame_ids, camera_params, camera_poses 

# Reads Data from PLY file
def loadPointsFromPLY(ply_path):
    # Read the point cloud
    pcd = o3d.io.read_point_cloud(ply_path)
    return pcd

# Reads Data from RGBD Images and stores them as a ply file
def loadPointsFromRGBD(path, frameid, ply_file_path, camera_params, camera_pose , save = False):
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
        o3d.io.write_point_cloud(ply_file_path, o3d_pc)
    return o3d_pc