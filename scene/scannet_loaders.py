import os 
import open3d as o3d
import numpy as np
from PIL import Image
from typing import NamedTuple
from tqdm import tqdm
import cv2
   
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    depth: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    intrinsics: np.array

def readFileMatrix(file_path):
    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # Parse the matrix values
    matrix = []
    for line in lines:
        values = line.strip().split()
        row = [float(value) for value in values]
        matrix.append(row)

    # Convert the matrix to a numpy array
    matrix = np.array(matrix)
    return matrix

# def getFOVfromIntrsinicMatrix(intrinsic_matrix):
#     # Extracting focal lengths and principal points
#     f_x = intrinsic_matrix[0, 0]
#     f_y = intrinsic_matrix[1, 1]
#     c_x = intrinsic_matrix[0, 2]
#     c_y = intrinsic_matrix[1, 2]

#     # Calculating field of view
#     fov_x = 2 * np.arctan2(c_x, f_x)
#     fov_y = 2 * np.arctan2(c_y, f_y)

#     # Converting radians to degrees
#     fov_x_deg = np.degrees(fov_x)
#     fov_y_deg = np.degrees(fov_y)

#     return fov_x_deg, fov_y_deg

def readScanNetConfig(config_files_path):
    # Define dictionaries to hold camera parameters
    rgb_camera_params = {}
    depth_camera_params = {}
    
    # extrinsic_color = readFileMatrix(os.path.join(config_files_path, "extrinsic_color.txt"))
    # extrinsic_depth = readFileMatrix(os.path.join(config_files_path, "extrinsic_depth.txt"))
    intrinsic_color = readFileMatrix(os.path.join(config_files_path, "intrinsic_color.txt"))
    intrinsic_depth = readFileMatrix(os.path.join(config_files_path, "intrinsic_depth.txt"))

    rgb_intrinsic = cv2.calibrationMatrixValues(intrinsic_color[:3,:3],(1296, 968), apertureWidth=0, apertureHeight=0)
    depth_intrinsic = cv2.calibrationMatrixValues(intrinsic_depth[:3,:3],(640, 480), apertureWidth=0, apertureHeight=0)
    
    rgb_camera_params['hFov'], rgb_camera_params['vFov'] = rgb_intrinsic[0], rgb_intrinsic[1]
    depth_camera_params['hFov'], depth_camera_params['vFov'] = depth_intrinsic[0], depth_intrinsic[1]

    return rgb_camera_params, depth_camera_params, intrinsic_color

def getWorld2View(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    # Add Homogeneous Coordinate
    Rt = np.eye(4)
    Rt[:3, :3] = R
    Rt[:3, 3] = t

    # Add Translation and Scale
    C2W = Rt
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center

    # Invert
    R = C2W[:3, :3]
    t = C2W[:3, 3]
    R_inv = R.T
    T_inv = -R_inv @ t
    world_to_camera = np.eye(4)
    world_to_camera[:3, :3] = R_inv
    world_to_camera[:3, 3] = T_inv

    return np.float32(world_to_camera)

def readScanNetCamInfo(scene_path):
    cam_infos = []
    frame_ids = []

    scene_path = os.path.join(scene_path, "sens_read")
    configs_path = os.path.join(scene_path, "pose")
    images_path = os.path.join(scene_path, "color")
    config_files = os.listdir(configs_path)
    config_files.sort(key=lambda x: int(x.split('.')[0])) # Sort the files based on frame id
    frame_step = 10

    # print(config_files)

    # Read Config
    rgb_camera_params, depth_camera_params, intrinsic_color = readScanNetConfig(os.path.join(scene_path, "intrinsic"))

    config_file_limit = 150 # Limiting the number of frames to process for GPU memory constraints
    # Read the camera extrinsics and intrinsics
    # for i in range(20): # Process first 20 frames
    print("Reading Camera Info")
    for i in tqdm(range(0, config_file_limit , frame_step)):
        file = config_files[i]
        config_file = os.path.join(configs_path, file)

        # Extracting frame id
        frame_id = file.split('.')[0]
        pose = readFileMatrix(config_file)

        # print(i, frame_id)
        # Extracting position and Rotation
        T = pose[:3, 3]
        R = pose[:3, :3]
        W2C = getWorld2View(R, T)
        R = W2C[:3, :3]
        T = W2C[:3, 3]
        # print(position)
        # print(rotation)
        # print(pose)

        # Extracting the camera image
        image_name = frame_id + '.jpg'
        image_path = os.path.join(images_path, image_name)
        # Load GT Image for Loss Calculation
        image = Image.open(image_path)
        # image = None

        # Extracting the camera intrinsics
        FovY = np.radians(rgb_camera_params['vFov'])
        FovX = np.radians(rgb_camera_params['hFov'])

        # print(FovY, FovX)

        frame_ids.append(frame_id)
        cam_infos.append(CameraInfo(uid=frame_id, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                    image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], intrinsics=intrinsic_color))

    return cam_infos, frame_ids