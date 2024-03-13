import os 
import numpy as np
from PIL import Image
from typing import NamedTuple
from tqdm import tqdm
import math
import scenenet_pb2 as sn

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

def normalize(v):
    return v/np.linalg.norm(v)

def world_to_camera_with_pose(view_pose):
    lookat_pose = position_to_np_array(view_pose.lookat)
    camera_pose = position_to_np_array(view_pose.camera)
    up = np.array([0,1,0])
    R = np.diag(np.ones(4))
    R[2,:3] = normalize(lookat_pose - camera_pose)
    R[0,:3] = normalize(np.cross(R[2,:3],up))
    R[1,:3] = -normalize(np.cross(R[0,:3],R[2,:3]))
    T = np.diag(np.ones(4))
    T[:3,3] = -camera_pose
    return R.dot(T)

def camera_to_world_with_pose(view_pose):
    return np.linalg.inv(world_to_camera_with_pose(view_pose))

def camera_intrinsic_transform(vfov=45,hfov=60,pixel_width=320,pixel_height=240):
    camera_intrinsics = np.zeros((3,4))
    camera_intrinsics[2,2] = 1
    camera_intrinsics[0,0] = (pixel_width/2.0)/math.tan(math.radians(hfov/2.0))
    camera_intrinsics[0,2] = pixel_width/2.0
    camera_intrinsics[1,1] = (pixel_height/2.0)/math.tan(math.radians(vfov/2.0))
    camera_intrinsics[1,2] = pixel_height/2.0
    return camera_intrinsics

def position_to_np_array(position,homogenous=False):
    if not homogenous:
        return np.array([position.x,position.y,position.z])
    return np.array([position.x,position.y,position.z,1.0])

def interpolate_poses(start_pose,end_pose,alpha):
    assert alpha >= 0.0
    assert alpha <= 1.0
    camera_pose = alpha * position_to_np_array(end_pose.camera)
    camera_pose += (1.0 - alpha) * position_to_np_array(start_pose.camera)
    lookat_pose = alpha * position_to_np_array(end_pose.lookat)
    lookat_pose += (1.0 - alpha) * position_to_np_array(start_pose.lookat)
    timestamp = alpha * end_pose.timestamp + (1.0 - alpha) * start_pose.timestamp
    pose = sn.Pose()
    pose.camera.x = camera_pose[0]
    pose.camera.y = camera_pose[1]
    pose.camera.z = camera_pose[2]
    pose.lookat.x = lookat_pose[0]
    pose.lookat.y = lookat_pose[1]
    pose.lookat.z = lookat_pose[2]
    pose.timestamp = timestamp
    return pose

def findTrajectoryfromSceneId(scene_id, trajectories):
    for traj in trajectories.trajectories:
        traj_scene_id = traj.render_path.split('/')[-1]
        if scene_id == traj_scene_id:
            return traj
    return None

def readSceneNetCamInfo(scene_path):
    scene_id = scene_path.split('\\')[-1]
    
    protobuf_path = os.path.join(scene_path, "scenenet_rgbd_train_0.pb")
    images_path = os.path.join(scene_path, "photo")
    depth_path = os.path.join(scene_path, "depth")

    # Read the protobuf file
    trajectories = sn.Trajectories()
    try:
        with open(protobuf_path,'rb') as f:
            trajectories.ParseFromString(f.read())
    except IOError:
        print('Scenenet protobuf data not found at location:{0}'.format(scene_path))
        print('Please ensure you have copied the pb file to the data directory')


    # Extracting the camera intrinsics
    intrinsic_matrix = camera_intrinsic_transform()
    image_width = 320
    image_height = 240
    FovX = np.radians(60)
    FovY = np.radians(45)

    # Select scene trajectory
    traj = findTrajectoryfromSceneId(scene_id, trajectories)
    if traj is None:
        print("Trajectory not found for sceneId {0}".format(scene_id))
        return None, None
    else:
        print("Found Trajectory with sceneId {0} with {1} views".format(traj.render_path , len(traj.views)))

    print("Reading Camera Info")    
    max_frames = len(traj.views)
    frame_step = 1
    cam_infos = []
    frame_ids = []

    for i in tqdm(range(0, max_frames , frame_step)):
        view = traj.views[i]

        # Get camera pose
        ground_truth_pose = interpolate_poses(view.shutter_open,view.shutter_close,0.5)
        W2C = world_to_camera_with_pose(ground_truth_pose)
                
        # frame_id = file.split('.')[0]

        R = W2C[:3, :3]
        T = W2C[:3, 3]
        # print(position)
        # print(rotation)
        # print(pose)

        # Extracting the camera image
        frame_id = str(view.frame_num)
        image_name = str(view.frame_num) + '.jpg'
        image_path = os.path.join(images_path, image_name)
        
        # Load GT Image for Loss Calculation
        image = Image.open(image_path)

        # Load GT depth for Loss Calculation
        depth_image_name = str(view.frame_num) + '.png'
        depth_image_path = os.path.join(depth_path, depth_image_name)

        depth_scale = 1.0  / 1000
        depth = Image.open(depth_image_path)
        depth_array = (np.array(depth) * depth_scale)
        # print(depth_array.shape, type(depth_array), depth_array.dtype, np.max(depth_array), np.min(depth_array),depth_array)
        depth = Image.fromarray(depth_array)

        frame_ids.append(frame_id)
        cam_infos.append(CameraInfo(uid=frame_id, R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth=depth,
                    image_path=image_path, image_name=image_name, width=image_width, height=image_height, intrinsics=intrinsic_matrix))

    return cam_infos, frame_ids