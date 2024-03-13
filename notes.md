# About
This repository is an extension to the original gaussian splatting reconstruction. Instead of using a precalculated point cloud from COLMAP as input, this repository loads RGBD frames directly and reconstructs the scene

# Datasets
There are 3 additional dataset loaders have been added to this project in addition to default loaders available in orginal respository

### ScanNet
This loads [SanNet](https://github.com/ScanNet/ScanNet) dataset using the **scannet_loaders.py** file

### SceneNet
This loads [SceneNet](https://robotvault.bitbucket.io/scenenet-rgbd.html) which is a synthetic dataset using the **scenenet_loaders.py** file

### Synthetic RGBD
This loads a custom synthetic RGBD dataset using the **rgbd_loaders.py** file

# Synthetic Dataset
Gaussian Splatting uses a OpenCV coordinate system, but the synthetic datasets uses OpenGL based coordinate system. so to load them we need to flip the Y and Z axes before converting to global coordinate system.

### Data Loading Pipeline
$$
camera2world = pose
$$

<br>

$$
GlobalPoints = (camera2world) . (flipYZ) . (CameraPoints)
$$

<br>

$$
CameraPoints = (flipYZ) . (world2camera) . (GlobalPoints)
$$

# Reconstruction
Scene reconstruction from RGBD frames is done in three steps

## Points Fusion
For every K frames this step adds new point to the scene from the current camera point of view based on the following conditions

- If the current pixel has visibility value less than visibility threshold (0.5)
- If the new point at the current pixel has closer depth to camera than the existing scene surface

The newly added gaussian points are initialized based on original work

### Feature Initialization
**Colors** - RGB values from each pixel is converted to SH coefficients

**Point positions** - Obtained from converting the points from RGBD frame to 
global coordinate systen

**Scaling** -  Initialized based on the logarithm of the square root of the distances (dist2) between the points. Simple-KNN package is used to fine K nearest points for each point

**Rotation** -  Initialized with all zeros except for the first column, where all values are set to 1

**Opacity** -  Initialized using the inverse sigmoid function with a constant value.

## Feature Tuning
This step fine tunes the features of newly added point around the scene where they have been added using differentiable gaussian splatting

The model renders the K-2 to K frames and update the point features for N Iterations

### Losses

**RGB Color Loss**

This is a linear combination of RGB L1 loss and SSIM Loss
```
rgb_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
```

**Depth Loss**

Mean absolute difference between Ground Truth depth from RGBD frame and the 
```
invalid_depth_mask = (depth_map == 0)
depth_error = torch.abs(depth_map - viewpoint_cam.original_depth)
depth_loss = (depth_error * ~invalid_depth_mask).mean()
```

## Densification
This steps performs the densify and prune of all the Gaussians in the scene
