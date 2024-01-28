# Synthetic Dataset
## Data Loading Pipeline

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

## Data
Total number of Frames - 300

Sampled Frames - 60


## Training

### Fusion Stage
- Fuse K frames at each step with overlapping window
- Optimise for N iteration 

### Fine Tuning Stage
- Fine Tune over N frames