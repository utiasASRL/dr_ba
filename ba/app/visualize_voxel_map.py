import struct
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp

base_path =  '/home/dl/Documents/phd/dev/dr_ba/output/ba_results/run_38'
voxel_path = osp.join(base_path, 'voxel_map.bin')
with open(voxel_path, "rb") as f:
    res, = struct.unpack("d", f.read(8))

    voxels = []
    while True:
        bytes = f.read(16)  # 2 ints (x, y) and 1 double (intensity)
        if not bytes:
            break
        x, y, intensity = struct.unpack("ii d", bytes)
        voxels.append((x, y, intensity))

# Visualize voxel map
voxels = np.array(voxels)
x = voxels[:, 0] * res
y = voxels[:, 1] * res
intensities = voxels[:, 2]

fig = plt.figure(figsize=(10, 8))
plt.scatter(x, y, c=intensities, cmap='viridis', s=1)
plt.colorbar(label='Intensity')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Voxel Map Visualization')
plt.axis('equal')

# Load in poses and color by error if they exist
pose_path = osp.join(base_path, 'scan_poses.csv')
if osp.exists(pose_path):
    # scan_id, x, y, yaw, x_gt, y_gt, yaw_gt
    data = np.loadtxt(pose_path, delimiter=",", skiprows=1)
    scan_ids = data[:, 0].astype(int)
    x_est = data[:,1] - data[0,1]
    y_est = data[:,2] - data[0,2]
    yaw_est = data[:,3] - data[0,3]
    x_gt = data[:,4] - data[0,4]
    y_gt = data[:,5] - data[0,5]
    yaw_gt = data[:,6] - data[0,6]

    trans_errors = np.sqrt((x_est - x_gt)**2 + (y_est - y_gt)**2)
    yaw_errors = np.abs(yaw_est - yaw_gt) * (180.0 / np.pi)

    print("RMSE Translational Error (m):", np.sqrt(np.mean(trans_errors**2)))
    print("RMSE Rotational Error (deg):", np.sqrt(np.mean(yaw_errors**2)))

    sc = plt.scatter(x_est, y_est, c=trans_errors, cmap='hot', s=20, edgecolors='k', label='Estimated Poses')
    plt.colorbar(sc, label='Translational Error (m)')
    plt.legend()

plt.show()
