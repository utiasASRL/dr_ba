import struct
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import sys
from map.voxel_map import Map


# Check if any argument is provided for base path

if len(sys.argv) > 1:
    base_path = sys.argv[1]
else:
    base_path =  '/home/dl/Documents/phd/dev/dr_ba/output/ba_results/run_96'

voxel_path = osp.join(base_path, 'voxel_map.bin')

vox_map = Map(res=1.0)  # Resolution will get overwritten when loading
vox_map.load_from_binary(voxel_path)
print("Voxel map size:", vox_map.size())
vox_map.plot(show=True)



with open(voxel_path, "rb") as f:
    # Metadata
    res, = struct.unpack("<d", f.read(8))
    num_poses, = struct.unpack("<I", f.read(4))
    num_voxels, = struct.unpack("<I", f.read(4))

    # Poses (36 bytes each)
    poses = []
    pose_struct = struct.Struct("<idddd")
    assert pose_struct.size == 36
    for _ in range(num_poses):
        pose_id, x, y, yaw, ate = pose_struct.unpack(
            f.read(pose_struct.size)
        )
        poses.append((pose_id, x, y, yaw, ate))

    # Voxels (16 bytes each)
    voxels = []
    voxel_struct = struct.Struct("<iid")
    assert voxel_struct.size == 16
    for _ in range(num_voxels):
        x, y, intensity = voxel_struct.unpack(f.read(voxel_struct.size))
        voxels.append((x, y, intensity))


print("Number of voxels loaded:", len(voxels))
print("Number of poses loaded:", len(poses))


# # Visualize voxel map
# voxels = np.array(voxels)
# print(voxels.shape)
# print("Voxel map resolution (m):", res)
# x = voxels[:, 0] * res
# y = voxels[:, 1] * res
# intensities = voxels[:, 2]

# fig = plt.figure(figsize=(10, 8))
# plt.scatter(x, y, c=intensities, cmap='viridis', s=1)
# plt.colorbar(label='Intensity')
# plt.xlabel('X (m)')
# plt.ylabel('Y (m)')
# plt.title('Voxel Map Visualization')
# plt.axis('equal')

# # Load in poses and color by error if they exist
# pose_path = osp.join(base_path, 'scan_poses.csv')
# if osp.exists(pose_path):
#     # scan_id, x, y, yaw, x_gt, y_gt, yaw_gt
#     data = np.loadtxt(pose_path, delimiter=",", skiprows=1)
#     # Expand if data is only 1 entry
#     if data.ndim == 1:
#         data = data.reshape(1, -1)
#     scan_ids = data[:, 0].astype(int)
#     x_est = data[:,1]
#     y_est = data[:,2]
#     yaw_est = data[:,3]
#     x_gt = data[:,4]
#     y_gt = data[:,5]
#     yaw_gt = data[:,6]
#     ate = np.round(data[:,7], 5)


#     print(max(ate), min(ate))

#     trans_errors = np.sqrt((x_est - x_gt)**2 + (y_est - y_gt)**2)
#     yaw_errors = np.abs(yaw_est - yaw_gt) * (180.0 / np.pi)

#     print("RMSE Translational Error (m):", np.sqrt(np.mean(trans_errors**2)))
#     print("RMSE Rotational Error (deg):", np.sqrt(np.mean(yaw_errors**2)))

#     sc = plt.scatter(x_est, y_est, c=ate, cmap='hot', s=20, edgecolors='k', label='Estimated Poses')
#     plt.colorbar(sc, label='ATE')
#     plt.legend()

# plt.show()
