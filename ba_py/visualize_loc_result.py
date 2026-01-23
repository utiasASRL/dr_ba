import struct
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import sys
from map.voxel_map import Map


# Check if any argument is provided for base path

if len(sys.argv) < 2:
    print("Usage: python visualize_loc_result.py <loc_path>")
    sys.exit(1)

loc_path = sys.argv[1]
if len(sys.argv) > 3:
    title_name = sys.argv[2]
else:
    title_name = "Localization Results"

voxel_path = osp.join(loc_path, 'voxel_map.bin')
# map_id,scan_id,est_x,est_y,est_yaw,gt_x,gt_y,gt_yaw
loc_result_path = osp.join(loc_path, 'loc_results.csv')

# Load localization results
loc_results = []
errs = []
with open(loc_result_path, 'r') as f:
    next(f)  # Skip header
    for line in f:
        tokens = line.strip().split(',')
        map_id = int(tokens[0])
        scan_id = int(tokens[1])
        est_x = float(tokens[2])
        est_y = float(tokens[3])
        est_yaw = float(tokens[4]) * (180.0 / np.pi)  # Convert to degrees
        gt_x = float(tokens[5])
        gt_y = float(tokens[6])
        gt_yaw = float(tokens[7]) * (180.0 / np.pi)  # Convert to degrees
        # Wrap yaw errors to [-180, 180]
        yaw_err = est_yaw - gt_yaw
        if yaw_err > 180.0:
            yaw_err -= 360.0
        elif yaw_err < -180.0:
            yaw_err += 360.0
        est_yaw = gt_yaw + yaw_err
        loc_results.append((map_id, scan_id, est_x, est_y, est_yaw, gt_x, gt_y, gt_yaw))
        errs.append((est_x - gt_x, est_y - gt_y, est_yaw - gt_yaw))

print("Number of localization results loaded:", len(loc_results))
vox_map = Map(res=1.0)  # Resolution will get overwritten when loading
vox_map.load_from_binary(voxel_path)
print("Voxel map size:", vox_map.size())
vox_map.plot_loc_result(loc_results, show=True, title=title_name)

print("RMSE (m), (m), (deg):")
errs = np.array(errs)
rmse_x = np.sqrt(np.mean(errs[:, 0]**2))
rmse_y = np.sqrt(np.mean(errs[:, 1]**2))
rmse_yaw = np.sqrt(np.mean(errs[:, 2]**2))
print(f"{rmse_x:.3f}, {rmse_y:.3f}, {rmse_yaw:.3f}")


# Plot histogram of localization errors
errs = np.array(errs)
fig, axs = plt.subplots(3, 1, figsize=(8, 12))
# Set title
fig.suptitle(title_name, fontsize=16)
axs[0].hist(errs[:, 0], bins=30, color='blue', alpha=0.7)
axs[0].set_title('Histogram of X Errors (m)')
axs[0].set_xlabel('Error (m)')
axs[0].set_ylabel('Frequency')
axs[1].hist(errs[:, 1], bins=30, color='green', alpha=0.7)
axs[1].set_title('Histogram of Y Errors (m)')
axs[1].set_xlabel('Error (m)')
axs[1].set_ylabel('Frequency')
axs[2].hist(errs[:, 2], bins=30, color='red', alpha=0.7)
axs[2].set_title('Histogram of Yaw Errors (deg)')
axs[2].set_xlabel('Error (deg)')
axs[2].set_ylabel('Frequency')
plt.tight_layout()
plt.show()