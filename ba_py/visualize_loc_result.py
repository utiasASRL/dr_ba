import struct
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import sys
from map.voxel_map import Map
import argparse


parser = argparse.ArgumentParser(description="Visualize localization results")
parser.add_argument(
    "--loc_path",
    type=str,
    required=True,
    help="Path to localization results"
)

parser.add_argument(
    "--title",
    type=str,
    default="Localization Results",
    help="Plot title (optional)"
)

parser.add_argument(
    "--show",
    action="store_true",
    help="Show the plot (flag)"
)

args = parser.parse_args()

loc_path = args.loc_path
title_name = args.title
show = args.show

voxel_path = osp.join(loc_path, 'voxel_map.bin')
# map_id,scan_id,est_x,est_y,est_yaw,gt_x,gt_y,gt_yaw
loc_result_path = osp.join(loc_path, 'loc_results.csv')

# Load localization results
loc_results = []
errs = []
stds = []
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
        if len(tokens) > 8:
            std_x = float(tokens[8])
            std_y = float(tokens[9])
            std_yaw = float(tokens[10]) * (180.0 / np.pi)  # Convert to degrees
            stds.append((std_x, std_y, std_yaw))
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
vox_map.plot_loc_result(loc_results, show=False, save_path=loc_path, title=title_name)
# vox_map.plot_loc_paper(loc_results, show=False, save_path=loc_path, title=title_name)

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
print("Saving error histograms to:", loc_path)
plt.savefig(osp.join(loc_path, 'loc_error_histograms.png'), dpi=300, bbox_inches="tight")

# Plot errors and 3-sigma bounds over frames
if len(stds) > 0:
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    # Set title
    fig.suptitle(title_name, fontsize=16)
    frame_ids = [res[1] for res in loc_results]
    errs = np.array(errs)
    stds = np.array(stds)
    axs[0].plot(frame_ids, errs[:, 0], label='X Error', color='blue')
    axs[0].fill_between(frame_ids,
                    -3 * stds[:, 0],
                    3 * stds[:, 0],
                    color='blue',
                    alpha=0.2,
                    label='3-Sigma Bound')
    axs[0].set_title('X Errors with 3-Sigma Bounds')
    axs[0].set_xlabel('Frame ID')
    axs[0].set_ylabel('Error (m)')
    axs[0].legend()
    axs[1].plot(frame_ids, errs[:, 1], label='Y Error', color='green')
    axs[1].fill_between(frame_ids,
                    -3 * stds[:, 1],
                    3 * stds[:, 1],
                    color='blue',
                    alpha=0.2,
                    label='3-Sigma Bound')
    axs[1].set_title('Y Errors with 3-Sigma Bounds')
    axs[1].set_xlabel('Frame ID')
    axs[1].set_ylabel('Error (m)')
    axs[1].legend()
    axs[2].plot(frame_ids, errs[:, 2], label='Yaw Error', color='red')
    axs[2].fill_between(frame_ids,
                    -3 * stds[:, 2],
                    3 * stds[:, 2],
                    color='blue',
                    alpha=0.2,
                    label='3-Sigma Bound')
    axs[2].set_title('Yaw Errors with 3-Sigma Bounds')
    axs[2].set_xlabel('Frame ID')
    axs[2].set_ylabel('Error (deg)')
    axs[2].legend()
    plt.tight_layout()
    print("Saving error plots with 3-sigma bounds to:", loc_path)
    plt.savefig(osp.join(loc_path, 'loc_error_with_uncertainty.png'), dpi=300, bbox_inches="tight")

if show:
    plt.show()

plt.close('all')