import struct
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import sys
from map.voxel_map import Map
import argparse
import os
import subprocess
import time


parser = argparse.ArgumentParser(description="Visualize localization results")
parser.add_argument(
    "--loc_path",
    type=str,
    required=True,
    help="Path to localization results"
)


parser.add_argument(
    "--show",
    action="store_true",
    help="Show the plot (flag)"
)

args = parser.parse_args()

loc_path = args.loc_path
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


out_dir = osp.join(loc_path, 'loc_video_frames')

# # If out_dir already has content, get confirmation I want to overwrite
# if osp.exists(out_dir) and len(os.listdir(out_dir)) > 0:
#     response = input(f"Output directory {out_dir} already exists and is not empty. Overwrite? (y/n): ")
#     if response.lower() != 'y':
#         print("Exiting without overwriting.")
#         sys.exit(0)

# Clear out_dir if it exists
if osp.exists(out_dir):
    for filename in os.listdir(out_dir):
        file_path = osp.join(out_dir, filename)
        if osp.isfile(file_path):
            os.remove(file_path)


vox_map.render_localization_video_frames(
        loc_results,
        out_dir,
        zoom_range=150.0,          # meters (half-width/height of zoom window)
        n_full=10,                # frames showing full map
        n_zoom=30,                # frames for zoom-in animation
        trail_N=10,             # trailing path length
        veh_len=6.0,            # triangle length (m)
        veh_wid=3.0,            # triangle width (m)
        dpi=100,
        scan_path='/home/dl/Documents/phd/dev/dr_ba/output/boreas-2025-07-18-15-12/local_maps/'
        # scan_path='/home/dl/Documents/phd/dev/dr_ba/output/boreas-2025-07-18-11-00/local_maps/'
        # scan_path='/home/dl/Documents/phd/dev/dr_ba/output/boreas-2025-01-08-10-59/local_maps/'
        # scan_path='/home/dl/Documents/phd/dev/dr_ba/output/boreas-2024-12-23-16-44/local_maps/'
        # scan_path='/home/dl/Documents/phd/dev/dr_ba/output/boreas-2024-12-04-11-56/blurred/local_maps/0_30pct_0_50minint'
    )


plt.close('all')

time.sleep(0.2)
video_path = osp.join(out_dir, 'loc_video.mp4')

# Create video using ffmpeg
print("Creating video using ffmpeg...")
cmd = [
    "ffmpeg",
    "-framerate", "30",
    "-i", out_dir + "/frame_%05d.png",
    "-pix_fmt", "yuv420p",
    video_path
]

subprocess.run(cmd, check=True)

# Open video file
if show:
    if sys.platform == "win32":
        os.startfile(video_path)
    elif sys.platform == "darwin":
        subprocess.run(["open", video_path])
    else:
        subprocess.run(["xdg-open", video_path])