import struct
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import sys
from map.voxel_map import Map
import os
import subprocess



# Check if any argument is provided for base path

if len(sys.argv) > 1:
    base_path = sys.argv[1]
else:
    print("Please provide the base path to the voxel map.")
    sys.exit(1)


# Find all files named voxel_map_*
# Get all files in directory
all_files = os.listdir(base_path)
voxel_files = [f for f in all_files if f.startswith('voxel_map_') and f.endswith('.bin')]
# Now sort by the number after voxel_map_
voxel_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

# Set up output
output_img_dir = osp.join(base_path, 'voxel_map_images')
os.makedirs(output_img_dir, exist_ok=True)

global_xmin = np.inf
global_xmax = -np.inf
global_ymin = np.inf
global_ymax = -np.inf

for vox_file in voxel_files:
    voxel_path = osp.join(base_path, vox_file)

    vox_map = Map(res=1.0)
    vox_map.load_from_binary(voxel_path)

    # voxel bounds
    keys = np.asarray(list(vox_map.voxels.keys()), dtype=np.int32)
    ix = keys[:, 0]
    iy = keys[:, 1]
    iy = -iy  # flip y for correct orientation

    global_xmin = min(global_xmin, ix.min() * vox_map.res)
    global_xmax = max(global_xmax, (ix.max() + 1) * vox_map.res)
    global_ymin = min(global_ymin, iy.min() * vox_map.res)
    global_ymax = max(global_ymax, (iy.max() + 1) * vox_map.res)

    # pose bounds (important!)
    if vox_map.poses:
        # Flip y for correct orientation
        xs = [p[1] for p in vox_map.poses]
        ys = [-p[2] for p in vox_map.poses]

        global_xmin = min(global_xmin, min(xs))
        global_xmax = max(global_xmax, max(xs))
        global_ymin = min(global_ymin, min(ys))
        global_ymax = max(global_ymax, max(ys))

global_extent = [global_xmin, global_xmax, global_ymin, global_ymax]
print("Global extent:", global_extent)


for vox_file in voxel_files:
    print("Processing:", vox_file)
    voxel_path = osp.join(base_path, vox_file)

    vox_map = Map(res=1.0)  # Resolution will get overwritten when loading
    vox_map.load_from_binary(voxel_path)
    # vox_map.plot(show=True)
    save_path = osp.join(output_img_dir, vox_file.replace('.bin', '.png'))
    vox_map.plot_paper(show=False, save_path=save_path, global_extent=global_extent)


plt.close('all')

video_path = osp.join(base_path, 'voxel_map_evolution.mp4')
# Create video using ffmpeg
ffmpeg_cmd = (
    f"ffmpeg -y -framerate 2 "
    f"-i {osp.join(output_img_dir, 'voxel_map_%d.png')} "
    f"-vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" "
    f"-c:v libx264 -pix_fmt yuv420p "
    f"{video_path}"
)
print("Creating video:", video_path)
os.system(ffmpeg_cmd)

# Open video file
if sys.platform == "win32":
    os.startfile(video_path)
elif sys.platform == "darwin":
    subprocess.run(["open", video_path])
else:
    subprocess.run(["xdg-open", video_path])
