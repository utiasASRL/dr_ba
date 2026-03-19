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
    base_path =  '/home/dl/Documents/phd/dev/dr_ba/output/ba_results/run_2'

voxel_path = osp.join(base_path, 'voxel_map.bin')

vox_map = Map(res=1.0)  # Resolution will get overwritten when loading
vox_map.load_from_binary(voxel_path)
print("Voxel map size:", vox_map.size())
vox_map.plot(show=True)
vox_map.plot_paper(show=True, plot_poses=True)

plt.show()
