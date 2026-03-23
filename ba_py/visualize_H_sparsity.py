import sys

import matplotlib.pyplot as plt
import numpy as np


# Check if any argument is provided for base path

if len(sys.argv) > 1:
    base_path = sys.argv[1]
else:
    print("Please provide the base path to the voxel map.")
    sys.exit(1)

with open(base_path + "/H.bin", "rb") as f:
    rows = np.fromfile(f, dtype=np.int32, count=1)[0]
    cols = np.fromfile(f, dtype=np.int32, count=1)[0]
    H = np.fromfile(f, dtype=np.float64).reshape(rows, cols)


fig, ax = plt.subplots(figsize=(12, 12))
plt.spy(H, markersize=0.5)
# Increase marker size
ax.tick_params(axis='both', which='major', labelsize=30)
ax.set_xlabel('Pose Index', fontsize=30)
ax.set_ylabel('Pose Index', fontsize=30)
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')

plt.grid(True)
plt.tight_layout()
plt.show()