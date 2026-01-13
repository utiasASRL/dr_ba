import struct
import matplotlib.pyplot as plt
import numpy as np

path =  '/home/dl/Documents/phd/dev/dr_ba/output/run_2/voxels.bin'
with open(path, "rb") as f:
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
plt.show()
