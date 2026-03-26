import matplotlib.pyplot as plt
import numpy as np


voxel_res = [10.0, 6.0, 3.0, 1.0, 0.5, 0.2, 0.1]
voxel_size = [100, 250, 934, 8189, 32505, 202405, 809706]

dr_ba_results = {
    'time': [1.49664, 1.48959, 1.49179, 1.54979, 1.69912, 2.84589, 7.35241],
    'ate': [0.170398, 0.0950314, 0.0413686, 0.0153036, 0.0158231, 0.0162953, 0.0164041],
    'nnz': [495, 495, 495, 495, 495, 495, 495]
}

combined_results = {
    'time': [1.50223, 1.53837, 1.65904, 4.06744, 11.0893, 74.8981, 413.569],
    'ate': [0.170398, 0.103507, 0.0269129, 0.0173049, 0.0175286, 0.017793, 0.0178204],
    'nnz': [3025, 7105, 25249, 220454, 875430, 5443360, 22116891]
}

# Drop the very first entry
voxel_size = voxel_size[1:]
dr_ba_results = {k: v[1:] for k, v in dr_ba_results.items()}
combined_results = {k: v[1:] for k, v in combined_results.items()}

fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)

axes[0].plot(voxel_size, dr_ba_results['time'], label='separable (Dr-BA)', marker='o')
axes[0].plot(voxel_size, combined_results['time'], label='combined', marker='o')
axes[0].set_ylabel('solve time (s)', fontsize=14)
axes[0].tick_params(labelsize=12)
axes[0].set_yscale('log')

axes[1].plot(voxel_size, dr_ba_results['nnz'], label='separable (Dr-BA)', marker='o')
axes[1].plot(voxel_size, combined_results['nnz'], label='combined', marker='o')
axes[1].set_ylabel('Hessian non-zeros', fontsize=14)
axes[1].tick_params(labelsize=12)
axes[1].set_yscale('log')

axes[2].plot(voxel_size, dr_ba_results['ate'], label='separable (Dr-BA)', marker='o')
axes[2].plot(voxel_size, combined_results['ate'], label='combined', marker='o')
axes[2].set_ylabel('ATE (m)', fontsize=14)
axes[2].set_xlabel('voxel count', fontsize=14)
axes[2].tick_params(labelsize=12, rotation=45)

axes[2].legend(fontsize=12)
plt.tight_layout()
plt.show()