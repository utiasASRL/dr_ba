import pandas as pd
import numpy as np

# Load loop predictions and poses
loops = pd.read_csv('raplace_loops.csv')
poses = pd.read_csv('global_pose.csv', header=None)
poses.columns = ['timestamp', 'x', 'y']  # update if your file has more cols

# Extract just XY
pose_dict = {i: poses.iloc[i][['x', 'y']].values for i in range(len(poses))}

# Compute GT distances for each loop pair
dists = []
for i, row in loops.iterrows():
    i1 = int(row['scan_i'])
    i2 = int(row['scan_j'])
    p1 = pose_dict.get(i1)
    p2 = pose_dict.get(i2)
    if p1 is not None and p2 is not None:
        dist = np.linalg.norm(p1 - p2)
    else:
        dist = np.nan
    dists.append(dist)

loops['gt_dist'] = dists

# Add true/false loop label (e.g., if within 10 meters)
loops['is_true_loop'] = loops['gt_dist'] < 10.0

print(f"Total loop candidates: {len(loops)}")
print(f"Valid loops (<10m): {loops['is_true_loop'].sum()}")
print(f"Invalid loops (>=10m): {(~loops['is_true_loop']).sum()}")
print(f"Precision: {loops['is_true_loop'].mean():.2%}")
