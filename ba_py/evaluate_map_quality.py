import struct
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import sys
from map.voxel_map import Map
import argparse
from pyboreas.utils.utils import (
    get_inverse_tf,
    rotToRollPitchYaw
)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import utils

def compute_rmse(errors):
    return np.sqrt(np.mean(np.array(errors)**2))
    # return np.mean(np.array(errors))


min_travel_dist_gap = 300.0  # meters
min_abs_dist = 25.0 # meters


drba_output_dir = "/home/dl/Documents/phd/dev/dr_ba/output/aaa_paper_results/ba/skyway"
pogo_output_dir = "/home/dl/Documents/phd/dev/dr_ba/output/"
tbv_output_dir = "/home/dl/Documents/phd/dev/dr_ba/output/output_tbv"

# RSS data
sequences = [
    # 'boreas-2024-12-03-12-54', # Glenshield
    # 'boreas-2025-01-08-10-59', # Glenshield
    # 'boreas-2025-01-08-11-22', # Glenshield
    # 'boreas-2025-01-08-12-28', # Glenshield

    # 'boreas-2024-12-05-14-12', # Industrial
    # 'boreas-2024-12-23-16-27', # Industrial
    # 'boreas-2024-12-23-16-44', # Industrial
    # 'boreas-2024-12-23-17-01', # Industrial

    'boreas-2024-12-04-11-45', # Skyway
    'boreas-2024-12-04-11-56', # Skyway
    'boreas-2024-12-04-12-08', # Skyway
    'boreas-2024-12-04-12-19', # Skyway

    # 'boreas-2025-07-18-10-33', # Forest
    # 'boreas-2025-07-18-11-00', # Forest
    # 'boreas-2025-07-18-11-25', # Forest
    # 'boreas-2025-07-18-11-53', # Forest

    # 'boreas-2025-07-18-14-55', # Farm
    # 'boreas-2025-07-18-15-12', # Farm
    # 'boreas-2025-07-18-15-30', # Farm
    # 'boreas-2025-07-18-15-48', # Farm
]

dro_rmse_trans_errors = []
dro_rmse_rot_errors = []
tbv_rmse_trans_errors = []
tbv_rmse_rot_errors = []
pogo_rmse_trans_errors = []
pogo_rmse_rot_errors = []
ba_rmse_trans_errors = []
ba_rmse_rot_errors = []
for seq_id in sequences:
    # Reset per-seq counters
    num_matched = 0
    num_entries = len(pogo_rmse_trans_errors)

    # Load in all poses
    all_gt_poses, gt_times = utils.getGTRadarPosesAndTimes(seq_id)
    all_dro_poses, dro_times = utils.getDroPosesAndTimes(seq_id)
    # all_tbv_poses, tbv_times = utils.readTBV2DTraj(osp.join(tbv_output_dir, seq_id))
    all_pogo_poses, pogo_times = utils.getPogoPosesAndTimes(seq_id)
    if (not osp.exists(osp.join(drba_output_dir, seq_id, 'ba_traj.csv'))):
        print("No BA trajectory found for sequence {}, skipping...".format(seq_id))
        continue

    ba_poses, ba_times = utils.getPogoPosesAndTimes(seq_id, ouput_path=drba_output_dir, file_name='ba_traj.csv', delimiter=',')

    # Trim poses to ba map poses
    interp_gt_poses = utils.getInterpolatedTrajectory(all_gt_poses, gt_times, ba_times/1e6)
    # Pogo and DRO poses should have 1:1 timestamps present with ba_times, just load them in directly
    interp_pogo_poses = all_pogo_poses[np.isin(pogo_times, ba_times)]
    interp_dro_poses = all_dro_poses[np.isin(dro_times, ba_times)]

    assert interp_gt_poses.shape[0] == ba_poses.shape[0] == interp_pogo_poses.shape[0] == interp_dro_poses.shape[0], "Number of poses do not match between GT, Pogo, DRO and BA trajectories!"

    # Get the closest scan to scan_ref that is at least 300m apart in terms of travelled distance
    distances = [0]

    # Compute the cumulative distance along the trajectory
    for i in range(1, interp_gt_poses.shape[0]):
        delta = interp_gt_poses[i, 0:3, 3] - interp_gt_poses[i-1, 0:3, 3]
        dist = np.linalg.norm(delta)
        distances.append(distances[-1] + dist)
    distances = np.array(distances)

    for ref in range(interp_gt_poses.shape[0]):
        last_selected_range = min_abs_dist
        match_found = False
        for i in range(interp_gt_poses.shape[0]):
            if np.abs(distances[i] - distances[ref]) < min_travel_dist_gap:
                continue
            dist = np.linalg.norm(interp_gt_poses[i, 0:3, 3] - interp_gt_poses[ref, 0:3, 3])
            if dist < last_selected_range:
                last_selected_range = dist
                min_j = i
                match_found = True
        if match_found:
            num_matched += 1

            # Compute gt transform
            gt_s1_s2 = get_inverse_tf(interp_gt_poses[ref]) @ interp_gt_poses[min_j]

            # Compute pogo error
            pogo_s1_s2 = get_inverse_tf(interp_pogo_poses[ref]) @ interp_pogo_poses[min_j]
            T_pogo = pogo_s1_s2 @ get_inverse_tf(gt_s1_s2)
            pogo_rmse_trans_errors.append(np.linalg.norm(T_pogo[0:2, 3]))
            r, p, y = rotToRollPitchYaw(T_pogo[0:3, 0:3])
            pogo_rmse_rot_errors.append(abs(y * 180.0 / np.pi))

            # Compute dro error
            dro_s1_s2 = get_inverse_tf(interp_dro_poses[ref]) @ interp_dro_poses[min_j]
            T_dro = dro_s1_s2 @ get_inverse_tf(gt_s1_s2)
            dro_rmse_trans_errors.append(np.linalg.norm(T_dro[0:2, 3]))
            r, p, y = rotToRollPitchYaw(T_dro[0:3, 0:3])
            dro_rmse_rot_errors.append(abs(y * 180.0 / np.pi))

            # Compute ba error
            ba_s1_s2 = get_inverse_tf(ba_poses[ref]) @ ba_poses[min_j]
            T_ba = ba_s1_s2 @ get_inverse_tf(gt_s1_s2)
            ba_rmse_trans_errors.append(np.linalg.norm(T_ba[0:2, 3]))
            r, p, y = rotToRollPitchYaw(T_ba[0:3, 0:3])
            ba_rmse_rot_errors.append(abs(y * 180.0 / np.pi))

    print("Sequence: {}, {} / {} matched pairs".format(seq_id, num_matched, interp_gt_poses.shape[0]))
    print("DRO RMSE translation error: {} m, rotation error: {} deg".format(compute_rmse(dro_rmse_trans_errors[num_entries:]), compute_rmse(dro_rmse_rot_errors[num_entries:])))
    print("Pogo RMSE translation error: {} m, rotation error: {} deg".format(compute_rmse(pogo_rmse_trans_errors[num_entries:]), compute_rmse(pogo_rmse_rot_errors[num_entries:])))
    print("BA RMSE translation error: {} m, rotation error: {} deg".format(compute_rmse(ba_rmse_trans_errors[num_entries:]), compute_rmse(ba_rmse_rot_errors[num_entries:])))

print("------------------------------")
print("Overall DRO RMSE translation error: {} m, rotation error: {} deg".format(compute_rmse(dro_rmse_trans_errors), compute_rmse(dro_rmse_rot_errors)))
print("Overall Pogo RMSE translation error: {} m, rotation error: {} deg".format(compute_rmse(pogo_rmse_trans_errors), compute_rmse(pogo_rmse_rot_errors)))
print("Overall BA RMSE translation error: {} m, rotation error: {} deg".format(compute_rmse(ba_rmse_trans_errors), compute_rmse(ba_rmse_rot_errors)))

fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.boxplot([dro_rmse_trans_errors, pogo_rmse_trans_errors, ba_rmse_trans_errors], labels=['DRO', 'Pogo', 'BA'])
plt.ylabel('Translation Error (m)')
plt.title('Translation Error Comparison')
plt.subplot(1, 2, 2)
plt.boxplot([dro_rmse_rot_errors, pogo_rmse_rot_errors, ba_rmse_rot_errors], labels=['DRO', 'Pogo', 'BA'])
plt.ylabel('Rotation Error (deg)')
plt.title('Rotation Error Comparison')
plt.tight_layout()
plt.show()