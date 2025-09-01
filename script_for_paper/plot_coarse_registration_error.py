import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import utils
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import seaborn as sns

kTypeColors = { 
            "Commercial": "green",
            "Glenshield": "orange",
            "Skyway": "blue",
            }
kTypeLabels = {
            "Commercial": "Commercial",
            "Glenshield": "Suburbs",
            "Skyway": "Skyway",
            }

kInlierPosThr = 1.5
kInlierRotThr = 1.0

def main():
    # Get the folders in the output directory
    output_paths = os.listdir("output")

    errors_coarse = {}
    errors_fine = {}


    for seq_id in output_paths:
        if not seq_id.startswith('boreas-'):
            continue

        print(f"Processing sequence {seq_id}...")

        try:
            # Load the coarse registrations
            loops_coarse = pd.read_csv(os.path.join("output", seq_id, "coarse_registrations.csv"))
            loops_fine = pd.read_csv(os.path.join("output", seq_id, "fine_registrations.csv"))
        except:
            print(f"Skipping sequence {seq_id} due to missing coarse registrations.")
            continue

        # Get the GT radar poses
        gt_poses, gt_times = utils.getGTRadarPosesAndTimes(seq_id)

        seq_type =  utils.getSeqType(seq_id)


        temp_errors_coarse = getRegistrationErrors(loops_coarse, gt_poses, gt_times)
        temp_errors_fine = getRegistrationErrors(loops_fine, gt_poses, gt_times)
        


        if(seq_type not in errors_coarse):
            errors_coarse[seq_type] = []
            errors_fine[seq_type] = []
        errors_coarse[seq_type].append({seq_id: temp_errors_coarse})
        errors_fine[seq_type].append({seq_id: temp_errors_fine})



    # Sort the sequences per alphabetical order
    errors_coarse = {k: sorted(v, key=lambda x: list(x.keys())[0]) for k, v in errors_coarse.items()}
    errors_fine = {k: sorted(v, key=lambda x: list(x.keys())[0]) for k, v in errors_fine.items()}


    # Get the number of matches and inliers per type
    for seq_type in errors_coarse.keys():
        inlier_matches_coarse = []
        for errors in errors_coarse[seq_type]:
            for seq_id, err in errors.items():
                for e in err:
                    if e[0] < kInlierPosThr and e[1] < kInlierRotThr:
                        inlier_matches_coarse.append(e)
        inlier_matches_fine = []
        for errors in errors_fine[seq_type]:
            for seq_id, err in errors.items():
                for e in err:
                    if e[0] < kInlierPosThr and e[1] < kInlierRotThr:
                        inlier_matches_fine.append(e)
        num_seq = len(errors_coarse[seq_type])
        num_inliers_coarse = len(inlier_matches_coarse)
        num_inliers_fine = len(inlier_matches_fine)
        num_coarse = 0
        for errors in errors_coarse[seq_type]:
            for seq_id, err in errors.items():
                num_coarse += len(err)
        num_fine = 0
        for errors in errors_fine[seq_type]:
            for seq_id, err in errors.items():
                num_fine += len(err)
        coarse_rmse_pos = np.sqrt(np.mean([e[0]**2 for e in inlier_matches_coarse]))
        fine_rmse_pos = np.sqrt(np.mean([e[0]**2 for e in inlier_matches_fine]))
        coarse_rmse_rot = np.sqrt(np.mean([e[1]**2 for e in inlier_matches_coarse]))
        fine_rmse_rot = np.sqrt(np.mean([e[1]**2 for e in inlier_matches_fine]))
        print("\n\n====== Sequence type", seq_type, "======")
        print("    Coarse:")
        print("        ", num_coarse / num_seq, "matches,", num_inliers_coarse / num_seq, "inliers ( ratio:", num_inliers_coarse / num_coarse, ")")
        print("        RMSE (trans):", coarse_rmse_pos)
        print("        RMSE (rot):", coarse_rmse_rot)
        print("    Fine:")
        print("        ", num_fine / num_seq, "matches,", num_inliers_fine / num_seq, "inliers ( ratio:", num_inliers_fine / num_fine, ")")
        print("        RMSE (trans):", fine_rmse_pos)
        print("        RMSE (rot):", fine_rmse_rot)

    # Plot the errors for each type
    fig, ax = plt.subplots(2, 2, figsize=(6, 4))
    # Similar as before but with seaborn's stripplot
    all_coarse_pos = []
    all_coarse_rot = []
    all_fine_pos = []
    all_fine_rot = []
    for seq_type in ['Glenshield', 'Commercial', 'Skyway']:
        for errors in errors_coarse[seq_type]:
            for seq_id, err in errors.items():
                for e in err:
                    all_coarse_pos.append({
                        'seq_type': seq_type,
                        'seq_id': seq_id,
                        'trans_err': e[0],
                        'rot_err': e[1]
                    })
                    all_coarse_rot.append({
                        'seq_type': seq_type,
                        'seq_id': seq_id,
                        'rot_err': e[1]
                    })
        for errors in errors_fine[seq_type]:
            for seq_id, err in errors.items():
                for e in err:
                    all_fine_pos.append({
                        'seq_type': seq_type,
                        'seq_id': seq_id,
                        'trans_err': e[0],
                        'rot_err': e[1]
                    })
                    all_fine_rot.append({
                        'seq_type': seq_type,
                        'seq_id': seq_id,
                        'rot_err': e[1]
                    })

    # --- Plot with seaborn stripplot using hue for color ---
    sns.stripplot(data=pd.DataFrame(all_coarse_pos), x='seq_id', y='trans_err', hue='seq_type',
                  palette=kTypeColors, dodge=True, ax=ax[0, 0], alpha=0.5)
    sns.stripplot(data=pd.DataFrame(all_coarse_rot), x='seq_id', y='rot_err', hue='seq_type',
                  palette=kTypeColors, dodge=True, ax=ax[0, 1], alpha=0.5)
    sns.stripplot(data=pd.DataFrame(all_fine_pos), x='seq_id', y='trans_err', hue='seq_type',
                  palette=kTypeColors, dodge=True, ax=ax[1, 0], alpha=0.5)
    sns.stripplot(data=pd.DataFrame(all_fine_rot), x='seq_id', y='rot_err', hue='seq_type',
                  palette=kTypeColors, dodge=True, ax=ax[1, 1], alpha=0.5)
    seq_types = ['Glenshield', 'Commercial', 'Skyway']
    # Replace with the labels
    seqs_per_type = [len(errors_coarse[stype]) for stype in seq_types]
    seq_types = [kTypeLabels[stype] for stype in seq_types]

    # Compute the center position for each group
    group_centers = np.cumsum([0] + seqs_per_type)[:-1] + np.array(seqs_per_type) / 2 - 0.5

    # Set the x-ticks at the group centers and label them with the sequence type
    ax[0, 0].set_xticks(group_centers)
    ax[0, 0].set_xticklabels(seq_types, fontsize=9)
    ax[0, 1].set_xticks(group_centers)
    ax[0, 1].set_xticklabels(seq_types, fontsize=9)
    ax[1, 0].set_xticks(group_centers)
    ax[1, 0].set_xticklabels(seq_types, fontsize=9)
    ax[1, 1].set_xticks(group_centers)
    ax[1, 1].set_xticklabels(seq_types, fontsize=9)

    # Remove the legends
    ax[0, 0].legend_.remove()
    ax[0, 1].legend_.remove()
    ax[1, 0].legend_.remove()
    ax[1, 1].legend_.remove()

    ax[0, 0].set_title('Coarse reg. - Pos. error', {'fontweight': 'bold'})
    ax[0, 1].set_title('Coarse reg. - Rot. error', {'fontweight': 'bold'})
    ax[1, 0].set_title('Fine reg. - Pos. error', {'fontweight': 'bold'})
    ax[1, 1].set_title('Fine reg. - Rot. error', {'fontweight': 'bold'})
    ax[0, 0].set_ylabel('Pos. Error [m]')
    ax[0, 1].set_ylabel('Rot. Error [deg]')
    ax[1, 0].set_ylabel('Pos. Error [m]')
    ax[1, 1].set_ylabel('Rot. Error [deg]')
    ax[0, 0].set_xlabel('')
    ax[0, 1].set_xlabel('')
    ax[1, 0].set_xlabel('')
    ax[1, 1].set_xlabel('')
    plt.tight_layout()
    plt.savefig('registration_errors_per_sequence.pdf')
    plt.show()


def getRegistrationErrors(loops, gt_poses, gt_times):
    errors = []
    for loop in loops.itertuples():
        time_i = utils.nameToTime(loop.scan_i_name)
        time_j = utils.nameToTime(loop.scan_j_name)

        pose_i = utils.getInterpolatedPose(gt_poses, gt_times, time_i)
        pose_j = utils.getInterpolatedPose(gt_poses, gt_times, time_j)

        # Compute the relative pose
        rel_pose_gt = np.linalg.inv(pose_i) @ pose_j
        xy = np.array([loop.x, loop.y])
        theta = loop.theta
        rel_pose_est = utils.XYThetaToPose(xy, theta)

        # Compute the error
        rel_pose_err = np.linalg.inv(rel_pose_est) @ rel_pose_gt
        trans_err = np.linalg.norm(rel_pose_err[:2, 3])
        rot_err = np.linalg.norm(R.from_matrix(rel_pose_err[:3, :3]).as_rotvec())*180.0/np.pi  # Convert to degrees
        errors.append((trans_err, rot_err))
    return errors


if __name__ == "__main__":
    main()