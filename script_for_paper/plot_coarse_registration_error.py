import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

kTypeColors = { "Glenshield": "orange",
                "Skyway": "blue"}


def main():
    # Get the folders in the output directory
    output_paths = os.listdir("output")

    errors_per_type = {}
    errors_total = []


    for seq_id in output_paths:
        if not seq_id.startswith('boreas-'):
            continue

        print(f"Processing sequence {seq_id}...")

        try:
            # Load the coarse registrations
            loops = pd.read_csv(os.path.join("output", seq_id, "coarse_registrations.csv"))
        except:
            print(f"Skipping sequence {seq_id} due to missing coarse registrations.")
            continue

        # Get the GT radar poses
        gt_poses, gt_times = utils.getGTRadarPosesAndTimes(seq_id)

        errors_seq = []
        seq_type =  utils.getSeqType(seq_id)

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
            errors_seq.append((trans_err, rot_err))
            errors_total.append((trans_err, rot_err))


        if(seq_type not in errors_per_type):
            errors_per_type[seq_type] = []
        errors_per_type[seq_type].append({seq_id: errors_seq})

        

    errors_total = np.array(errors_total)
    print("Overall RMSE trans:", np.sqrt(np.mean(errors_total[:, 0]**2)), "m, rot:", np.sqrt(np.mean(errors_total[:, 1]**2)), "deg")

    # Get the maximum of sequences per type
    max_seq = max([len(errors) for errors in errors_per_type.values()])

    # Sort the sequences per alphabetical order
    errors_per_type = {k: v for k, v in sorted(errors_per_type.items(), key=lambda item: item[0])}

    # Plot the errors for each type
    fig, ax = plt.subplots(2, max_seq, figsize=(18, 8))
    zoomed_area = [-0.5, 5.5, -0.25, 2.75]
    all_area = [-0.5, np.max(errors_total[:, 0]) + 0.5, -0.25, 180.0]
    for i, (seq_type, errors) in enumerate(errors_per_type.items()):
        for j, seq_errors in enumerate(errors):
            seq_id = list(seq_errors.keys())[0]
            trans_errors = np.array([e[0] for e in seq_errors[seq_id]])
            rot_errors = np.array([e[1] for e in seq_errors[seq_id]])
            ax[0,j].scatter(trans_errors, rot_errors, label=seq_type + " n=" + str(len(trans_errors)), alpha=0.5, color=kTypeColors[seq_type], facecolors='none')
            ax[0,j].legend(loc='upper right')
            ax[0,j].set_ylim(all_area[2], all_area[3])
            ax[0,j].set_xlim(all_area[0], all_area[1])

            ax[1,j].scatter(trans_errors, rot_errors, label=seq_type + " n=" + str(len(rot_errors)), alpha=0.5, color=kTypeColors[seq_type], facecolors='none')
            ax[1,j].legend(loc='upper right')
            ax[1,j].set_xlabel("Translation Error [m]")
            ax[1,j].set_ylim(zoomed_area[2], zoomed_area[3])
            ax[1,j].set_xlim(zoomed_area[0], zoomed_area[1])
            ax[1,j].set_xlabel("Translation Error [m]")

    ax[0,0].set_ylabel("Rotation Error [deg]")
    ax[1,0].set_ylabel("Rotation Error [deg]")
    fig.suptitle("Coarse Registration Errors per Sequence Type")
    plt.tight_layout()
    plt.savefig("output/coarse_registration_errors.pdf")
    plt.show()


if __name__ == "__main__":
    main()