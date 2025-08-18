import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import utils
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



def main():
    # Get the folders in the output directory
    output_paths = os.listdir("output")

    # Sort the output paths
    output_paths.sort()

    errors_per_type = {}


    for seq_id in output_paths:
        if not seq_id.startswith('boreas-'):
            continue

        print(f"Processing sequence {seq_id}...")

        # Read the loops
        loops = pd.read_csv(os.path.join("output", seq_id, "coarse_registrations.csv"))


        # Get the GT, odom and pogo radar poses
        odom_poses, odom_times = utils.getDroPosesAndTimes(seq_id)
        pogo_poses, pogo_times = utils.getPogoPosesAndTimes(seq_id)
        if(np.any(odom_times != pogo_times)):
            raise ValueError("Odom and Pogo times do not match!")

        # Convert times to seconds
        odom_times = odom_times * 1e-6
        pogo_times = pogo_times * 1e-6
        

        compute_gt = True
        gt_interp_path = os.path.join("output", seq_id, "gt_interpolated.npz")
        if(os.path.exists(gt_interp_path)):
            data = np.load(gt_interp_path)
            gt_poses_interp = data['poses']
            gt_times = data['times']
            if(np.max(np.abs(gt_times - odom_times))) > 1e-3:
                compute_gt = True
            else:
                compute_gt = False

        if compute_gt:
            gt_poses, gt_times = utils.getGTRadarPosesAndTimes(seq_id)
            gt_poses_interp = np.zeros(odom_poses.shape)
            for i in range(len(odom_times)):
                gt_poses_interp[i,:,:] = utils.getInterpolatedPose(gt_poses, gt_times, odom_times[i])
            inv_gt_first = np.linalg.inv(gt_poses_interp[0]).reshape(1,4,4)
            gt_poses_interp = inv_gt_first @ gt_poses_interp

            np.savez(gt_interp_path, poses=gt_poses_interp, times=odom_times)


        # Align the poses with identity
        inv_odom_first = np.linalg.inv(odom_poses[0]).reshape(1,4,4)
        inv_pogo_first = np.linalg.inv(pogo_poses[0]).reshape(1,4,4)

        odom_poses = inv_odom_first @ odom_poses
        pogo_poses = inv_pogo_first @ pogo_poses

        # Display the results trajectories
        plt.figure(figsize=(8,8))
        plt.plot(odom_poses[:,0,3], odom_poses[:,1,3], label='Odom', color='orange')
        plt.plot(pogo_poses[:,0,3], pogo_poses[:,1,3], 'b', label='Pogo')
        plt.plot(gt_poses_interp[:,0,3], gt_poses_interp[:,1,3], 'r--', label='GT')
        for loop in loops.itertuples():
            time_i = utils.nameToTime(loop.scan_i_name)
            time_j = utils.nameToTime(loop.scan_j_name)
            id_i = np.argmin(np.abs(odom_times - time_i))
            id_j = np.argmin(np.abs(odom_times - time_j))
            plt.plot([odom_poses[id_i, 0, 3], odom_poses[id_j, 0, 3]], [odom_poses[id_i, 1, 3], odom_poses[id_j, 1, 3]], 'g')

        plt.legend()
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.axis('equal')
        plt.title(f"Trajectories for sequence {seq_id}")
        plt.savefig(os.path.join("output", seq_id, "trajectories.pdf"))
        plt.show()

        
        



        


if __name__ == "__main__":
    main()