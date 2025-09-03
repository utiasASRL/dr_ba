import matplotlib
matplotlib.use('TkAgg')
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import utils
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


kMethodsAndPath = {
    #'fast_lio': 'output_fast_lio',
    '2fast_2lamaa': 'output_2fast_2lamaa'
    }

kSeqTypes = ['Glenshield', 'Skyway']


def main():
    errors = {}


    for method, path in kMethodsAndPath.items():
        # Get the list of files in output directory
        output_files = os.listdir(path)
        output_files.sort()
        errors[method] = {}
        for file in output_files:
            if not file.endswith('.csv') and not file.endswith('.txt'):
                continue
            seq_id = file.split('.')[0]
            seq_type = utils.getSeqType(seq_id)
            if seq_type not in kSeqTypes:
                continue
            print(f"Processing sequence {seq_id} for method {method}...")


            gt_poses, gt_times = utils.getGTRadarPosesAndTimes(seq_id)

            # Read the trajectory
            if method == 'fast_lio':
                poses, times = utils.readFastLio2DTraj(os.path.join(path, file), seq_id)
            elif method == '2fast_2lamaa':
                poses, times = utils.read2Fast2Lamaa2DTraj(os.path.join(path, file), seq_id)

            gt_poses_interp = utils.getInterpolatedTrajectory(gt_poses, gt_times, times)

            # Compute the absolute trajectory errors
            ate = utils.get2dATE(gt_poses_interp, poses, True, est_colour='green', path=os.path.join(path, seq_id + '_traj.pdf'), gt_colour='red')
            print("2D Absolute Trajectory Error (RMSE ATE):", ate, "m")

            if seq_type not in errors[method]:
                errors[method][seq_type] = []
            errors[method][seq_type].append({seq_id: ate})


    # Show the errors per method and per sequence type
    for method, seq_errors in errors.items():
        print("\n\n============ Method:", method)
        for seq_type, seq_errors in seq_errors.items():
            print("\n----Sequence Type:", seq_type, " mean = ", np.mean([list(error.values())[0] for error in seq_errors]))
            for error in seq_errors:
                for seq_id, ate in error.items():
                    print("    Sequence ID: ", seq_id, " ATE = ", ate)

if __name__ == "__main__":
    main()