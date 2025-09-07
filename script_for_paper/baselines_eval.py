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
    'navtech_slam': 'output_navtech_slam',
    'navtech_slam_rss': 'output_navtech_slam_rss',
    'fast_lio': 'output_fast_lio',
    '2fast_2lamaa': 'output_2fast_2lamaa',
    'tbv_original': '/home/ced/Documents/data/boreas/for_tbv/original_train/TBV_Eval',
    'tbv_rss': '/home/ced/Documents/data/boreas/for_tbv/rss/TBV_Eval',
}

kSeqTypes = ['Glenshield', 'Skyway', 'Commercial', 'Original_train']


def main():
    errors = {}


    for method, path in kMethodsAndPath.items():
        # Get the list of files in output directory
        output_files = os.listdir(path)
        output_files.sort()
        errors[method] = {}
        for file in output_files:
            if "_errors.csv" in file:
                continue
            if 'tbv' in method:
                if file == 'boreas':
                    continue
                job_path = os.path.join(path, file, 'job_0')
                poses, times, seq_id = utils.readTBV2DTraj(job_path)
                pdf_base_path = "output_tbv"
            else:
                pdf_base_path = path
                if not file.endswith('.csv') and not file.endswith('.txt'):
                    continue
                seq_id = file.split('.')[0]

                # Read the trajectory
                if method == 'fast_lio':
                    if '_' in file:
                        continue
                    seq_type = utils.getSeqType(seq_id)
                    if seq_type not in kSeqTypes:
                        continue
                    poses, times = utils.readFastLio2DTraj(os.path.join(path, file), seq_id)
                elif method == '2fast_2lamaa':
                    poses, times = utils.read2Fast2Lamaa2DTraj(os.path.join(path, file), seq_id)
                elif 'navtech_slam' in method:
                    if not ('pgo' in seq_id):
                        continue
                    seq_id = seq_id.replace('_pgo', '')
                    if 'rss' in method:
                        seq_id = seq_id[:-4]
                    poses, times = utils.readNavtechSLAM2DTraj(os.path.join(path, file))
                else:
                    raise ValueError(f"Unknown method {method}")

            seq_type = utils.getSeqType(seq_id)
            if seq_type not in kSeqTypes:
                continue
            print(f"Processing sequence {seq_id} for method {method}...")


            gt_poses, gt_times = utils.getGTRadarPosesAndTimes(seq_id)
            gt_poses_interp = utils.getInterpolatedTrajectory(gt_poses, gt_times, times)

            # Compute the absolute trajectory errors
            ate = utils.get2dATE(gt_poses_interp, poses, True, est_colour='green', path=os.path.join(pdf_base_path, seq_id + '_traj.pdf'), gt_colour='red')
            epe = np.linalg.norm((np.linalg.inv(np.linalg.inv(gt_poses_interp[0,:,:]) @ gt_poses_interp[-1,:,:]) @ np.linalg.inv(poses[0,:,:]) @ poses[-1,:,:])[:2,3])
            print("ATE:", ate, "m", "\t\tEPE:", epe, "m")

            if seq_type not in errors[method]:
                errors[method][seq_type] = {}
            if seq_id in errors[method][seq_type]:
                print("Sequence ID already exists, take the minimum ATE")
                if ate < errors[method][seq_type][seq_id]['ATE']:
                    errors[method][seq_type][seq_id] = {'ATE': ate, 'EPE': epe}
            else:
                errors[method][seq_type][seq_id] = {'ATE': ate, 'EPE': epe}

            # Save ATE and EPE to a csv file
            df = pd.DataFrame.from_dict(errors[method][seq_type][seq_id], orient='index').T
            df.to_csv(os.path.join(pdf_base_path, seq_id + '_errors.csv'), index=False)


    # Show the errors per method and per sequence type
    for method, seq_errors in errors.items():
        print("\n\n============ Method:", method)
        for seq_type, seq_errors in seq_errors.items():
            mean_ate = np.mean(np.array([seq_errors[error]['ATE'] for error in seq_errors]))
            mean_epe = np.mean(np.array([seq_errors[error]['EPE'] for error in seq_errors]))
            print("\n----Sequence Type:", seq_type, " mean ATE:", mean_ate, "m", " mean EPE:", mean_epe, "m")
            for seq_id, values in seq_errors.items():
                print("    Sequence ID: ", seq_id, " ATE = ", values['ATE'], "m", " EPE = ", values['EPE'], "m")

if __name__ == "__main__":
    main()