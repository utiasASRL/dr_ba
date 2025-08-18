import os
import yaml
import numpy as np
import pandas as pd
from pyboreas.utils import odometry
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import matplotlib.pyplot as plt



def getOutputDataDir():
    # Fetch the sequence ID from the DRO config file
    with open(os.path.join("dro", "config.yaml"), 'r') as f:
        opts = yaml.safe_load(f)
    if opts['data']['multi_sequence']:
        raise ValueError("This script is not designed for multi-sequence data.")
    data_dir = opts['data']['data_path']
    if data_dir.endswith('/'):
        data_dir = data_dir[:-1]
    sequence_id = data_dir.split('/')[-1]

    # Get the data path
    data_dir = os.path.join("output", sequence_id)
    return data_dir


def getDataDir():
    # Fetch the sequence ID from the DRO config file
    with open(os.path.join("dro", "config.yaml"), 'r') as f:
        opts = yaml.safe_load(f)
    if opts['data']['multi_sequence']:
        raise ValueError("This script is not designed for multi-sequence data.")
    data_dir = opts['data']['data_path']
    if data_dir.endswith('/'):
        data_dir = data_dir[:-1]
    # Get the sequence ID
    sequence_id = data_dir.split('/')[:-1]
    # Combine to get the full data path
    data_dir = '/'.join(sequence_id)
    return data_dir



def affineToPoseAndScale(affine_matrix, pix_res, img_shape):
    # Transform to convert the opencv frame to the local map frame
    T_cv_local_map = np.array([[0, 1, 0, pix_res*img_shape[1]/2],
                               [-1, 0, 0, pix_res*img_shape[0]/2],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])
    T_local_map_cv = np.linalg.inv(T_cv_local_map)

    # Get the scale from the affine matrix
    scale = np.linalg.norm(affine_matrix[0, :2])
    # Get the rotation from the affine matrix
    rotation = np.arctan2(affine_matrix[1, 0], affine_matrix[0, 0])
    pose = np.eye(4)
    pose[0, 0] = np.cos(rotation)
    pose[0, 1] = -np.sin(rotation)
    pose[1, 0] = np.sin(rotation)
    pose[1, 1] = np.cos(rotation)
    pose[0, 3] = affine_matrix[0, 2] * pix_res
    pose[1, 3] = affine_matrix[1, 2] * pix_res

    # Convert the pose to the local map frame
    pose = T_local_map_cv @ pose @ T_cv_local_map
    return pose, scale



def getPixelResolution():
    # Fetch the pixel resolution from the DRO config file
    with open(os.path.join("dro", "config.yaml"), 'r') as f:
        opts = yaml.safe_load(f)
    return opts['direct']['local_map_res']

def getMaxLocalMapRange():
    # Fetch the max local map range from the DRO config file
    with open(os.path.join("dro", "config.yaml"), 'r') as f:
        opts = yaml.safe_load(f)
    return opts['direct']['max_local_map_range']


def poseToXYTheta(pose):
    # Convert the pose to (x, y, theta)
    x = pose[0, 3]
    y = pose[1, 3]
    theta = np.arctan2(pose[1, 0], pose[0, 0])
    return np.array([x, y]), theta


def XYThetaToPose(xy, theta):
    # Convert (x, y, theta) to pose
    pose = np.eye(4)
    pose[0, 0] = np.cos(theta)
    pose[0, 1] = -np.sin(theta)
    pose[1, 0] = np.sin(theta)
    pose[1, 1] = np.cos(theta)
    pose[0, 3] = xy[0]
    pose[1, 3] = xy[1]
    return pose



def getGTRadarPosesAndTimes(seq_id):
    # Get the data path
    data_path = os.path.join(getDataDir(), seq_id)

    # Get the gps poses
    gt_path = os.path.join(data_path, "applanix", "gps_post_process.csv")
    T_radar_applanix = np.loadtxt(os.path.join(data_path, "calib", "T_radar_lidar.txt")) @ np.linalg.inv(np.loadtxt(os.path.join(data_path, "calib", "T_applanix_lidar.txt")))
    gt_poses, gt_times = odometry.read_traj_file_gt(gt_path, T_radar_applanix, 2)
    gt_data_raw = pd.read_csv(gt_path)
    gt_times = gt_data_raw.iloc[:, 0].to_numpy()
    poses = []
    for pose in gt_poses:
        poses.append(np.linalg.inv(pose))
    gt_poses = np.array(poses)
    return gt_poses, gt_times

def getPogoPosesAndTimes(seq_id):
    # Get the results path
    data_path = os.path.join("output", seq_id, "pose_graph_traj.txt")
    if not os.path.exists(data_path):
        print(f"Skipping sequence {seq_id} due to missing results.")
        return None, None
    data_raw = pd.read_csv(data_path, delimiter=' ')
    times = data_raw.iloc[:, 0].to_numpy()
    poses = []
    for i in range(len(times)):
        pose = np.eye(4)
        pose[:2, 3] = data_raw.iloc[i, 1:3].to_numpy()
        pose[:2, :2] = np.array([[np.cos(data_raw.iloc[i, 3]), -np.sin(data_raw.iloc[i, 3])],
                                  [np.sin(data_raw.iloc[i, 3]), np.cos(data_raw.iloc[i, 3])]])
        poses.append(pose)
    poses = np.array(poses)
    return poses, times

def getDroPosesAndTimes(seq_id):
    # Get the results path
    data_path = os.path.join("output", seq_id, "odometry_2d", seq_id + ".txt")
    if not os.path.exists(data_path):
        print(f"Skipping sequence {seq_id} due to missing results.")
        return None, None
    data_raw = pd.read_csv(data_path, delimiter=' ')
    times = data_raw.iloc[:, 0].to_numpy()
    poses = []
    for i in range(len(times)):
        pose = np.eye(4)
        pose[:2, 3] = data_raw.iloc[i, 1:3].to_numpy()
        pose[:2, :2] = np.array([[np.cos(data_raw.iloc[i, 3]), -np.sin(data_raw.iloc[i, 3])],
                                  [np.sin(data_raw.iloc[i, 3]), np.cos(data_raw.iloc[i, 3])]])
        poses.append(pose)
    poses = np.array(poses)
    return poses, times

def getInterpolatedPose(poses, times, query_time):
    # Interpolate the pose at the query time
    if query_time < times[0] or query_time > times[-1]:
        print("Query time out of range:", query_time, times[0], times[-1])
        if query_time < times[0]:
            return poses[0]
        else:
            return poses[-1]
    for i in range(len(times)-1):
        if times[i] <= query_time <= times[i+1]:
            t = (query_time - times[i]) / (times[i+1] - times[i])
            pose1 = poses[i]
            pose2 = poses[i+1]
            # Interpolate the translation
            delta_pos = (1 - t) * pose1[:3, 3] + t * pose2[:3, 3]
            # Interpolate the rotation using slerp
            rotations = R.from_matrix([pose1[:3, :3], pose2[:3, :3]])
            r = Slerp([0, 1], rotations)(t)
            delta_rot = r.as_matrix()
            # Combine the translation and rotation into a pose
            delta_pose = np.eye(4)
            delta_pose[:3, :3] = delta_rot
            delta_pose[:3, 3] = delta_pos
            return delta_pose
    return poses[-1]


def nameToTime(name):
    # Extract the timestamp from the filename
    time_str = name.split('.')[0]
    return float(time_str)*1e-6

def get2dATE(gt_poses, est_poses, save_fig=False, est_colour='b', est_label='Estimated', gt_colour='orange', gt_label='Ground Truth', path=None):
    if gt_poses is None or est_poses is None:
        print("Invalid input poses.")
        return None

    if len(gt_poses) != len(est_poses):
        print("Ground truth and estimated poses must have the same length.")
        return None

    # Align the trajectories (find the R and t that best aligns the trajectories)
    gt_xy = gt_poses[:, :2, 3]
    est_xy = est_poses[:, :2, 3]
    gt_centroid = np.mean(gt_xy, axis=0)
    est_centroid = np.mean(est_xy, axis=0)

    # Center the trajectories
    gt_xy_centered = gt_xy - gt_centroid
    est_xy_centered = est_xy - est_centroid

    # Compute the covariance matrix
    H = gt_xy_centered.T @ est_xy_centered

    # Compute the SVD
    U, S, Vt = np.linalg.svd(H)

    # Compute the rotation
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        print("Reflection detected, correcting...")
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute the translation
    t = est_centroid - R @ gt_centroid

    # Apply the transformation to the est_xy
    est_xy_aligned = est_xy @ R - R.T@t

    ate = np.sqrt(np.mean(np.sum((gt_xy - est_xy_aligned)**2, axis=1)))

    #print("2D Absolute Trajectory Error (RMSE ATE):", ate)

    if save_fig:
        if path is None:
            path = "output/ate_2d_trajectory.pdf"
        plt.figure(figsize=(6, 6))
        plt.plot(est_xy_aligned[:, 0], est_xy_aligned[:, 1], label=est_label, color=est_colour, linewidth=0.5)
        plt.plot(gt_xy[:, 0], gt_xy[:, 1], label=gt_label, color=gt_colour, linewidth=0.5, linestyle='--')
        plt.legend(loc='upper left')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.savefig(path)
        plt.close()

    return ate

def getSeqType(seq_id):

    seqs = {
        'boreas-2024-12-03-10-24': 'Glenshield',
        'boreas-2024-12-03-12-54': 'Glenshield',
        'boreas-2025-01-08-10-59': 'Glenshield',
        'boreas-2025-01-08-11-22': 'Glenshield',
        'boreas-2025-01-08-11-44': 'Glenshield',
        'boreas-2025-01-08-12-28': 'Glenshield',

        'boreas-2024-12-03-13-13': 'Highway',
        'boreas-2024-12-03-13-34': 'Highway',
        'boreas-2024-12-10-12-07': 'Highway',
        'boreas-2024-12-10-12-24': 'Highway',
        'boreas-2024-12-10-12-38': 'Highway',
        'boreas-2024-12-10-12-56': 'Highway',

        'boreas-2024-12-04-14-28': 'Tunnel',
        'boreas-2024-12-04-14-34': 'Tunnel',
        'boreas-2024-12-04-14-38': 'Tunnel',
        'boreas-2024-12-04-14-44': 'Tunnel',
        'boreas-2024-12-04-14-50': 'Tunnel',
        'boreas-2024-12-04-14-59': 'Tunnel',

        'boreas-2024-12-04-11-45': 'Skyway',
        'boreas-2024-12-04-11-56': 'Skyway',
        'boreas-2024-12-04-12-08': 'Skyway',
        'boreas-2024-12-04-12-19': 'Skyway',
        'boreas-2024-12-04-12-34': 'Skyway',
    }
    if seq_id in seqs:
        return seqs[seq_id]
    else:
        return 'Unknown'
