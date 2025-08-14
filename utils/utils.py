import os
import yaml
import numpy as np


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


def affineToPoseAndScale(affine_matrix, pix_res, img_shape):
    # Get the scale from the affine matrix
    scale = np.linalg.norm(affine_matrix[0, :2])
    # Get the rotation from the affine matrix
    rotation = np.arctan2(affine_matrix[1, 0], affine_matrix[0, 0])
    pose = np.eye(3)
    pose[0, 0] = np.cos(rotation)
    pose[0, 1] = -np.sin(rotation)
    pose[1, 0] = np.sin(rotation)
    pose[1, 1] = np.cos(rotation)
    pose[0, 2] = affine_matrix[0, 2] * pix_res
    pose[1, 2] = affine_matrix[1, 2] * pix_res
    return pose, scale



def getPixelResolution():
    # Fetch the pixel resolution from the DRO config file
    with open(os.path.join("dro", "config.yaml"), 'r') as f:
        opts = yaml.safe_load(f)
    return opts['direct']['local_map_res']