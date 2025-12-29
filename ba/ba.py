import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from utils import utils
from pylgmath import se3op
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from sksparse.cholmod import cholesky

from ba.scans.loader import ScanLoader
from ba.scans.point_scan import PointScan
from ba.scans.local_map_scan import LocalMapScan
from ba.map.voxel_map import Map

kSeqId = 'boreas-2024-12-03-12-54'

def plot_cost_history(cost_history, path=None):
    plt.figure()
    plt.plot(cost_history, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost History')
    plt.grid()
    if path is not None:
        plt.savefig(path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()

def plot_pose_errors(trans_errors, rot_errors, path=None):
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Translational Error (m)', color=color)
    ax1.plot(trans_errors, marker='o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid()
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Rotational Error (deg)', color=color)  # we already handled the x-label with ax1
    ax2.plot(rot_errors, marker='o', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    if path is not None:
        plt.savefig(path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()

def main(seq_id):
    # Map parameters
    map_res = 0.5  # meters

    # Raw measurement parameters
    max_dist = 80.0  # meters

    # Downsample control
    num_init_iter = 30
    init_downsample = 0.1
    refine_downsample = 1.0

    # Distance-based keyframing
    max_translation = 15.0  # meters
    max_rotation = np.deg2rad(30.0)  # radians
    max_sample = 2

    # Init error parameters
    translation_std = 1.0  # meters
    rotation_std = np.deg2rad(5.0)  # radians

    # Optimization parameters
    max_iter = 200
    tol = 1e-3

    # Input type
    input_type = 'local_map' # 'scan' or 'local_map'
    img_res = 0.1  # meters per pixel

    # Weights
    prior_map_std = 1.0
    measurement_std = 1.0

    # Get the list of npy files in output/<seq_id>/scans
    if input_type == 'local_map':
        scan_path = f'output/{seq_id}/local_maps/'
    else:
        scan_path = f'output/{seq_id}/scans/'

    # Output paths
    voxel_img_output_path = f'output/{seq_id}/voxel_maps/'
    os.makedirs(voxel_img_output_path, exist_ok=True)
    # Clear out old images
    for f in os.listdir(voxel_img_output_path):
        os.remove(osp.join(voxel_img_output_path, f))

    # Load in gt
    all_gt_poses, gt_times = utils.getGTRadarPosesAndTimes(kSeqId)
    print("Loaded GT poses:", len(all_gt_poses))

    # Form map
    vox_map = Map(res=map_res)

    # Form scan loader
    scan_loader = ScanLoader(seq_id)

    # Initialize looping through trajectory
    frame_0_pose = None
    prev_keyframe_pose = None

    # Load in all data once
    print("Loading in scans...")
    num_scans = len(os.listdir(scan_path))
    num_frames = 0
    gt_poses = {}
    for idx, scan in enumerate(sorted(os.listdir(scan_path))):
        if num_frames >= max_sample:
            break

        # Load in scan pose
        timestamp_scan = int(scan[:-4])/1e6  # convert to seconds
        interp_pose = utils.getInterpolatedPose(all_gt_poses, gt_times, timestamp_scan)

        if prev_keyframe_pose is not None:
            # Compute change since prev keyframe
            delta_pose = np.linalg.inv(interp_pose) @ prev_keyframe_pose
            delta_pose_vec = se3op.tran2vec(delta_pose)
            translation_mag = np.linalg.norm(delta_pose_vec[:2])
            rotation_mag = np.abs(delta_pose_vec[5])  # yaw change

            if translation_mag < max_translation and rotation_mag < max_rotation:
                continue  # skip this frame

        print(f'Processing frame {idx} / {num_scans}')
        num_frames += 1
        prev_keyframe_pose = interp_pose

        if frame_0_pose is None:
            frame_0_pose = interp_pose
        # Transform scan to frame_0
        rel_pose = np.linalg.inv(frame_0_pose) @ interp_pose
        gt_poses[idx] = rel_pose

        # Save initial guess for pose
        if idx != 0:
            vec_noise = np.zeros((6,1))
            vec_noise[0:2] = np.random.uniform(-translation_std, translation_std, (2,1))  # 0.5 m std dev
            vec_noise[5] = np.random.uniform(-rotation_std, rotation_std, (1,1))  # 5 deg std dev
            # vec_noise[5] = np.deg2rad(-5.0)
            # vec_noise[0] = 0.2
            # vec_noise[1] = 0.2
            noise_T = se3op.vec2tran(vec_noise)
            rel_pose = rel_pose @ noise_T

        # Populate voxels based on scan
        if input_type == 'local_map':
            scan_data = plt.imread(osp.join(scan_path, scan))
            scan = LocalMapScan(rel_pose, scan_data, img_res, idx)
        else:
            scan_data = np.load(osp.join(scan_path, scan))
            scan = PointScan(rel_pose, scan_data, idx)

        scan_loader.add_scan(scan)
        vox_map.init_map(rel_pose, max_dist)

    # Set up optimization
    print("Starting optimization...")

    # Form pose state order
    sorted_pose_keys = sorted(scan_loader.scan_ids)
    pose_key_to_idx = {key: i for i, key in enumerate(sorted_pose_keys)}

    # Set up constant covariances
    Q_meas_sqrt = 1/measurement_std
    xy1  = np.empty(4)

    prev_cost = np.inf
    cost_history = []
    num_states = len(sorted_pose_keys)
    print("Number of states:", num_states)
    print("Initial poses:\n")
    trans_errors = []
    rot_errors = []
    avg_rotational_err = 0.0
    avg_translational_err = 0.0
    for scan_id in sorted_pose_keys:
        s = pose_key_to_idx[scan_id]
        print(f"State {s}:\n", scan_loader.get_scan(scan_id).pose)
        # Compute initial error
        gt_pose = gt_poses[scan_id]
        pose_err = se3op.tran2vec(np.linalg.inv(scan_loader.get_scan(scan_id).pose) @ gt_pose).flatten()
        # print("Initial pose error (x,y,yaw):", pose_err[0], pose_err[1], np.rad2deg(pose_err[5]))
        avg_translational_err += np.linalg.norm(pose_err[0:2])
        avg_rotational_err += np.abs(pose_err[5])
    avg_translational_err /= (num_states - 1)
    avg_rotational_err /= (num_states - 1)
    trans_errors.append(avg_translational_err)
    rot_errors.append(np.rad2deg(avg_rotational_err))
    print("Average initial translational error (m): {:.4f}, Average initial rotational error (deg): {:.2f}".format(
        avg_translational_err, np.rad2deg(avg_rotational_err)
    ))

    scan_by_id = {scan.id: scan for scan in scan_loader.scans}
    for iter in range(max_iter):
        print(f"\n--- Iteration {iter} ---")
        if iter < num_init_iter:
            downsample_factor = init_downsample
        else:
            downsample_factor = refine_downsample

        # Form map state order
        sorted_map_keys = vox_map.get_sorted_voxels(downsample_factor=downsample_factor)
        map_key_to_idx = {key: i for i, key in enumerate(sorted_map_keys)}
        num_voxels = len(sorted_map_keys)
        print("Number of voxels:", num_voxels)

        # Construct necessary matrices. We're trying to avoid ever forming a matrix
        # with rows/columns corresponding to num measurements
        H_TT = np.zeros((num_states * 3, num_states * 3))
        H_TM = np.zeros((num_states * 3, num_voxels))
        H_MM = lil_matrix((num_voxels, num_voxels))

        # Measurement pre-multipliers
        J_T_B = np.zeros((num_states * 3, 1)) # J_T^T @ B
        J_M_B = np.zeros((num_voxels, 1)) # J_M^T @ B

        cost = 0.0

        print("Populating matrices...")
        # Add prior to all map voxels to be zero intensity
        for vox_key in sorted_map_keys:
            v_idx = map_key_to_idx[vox_key]
            vox_int = vox_map.voxels[vox_key].intensity
            vox_x = vox_key[0] * map_res
            vox_y = vox_key[1] * map_res

            vox_covered = False
            for scan_id in sorted_pose_keys:
                # Load in values
                scan = scan_by_id[scan_id]

                # Interpolate in scan
                I_i, d_I_d_T = scan.interpolate(vox_x, vox_y, jac=True)  # just to register coverage
                if np.isnan(I_i):
                    continue
                vox_covered = True

                # Weight intensity and Jacobian
                I_i_weighted = Q_meas_sqrt * I_i

                # Minus sign since error is vox_int - I_i
                d_e_d_T = - Q_meas_sqrt * d_I_d_T
                
                # Compute Jacobian w.r.t map voxel
                d_e_d_M = Q_meas_sqrt

                # Populate map matrices
                H_MM[v_idx, v_idx] += d_e_d_M * d_e_d_M
                J_M_B[v_idx, 0] += d_e_d_M * I_i_weighted

                # Populate state matrices
                s_idx = pose_key_to_idx[scan_id] - 1  # first pose is fixed
                if s_idx >= 0:  
                    H_TT[s_idx*3:(s_idx+1)*3, s_idx*3:(s_idx+1)*3] += d_e_d_T.T @ d_e_d_T
                    H_TM[s_idx*3:(s_idx+1)*3, v_idx:v_idx+1] += d_e_d_T.T * d_e_d_M
                    J_T_B[s_idx*3:(s_idx+1)*3, 0:1] += d_e_d_T.T * I_i_weighted

                # Compute error for cost
                cost += (vox_int - I_i)**2

            if not vox_covered:
                # Add prior for this voxel
                H_MM[v_idx, v_idx] += (1/prior_map_std**2)
                cost += vox_int**2 / (prior_map_std**2)

        print("Solving for state updates...")
        # One-time factorization
        H_MM = H_MM.tocsc()
        H_MM_factor = cholesky(H_MM)

        lhs = H_TT - H_TM @ H_MM_factor(H_TM.T) + 1e-8 * np.eye(num_states * 3)
        rhs = - H_TM @ H_MM_factor(J_M_B) + J_T_B
        # Scale alpha with iteration
        alpha = max(1.0 / (1.0 + iter), 1.0)
        # alpha = 1.0


        del_x = alpha * np.linalg.solve(lhs, rhs)
        M = H_MM_factor(J_M_B)

        # Update states
        print("Updating states...")
        avg_translational_err = 0.0
        avg_rotational_err = 0.0
        for scan_id in sorted_pose_keys:
            s_idx = pose_key_to_idx[scan_id] - 1  # first pose is fixed
            if s_idx < 0:
                continue
            delta_vec = del_x[s_idx*3:(s_idx+1)*3, 0:1]
            delta_vec_se3 = np.zeros((6,1))
            delta_vec_se3[0:2] = delta_vec[0:2]
            delta_vec_se3[5] = delta_vec[2]
            # print("Delta vec for state {}: {}".format(s_idx+1, delta_vec_se3.T))
            delta_T = se3op.vec2tran(-delta_vec_se3)
            new_scan_pose = delta_T @ scan_loader.get_scan(scan_id).pose
            scan_loader.get_scan(scan_id).update_pose(new_scan_pose)
            # print("Updated pose for state {}:\n{}".format(s_idx+1, new_scan_pose))
            gt_pose = gt_poses[scan_id]
            pose_err = se3op.tran2vec(np.linalg.inv(new_scan_pose) @ gt_pose).flatten()
            # print("Pose error for state {}: x,y,yaw: {:.4f}, {:.4f}, {:.2f} deg".format(
            #     s_idx+1, pose_err[0], pose_err[1], np.rad2deg(pose_err[5])
            # ))
            avg_translational_err += np.linalg.norm(pose_err[0:2])
            avg_rotational_err += np.abs(pose_err[5])
        
        avg_translational_err /= (num_states - 1)
        avg_rotational_err /= (num_states - 1)
        trans_errors.append(avg_translational_err)
        rot_errors.append(np.rad2deg(avg_rotational_err))
        print("Average translational error (m): {:.4f}, Average rotational error (deg): {:.2f}".format(
            avg_translational_err, np.rad2deg(avg_rotational_err)
        ))
        plot_pose_errors(trans_errors, rot_errors, path=osp.join(voxel_img_output_path, 'pose_errors.png'))

        # Update map
        for vox_key in sorted_map_keys:
            vox = vox_map.voxels[vox_key]
            v_idx = map_key_to_idx[vox_key]
            new_int = M[v_idx, 0]
            vox.update_intensity(new_int)

        # vox_map.plot()
        # Convergence measured by change in state estimates
        print("Cost:", cost)

        # Check convergence
        if np.abs(prev_cost - cost) < tol:
            print("Converged.")
            break
        if iter == max_iter - 1:
            print("Reached max iterations.")
        prev_cost = cost
        cost_history.append(cost)

        print("Updated voxel map:")
        vox_map.plot(save_path=osp.join(voxel_img_output_path, f'voxel_map_iter_{iter}.png'), iter=iter)
        # Plot cost history
        plot_cost_history(cost_history, path=osp.join(voxel_img_output_path, 'cost_history.png'))

    print("Final states:")
    for scan_id in sorted_pose_keys:
        s_idx = pose_key_to_idx[scan_id]
        # print(f"State {s_idx}:\n", scan_loader.get_scan(scan_id).pose)
        gt_pose = gt_poses[scan_id]
        pose_err = se3op.tran2vec(np.linalg.inv(scan_loader.get_scan(scan_id).pose) @ gt_pose)
        print("Final pose error (x,y,yaw):", pose_err[0], pose_err[1], np.rad2deg(pose_err[5]))
    
    vox_map.plot()

if __name__ == '__main__':
    main(kSeqId)
