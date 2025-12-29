import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from utils import utils
from pylgmath import se3op
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from sksparse.cholmod import cholesky
from scipy.spatial import Delaunay
from math import nan


kSeqId = 'boreas-2024-12-03-12-54'

class ScanLoader:
    def __init__(self, seq_id):
        self.seq_id = seq_id
        self.scans = []
        self.scan_ids = []

    def add_scan(self, pose, scan, id):
        self.scans.append(self.Scan(pose, scan, id))
        self.scan_ids.append(id)

    def get_scan(self, id):
        idx = self.scan_ids.index(id)
        return self.scans[idx]

    class Scan:
        def __init__(self, pose, points, id):
            """
            points: (N,3) array [x, y, intensity]
            """
            self.pose = pose
            self.pose_2d = np.eye(3)
            self.pose_2d[0:2, 0:2] = pose[0:2, 0:2]
            self.pose_2d[0:2, 2] = pose[0:2, 3]
            self.id = id
            self.points = np.asarray(points)
            self.xy = self.points[:, :2]
            self.Iv = self.points[:, 2]
            self.tri = Delaunay(self.xy)

        def update_pose(self, new_pose):
            self.pose = new_pose
            self.pose_2d = np.eye(3)
            self.pose_2d[0:2, 0:2] = new_pose[0:2, 0:2]
            self.pose_2d[0:2, 2] = new_pose[0:2, 3]

        def point_to_scan_frame(self, x_map, y_map, jac=False):
            T_sm = np.linalg.inv(self.pose_2d)
            xy1 = np.array([[x_map], [y_map], [1.0]])
            p_s = T_sm @ xy1

            if not jac:
                return p_s[0, 0], p_s[1, 0]
            
            D = np.array([[1, 0, 0],
                          [0, 1, 0]])  # (2,3)
            d_p_s_d_T = D @ T_sm @ circle_dot_operator( p_s[0:2, 0:1] )  # (2,3)
            return p_s[0, 0], p_s[1, 0], d_p_s_d_T

        def check_covarage_at_point(self, x_map, y_map):
            x, y = self.point_to_scan_frame(x_map, y_map)
            p = np.array([[x, y]], dtype=float)
            s = self.tri.find_simplex(p)
            s = int(s[0])
            return s >= 0

        def interpolate(self, x_map, y_map, jac=False):
            x, y, d_pij_d_T = self.point_to_scan_frame(x_map, y_map, jac=True)
            p_ij = np.array([[x, y]], dtype=float)

            s = self.tri.find_simplex(p_ij)
            s = int(s[0])
            if s < 0:
                return (nan, nan) if jac else np.nan

            verts = self.tri.simplices[s]        # shape (3,)
            T = self.tri.transform[s]            # shape (3,2): [A; b] packed as (2x2, 2, ...)

            # SciPy stores affine transform: bary = A @ (p - b)
            A = T[:2, :]                         # (2,2)
            b = T[2, :]                          # (2,)

            v = (p_ij[0] - b)                       # (2,)
            w0_w1 = A @ v                        # (2,) -> [w0, w1]
            w0, w1 = w0_w1
            w2 = 1.0 - w0 - w1

            I0, I1, I2 = self.Iv[verts]
            I = w0 * I0 + w1 * I1 + w2 * I2

            if not jac:
                return I

            # Gradient is constant within the triangle:
            dI_dw = np.array([I0 - I2, I1 - I2]).reshape(1, -1)  # (1,2)
            dI_dpij = dI_dw @ A # (1,2) @ (2,2) -> (1,2)
            grad = dI_dpij @ d_pij_d_T  # (1,2) @ (2,3) -> (1,3)
            return I, grad

class Map:

    class Voxel:
        def __init__(self, intensity=0.0):
            self.intensity = intensity
            self.num_points = 1
            self.scan_ids = set()
        
        def add_scan_id(self, scan_id):
            self.scan_ids.add(scan_id)

        def update_intensity(self, new_intensity):
            self.intensity = new_intensity

    def __init__(self, res=0.2):
        self.res = res
        self.voxels = {} # key: (x_idx, y_idx), value: Voxel

    def index(self, x, y):
        x_idx = int(np.floor(x / self.res))
        y_idx = int(np.floor(y / self.res))
        return (x_idx, y_idx)

    def size(self):
        return len(self.voxels)

    def zero_out_map(self):
        for vox in self.voxels.values():
            vox.intensity = 0.0
            vox.num_points = 0

    def randomize_map(self):
        for vox in self.voxels.values():
            vox.intensity = np.random.uniform(0.0, 0.2)
            vox.num_points = 0

    def add_single_voxel(self, x, y, intensity):
        idx = (x, y)
        # idx = self.index(x, y)
        if idx not in self.voxels:
            self.voxels[idx] = self.Voxel(intensity)

    def add_map_coverage_at_point(self, x, y):
        ab = self.index(x, y)
        a1b = (ab[0] + 1, ab[1])
        ab1 = (ab[0], ab[1] + 1)
        a1b1 = (ab[0] + 1, ab[1] + 1)
        self.add_single_voxel(ab[0], ab[1], 0.0)
        self.add_single_voxel(a1b[0], a1b[1], 0.0)
        self.add_single_voxel(ab1[0], ab1[1], 0.0)
        self.add_single_voxel(a1b1[0], a1b1[1], 0.0)

    def check_map_coverage_at_point(self, x, y):
        ab = self.index(x, y)
        a1b = (ab[0] + 1, ab[1])
        ab1 = (ab[0], ab[1] + 1)
        a1b1 = (ab[0] + 1, ab[1] + 1)
        return (ab in self.voxels) and (a1b in self.voxels) and (ab1 in self.voxels) and (a1b1 in self.voxels)

    def get_sorted_voxels(self):
        return sorted(self.voxels.keys(), key=lambda k: (k[0], k[1]))

    def init_map(self, pose, max_dist):
        # Initialize empty voxels within max_dist of the pose
        x_center = pose[0, 3]
        y_center = pose[1, 3]
        a_center, b_center = self.index(x_center, y_center)
        num_voxels = int(np.ceil(max_dist / self.res))
        for da in range(-num_voxels, num_voxels +1):
            for db in range(-num_voxels, num_voxels +1):
                a = a_center + da
                b = b_center + db
                # x_vox = a * self.res
                # y_vox = b * self.res
                self.add_single_voxel(a, b, 0.0)

    def bilinear_interpolate(self, x, y, jac=False):
        # Get voxel indices
        (a, b) = self.index(x, y)

        # Get intensities
        int_ab = self.voxels.get((a, b)).intensity
        int_a1b = self.voxels.get((a + 1, b)).intensity
        int_ab1 = self.voxels.get((a, b + 1)).intensity
        int_a1b1 = self.voxels.get((a + 1, b + 1)).intensity

        # Get weights (measure offsets in meters within the voxel)
        x_tilde = x - a * self.res
        y_tilde = y - b * self.res
        w0 = (1 - x_tilde/self.res) * (1 - y_tilde/self.res)
        w1 = (1 - x_tilde/self.res) * (y_tilde/self.res)
        w2 = (x_tilde/self.res) * (1 - y_tilde/self.res)
        w3 = (x_tilde/self.res) * (y_tilde/self.res)

        # Bilinear interpolation
        int_xy = (w0 * int_ab + w1 * int_ab1 + w2 * int_a1b + w3 * int_a1b1)

        if jac:
            # Compute Jacobian
            d_B_d_x = (y_tilde - self.res)/self.res**2 * (int_ab - int_a1b) + y_tilde/self.res**2 * (int_a1b1 - int_ab1)
            d_B_d_y = (x_tilde - self.res)/self.res**2 * (int_ab - int_ab1) + x_tilde/self.res**2 * (int_a1b1 - int_a1b)
            d_B_d_M = np.array([w0, w1, w2, w3])
            return int_xy, np.array([d_B_d_x, d_B_d_y, 0.0]).reshape(1, -1), d_B_d_M.reshape(1, -1)

        return int_xy

    def get_intensity(self, x, y):
        idx = self.index(x, y)
        return self.voxels.get(idx, 0)

    def plot(self, save_path=None):
        if not self.voxels:
            return

        # --- convert sparse dict to dense raster ---
        keys = np.asarray(list(self.voxels.keys()), dtype=np.int32)
        vals = np.asarray([v.intensity for v in self.voxels.values()], dtype=np.float32)

        # --- warn on invalid intensity range ---
        invalid_low = np.any(vals < 0.0)
        invalid_high = np.any(vals > 1.0)
        if invalid_low or invalid_high:
            print(
                "[WARN] Voxel intensity values outside [0, 1] detected "
                f"(min={np.nanmin(vals):.3f}, max={np.nanmax(vals):.3f})"
            )

        ix = keys[:, 0]
        iy = keys[:, 1]

        ix_min, ix_max = ix.min(), ix.max()
        iy_min, iy_max = iy.min(), iy.max()

        H = ix_max - ix_min + 1
        W = iy_max - iy_min + 1

        img = np.full((H, W), np.nan, dtype=np.float32)
        img[ix - ix_min, iy - iy_min] = vals

        # --- plotting ---
        fig, ax = plt.subplots(facecolor="black")
        ax.set_facecolor("black")

        im = ax.imshow(
            img.T,
            origin="lower",
            cmap="viridis",
            vmin=0.0,          # clamp colormap
            vmax=1.0,          # clamp colormap
            extent=[
                ix_min * self.res,
                (ix_max + 1) * self.res,
                iy_min * self.res,
                (iy_max + 1) * self.res,
            ],
            interpolation="nearest",
        )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Intensity", color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.get_yticklabels(), color="white")

        ax.set_xlim(ix_min * self.res, (ix_max + 1) * self.res)
        ax.set_ylim(iy_min * self.res, (iy_max + 1) * self.res)

        ax.set_title("Voxel Map", color="white")
        ax.set_xlabel("X", color="white")
        ax.set_ylabel("Y", color="white")
        ax.tick_params(colors="white")
        ax.set_aspect("equal")

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="black")
        else:
            plt.show()
        plt.close(fig)


def circle_dot_operator(p):
    assert p.shape == (2,1)
    S = np.array([[0, -1],
                  [1, 0]])
    res = np.zeros((3,3))
    res[0:2,0:2] = np.eye(2)
    res[0:2,2:3] = S @ p

    return res

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

def main(seq_id):
    # Map parameters
    map_res = 0.2  # meters

    # Raw measurement parameters
    max_dist = 80.0  # meters
    downsample_factor = 10

    # Distance-based keyframing
    max_translation = 30.0  # meters
    max_rotation = np.deg2rad(30.0)  # radians
    max_sample = 3

    # Optimization parameters
    max_iter = 100
    tol = 1e-4

    # Weights
    prior_map_std = 0.5
    measurement_std = 1.0

    # Get the list of npy files in output/<seq_id>/scans
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
        if not scan.endswith('.npy'):
            continue
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
            vec_noise[0:2] = np.random.uniform(-0.2, 0.2, (2,1))  # 0.5 m std dev
            vec_noise[5] = np.random.uniform(-np.deg2rad(1.0), np.deg2rad(1.0), (1,1))  # 5 deg std dev
            # vec_noise[5] = np.deg2rad(-5.0)
            # vec_noise[0] = 0.2
            # vec_noise[1] = 0.2
            noise_T = se3op.vec2tran(vec_noise)
            rel_pose = rel_pose @ noise_T

        # Populate voxels based on scan
        scan_data = np.load(osp.join(scan_path, scan))
        scan_loader.add_scan(rel_pose, scan_data, idx)
        vox_map.init_map(rel_pose, max_dist)

    # Set up optimization
    print("Starting optimization...")

    # Form pose state order
    sorted_pose_keys = sorted(scan_loader.scan_ids)
    pose_key_to_idx = {key: i for i, key in enumerate(sorted_pose_keys)}
    # Form map state order
    sorted_map_keys = vox_map.get_sorted_voxels()
    map_key_to_idx = {key: i for i, key in enumerate(sorted_map_keys)}

    num_measurements = 0
    for vox_key in sorted_map_keys:
        vox = vox_map.voxels[vox_key]
        num_measurements += len(vox.scan_ids) 
    num_voxels = vox_map.size()

    print("Number of measurements:", num_measurements)
    print("Number of voxels:", num_voxels)

    # Set up constant covariances
    Q_meas_sqrt = 1/measurement_std
    xy1  = np.empty(4)

    prev_cost = np.inf
    cost_history = []
    num_states = len(sorted_pose_keys)
    print("Number of states:", num_states)
    print("Initial poses:\n")
    avg_rotational_err = 0.0
    avg_translational_err = 0.0
    for scan_id in sorted_pose_keys:
        s = pose_key_to_idx[scan_id]
        print(f"State {s}:\n", scan_loader.get_scan(scan_id).pose)
        # Compute initial error
        gt_pose = gt_poses[scan_id]
        pose_err = se3op.tran2vec(np.linalg.inv(scan_loader.get_scan(scan_id).pose) @ gt_pose).flatten()
        print("Initial pose error (x,y,yaw):", pose_err[0], pose_err[1], np.rad2deg(pose_err[5]))
        avg_translational_err += np.linalg.norm(pose_err[0:2])
        avg_rotational_err += np.abs(pose_err[5])
    avg_translational_err /= (num_states - 1)
    avg_rotational_err /= (num_states - 1)
    print("Average initial translational error (m): {:.4f}, Average initial rotational error (deg): {:.2f}".format(
        avg_translational_err, np.rad2deg(avg_rotational_err)
    ))

    scan_by_id = {scan.id: scan for scan in scan_loader.scans}
    for iter in range(max_iter):
        print(f"\n--- Iteration {iter} ---")

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
            print("Pose error for state {}: x,y,yaw: {:.4f}, {:.4f}, {:.2f} deg".format(
                s_idx+1, pose_err[0], pose_err[1], np.rad2deg(pose_err[5])
            ))
            avg_translational_err += np.linalg.norm(pose_err[0:2])
            avg_rotational_err += np.abs(pose_err[5])
        
        avg_translational_err /= (num_states - 1)
        avg_rotational_err /= (num_states - 1)
        print("Average translational error (m): {:.4f}, Average rotational error (deg): {:.2f}".format(
            avg_translational_err, np.rad2deg(avg_rotational_err)
        ))

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
        vox_map.plot(save_path=osp.join(voxel_img_output_path, f'voxel_map_iter_{iter}.png'))
        # Plot cost history
        plot_cost_history(cost_history, path=osp.join(voxel_img_output_path, 'cost_history.png'))

    print("Final states:")
    for scan_id in sorted_pose_keys:
        s_idx = pose_key_to_idx[scan_id]
        print(f"State {s_idx}:\n", scan_loader.get_scan(scan_id).pose)
        gt_pose = gt_poses[scan_id]
        pose_err = se3op.tran2vec(np.linalg.inv(scan_loader.get_scan(scan_id).pose) @ gt_pose)
        print("Final pose error (x,y,yaw):", pose_err[0], pose_err[1], np.rad2deg(pose_err[5]))
    
    vox_map.plot()

if __name__ == '__main__':
    main(kSeqId)
