import numpy as np
import random
import matplotlib.pyplot as plt
import struct
from pylgmath import se3op


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
        self.poses = []
        self.pose_ids = []

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

    def get_sorted_voxels(self, downsample_factor=1.0):
        """
        downsample_factor: float in (0, 1], fraction of keys to keep.
                        1.0 keeps all keys.
        """
        keys = list(self.voxels.keys())

        if downsample_factor < 1.0:
            n_keep = max(1, int(len(keys) * downsample_factor))
            keys = random.sample(keys, n_keep)

        return sorted(keys, key=lambda k: (k[0], k[1]))

    def init_map(self, pose, max_dist):
        # Initialize empty voxels within max_dist of the pose
        x_center = pose[0, 3]
        y_center = pose[1, 3]
        a_center, b_center = self.index(x_center, y_center)
        num_voxels = int(np.ceil(max_dist / self.res))
        for da in range(-num_voxels, num_voxels +1):
            for db in range(-num_voxels, num_voxels +1):
                if (da*self.res)**2 + (db*self.res)**2 > max_dist**2:
                    continue
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

    def load_from_binary(self, file_path):
        with open(file_path, "rb") as f:
            # Metadata (explicit little-endian, no padding)
            self.res, = struct.unpack("<d", f.read(8))
            num_poses, = struct.unpack("<I", f.read(4))
            num_voxels, = struct.unpack("<I", f.read(4))

            # Poses: int32 + 4 doubles = 28 bytes
            pose_struct = struct.Struct("<idddd")
            for _ in range(num_poses):
                pose_id, x, y, yaw, ate = pose_struct.unpack(f.read(pose_struct.size))
                ate = np.round(ate, 5)
                self.pose_ids.append(pose_id)
                self.poses.append((pose_id, x, y, yaw, ate))

            # Voxels: int32, int32, double = 16 bytes
            voxel_struct = struct.Struct("<iid")
            voxels_read = 0
            for _ in range(num_voxels):
                data = f.read(voxel_struct.size)
                if len(data) < voxel_struct.size:
                    print(f"[WARN] File incomplete. Read {voxels_read}/{num_voxels} voxels")
                    break
                x, y, intensity = voxel_struct.unpack(data)
                self.voxels[(x, y)] = self.Voxel(intensity)
                voxels_read += 1

    # loc_results is list containing map_id,scan_id,est_x,est_y,est_yaw,gt_x,gt_y,gt_yaw
    def plot_loc_result(self, loc_results, show=False, title="Voxel Map with Localization Results"):
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
        fig, ax = plt.subplots(facecolor="black", figsize=(10, 8))
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

        # Plot localization results

        # Get corresponding map pose
        est_xs = []
        est_ys = []
        translation_errs = []
        for ii in range(len(loc_results)):
            map_id = loc_results[ii][0]
            pose_idx = self.pose_ids.index(map_id)
            map_node_pose = self.poses[pose_idx]
            assert map_node_pose[0] == map_id, "Map ID mismatch when plotting localization results."

            # Transform estimated and ground truth positions to map frame
            est_x_local = loc_results[ii][2]
            est_y_local = loc_results[ii][3]
            est_yaw_local = loc_results[ii][4] * (np.pi / 180.0)  # Convert to radians
            gt_x_local = loc_results[ii][5]
            gt_y_local = loc_results[ii][6]
            gt_yaw_local = loc_results[ii][7] * (np.pi / 180.0)  # Convert to radians

            # Create SE3 transforms
            map_pose = se3op.vec2tran(np.array([0.0, 0.0, 0.0, 0.0, 0.0, map_node_pose[3]]).reshape(-1, 1))
            map_pose[0, 3] = map_node_pose[1]
            map_pose[1, 3] = map_node_pose[2]
            est_pose = se3op.vec2tran(-np.array([0.0, 0.0, 0.0, 0.0, 0.0, -est_yaw_local]).reshape(-1, 1))
            est_pose[0, 3] = est_x_local
            est_pose[1, 3] = est_y_local
            gt_pose = se3op.vec2tran(-np.array([0.0, 0.0, 0.0, 0.0, 0.0, -gt_yaw_local]).reshape(-1, 1))
            gt_pose[0, 3] = gt_x_local
            gt_pose[1, 3] = gt_y_local

            # print("Map ID:", map_id)
            # print("Estimated local pose:\n", est_pose)
            # print("Nearest map estimated pose:\n", map_pose)

            est_in_map = map_pose @ est_pose
            gt_in_map = map_pose @ gt_pose

            est_xs.append(est_in_map[0, 3])
            est_ys.append(est_in_map[1, 3])
            translation_error = np.sqrt((est_x_local - gt_x_local)**2 + (est_y_local - gt_y_local)**2)
            translation_errs.append(translation_error)


        # reference_map_ids = [res[0] for res in loc_results]

        # est_xs = [res[2] for res in loc_results]
        # est_ys = [res[3] for res in loc_results]
        # gt_xs = [res[5] for res in loc_results]
        # gt_ys = [res[6] for res in loc_results]
        ax.scatter(est_xs, est_ys, c=translation_errs, cmap='hot', s=10, label='Estimated Positions')
        cbar_err = plt.colorbar(ax.collections[-1], ax=ax)
        cbar_err.set_label("Translation RMSE (m)", color="white")
        cbar_err.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar_err.ax.get_yticklabels(), color="white")
        ax.legend()

        ax.set_xlim(ix_min * self.res, (ix_max + 1) * self.res)
        ax.set_ylim(iy_min * self.res, (iy_max + 1) * self.res)
        ax.set_title(title, color="white")
        ax.set_xlabel("X", color="white")
        ax.set_ylabel("Y", color="white")
        ax.tick_params(colors="white")
        ax.set_aspect("equal")

        if show:
            plt.show()
            plt.close(fig)

    def plot(self, save_path=None, iter=None, show=False):
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
        fig, ax = plt.subplots(facecolor="black", figsize=(10, 8))
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

        # Plot poses overlaid if available
        if self.poses:
            pose_xs = [p[1] for p in self.poses]
            pose_ys = [p[2] for p in self.poses]
            ate = [p[4] for p in self.poses]
            sc = ax.scatter(pose_xs, pose_ys, c=ate, cmap='hot', s=20, edgecolors='k', label='Poses')
            cbar_ate = plt.colorbar(sc, ax=ax)
            cbar_ate.set_label("ATE", color="white")
            cbar_ate.ax.yaxis.set_tick_params(color="white")
            plt.setp(cbar_ate.ax.get_yticklabels(), color="white")

        ax.set_xlim(ix_min * self.res, (ix_max + 1) * self.res)
        ax.set_ylim(iy_min * self.res, (iy_max + 1) * self.res)

        if iter is not None:
            ax.set_title(f"Voxel Map - Iteration {iter}", color="white")
        else:
            ax.set_title("Voxel Map", color="white")
        ax.set_xlabel("X", color="white")
        ax.set_ylabel("Y", color="white")
        ax.tick_params(colors="white")
        ax.set_aspect("equal")

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="black")
            plt.close(fig)
        elif show:
            plt.show()
            plt.close(fig)

