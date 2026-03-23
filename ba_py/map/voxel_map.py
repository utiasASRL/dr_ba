import numpy as np
import random
import matplotlib.pyplot as plt
import struct
from pylgmath import se3op
import os
import os.path as osp
from scipy.ndimage import rotate
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import glob
from PIL import Image
from matplotlib.transforms import Affine2D

plt.rcParams.update({
    "text.usetex": True,          # Use LaTeX for all text
    "font.family": "serif",       # Or any LaTeX-supported font
    "legend.frameon": True
})

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
    def plot_loc_result(self, loc_results, save_path=None, show=False, title="Voxel Map with Localization Results"):
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


        # --- parameters ---
        trail_N = 25          # number of past poses to show
        veh_len = 6.0         # meters (triangle length)
        veh_wid = 3.0         # meters (triangle width)

        # --- triangle glyph (vehicle footprint) ---
        tri_patch = Polygon(
            np.zeros((3, 2)),
            closed=True,
            facecolor="cyan",
            edgecolor="black",
            linewidth=1.0,
            zorder=6
        )
        ax.add_patch(tri_patch)

        # --- trailing path ---
        trail_line, = ax.plot(
            [],
            [],
            color="cyan",
            linewidth=4,
            alpha=0.6,
            zorder=5
        )

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

        if save_path is not None:
            fig_path = osp.join(save_path, 'voxel_map_loc_results.png')
            plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="black")

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

    # loc_results is list containing map_id,scan_id,est_x,est_y,est_yaw,gt_x,gt_y,gt_yaw
    def plot_loc_paper(self, loc_results, save_path=None, show=False, title="Voxel Map with Localization Results"):
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


        # Cap vals at 0.5 for better visualization
        vals = np.clip(vals, 0.0, 0.6)
        vals = (vals - 0.0) / (0.3 - 0.0)

        ix, iy = keys[:, 0], keys[:, 1]
        iy = -iy  # flip y for correct orientation
        ix_min, ix_max = ix.min(), ix.max()
        iy_min, iy_max = iy.min(), iy.max()

        H = ix_max - ix_min + 1
        W = iy_max - iy_min + 1

        img = np.full((H, W), np.nan, dtype=np.float32)
        img[ix - ix_min, iy - iy_min] = vals

        # --- plotting ---
        fig, ax = plt.subplots(facecolor="white", figsize=(10, 8))
        ax.set_facecolor("white")

        im = ax.imshow(
            img.T,
            origin="lower",
            cmap="magma_r",
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

        # Plot localization results (estimated + ground truth trajectories)
        est_xs, est_ys = [], []
        gt_xs, gt_ys = [], []

        for ii in range(len(loc_results)):
            map_id = loc_results[ii][0]
            pose_idx = self.pose_ids.index(map_id)
            map_node_pose = self.poses[pose_idx]
            assert map_node_pose[0] == map_id, "Map ID mismatch when plotting localization results."

            # Local poses
            est_x_local = loc_results[ii][2]
            est_y_local = loc_results[ii][3]
            est_yaw_local = np.deg2rad(loc_results[ii][4])

            gt_x_local = loc_results[ii][5]
            gt_y_local = loc_results[ii][6]
            gt_yaw_local = np.deg2rad(loc_results[ii][7])

            # Map pose
            map_pose = se3op.vec2tran(
                np.array([0.0, 0.0, 0.0, 0.0, 0.0, map_node_pose[3]]).reshape(-1, 1)
            )
            map_pose[0, 3] = map_node_pose[1]
            map_pose[1, 3] = map_node_pose[2]

            # Estimated pose in local frame
            est_pose = se3op.vec2tran(
                -np.array([0.0, 0.0, 0.0, 0.0, 0.0, -est_yaw_local]).reshape(-1, 1)
            )
            est_pose[0, 3] = est_x_local
            est_pose[1, 3] = est_y_local

            # Ground-truth pose in local frame
            gt_pose = se3op.vec2tran(
                -np.array([0.0, 0.0, 0.0, 0.0, 0.0, -gt_yaw_local]).reshape(-1, 1)
            )
            gt_pose[0, 3] = gt_x_local
            gt_pose[1, 3] = gt_y_local

            # Transform into map frame
            est_in_map = map_pose @ est_pose
            gt_in_map = map_pose @ gt_pose

            # Flip y for correct orientation
            est_xs.append(est_in_map[0, 3])
            est_ys.append(-est_in_map[1, 3]) 
            gt_xs.append(gt_in_map[0, 3])
            gt_ys.append(-gt_in_map[1, 3])

        # Plot trajectories
        ax.plot(
            gt_xs, gt_ys,
            color="red",
            linewidth=2,
            linestyle="-",
            label=r"ground truth"
        )
        ax.plot(
            est_xs, est_ys,
            color="black",
            linewidth=2,
            linestyle="--",
            label=r"localized"
        )

        leg = ax.legend(facecolor="white", edgecolor="black", labelcolor="black", loc='upper right', fontsize=22)
        for legline in leg.get_lines():
            legline.set_linewidth(5)  # set the line thickness in the legend

        add_scale_bar(ax, ix_min, ix_max, iy_min, iy_max, self.res, fraction=0.30)

        ax.set_aspect("equal")
        ax.set_title(title, fontsize=14)
        ax.axis("off")
        ax.margins(0)
        plt.tight_layout(pad=0)

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        if show:
            plt.show()
            plt.close(fig)


    def plot_paper(self, save_path=None, show=False, title="Map", global_extent=None, plot_poses=True):
        if not self.voxels:
            return

        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon

        # --- convert sparse dict to dense raster ---
        keys = np.asarray(list(self.voxels.keys()), dtype=np.int32)
        vals = np.asarray([v.intensity for v in self.voxels.values()], dtype=np.float32)

        ix = keys[:, 0]
        iy = keys[:, 1]
        iy = -iy  # flip y for correct orientation

        ix_min, ix_max = ix.min(), ix.max()
        iy_min, iy_max = iy.min(), iy.max()

        H = ix_max - ix_min + 1
        W = iy_max - iy_min + 1

        vals = np.clip(vals, 0.0, 0.6)
        vals = vals / 0.3

        img = np.full((H, W), np.nan, dtype=np.float32)
        img[ix - ix_min, iy - iy_min] = vals

        # --- plotting (metric space) ---
        fig, ax = plt.subplots(figsize=(6, 6), facecolor="white")
        ax.set_facecolor("white")

        extent = [
            ix_min * self.res,
            (ix_max + 1) * self.res,
            iy_min * self.res,
            (iy_max + 1) * self.res,
        ]

        ax.imshow(
            img.T,
            origin="lower",
            cmap="magma_r",
            vmin=0.0,
            vmax=1.0,
            extent=extent,
            interpolation="nearest",
        )

        # --- pose triangle helper (metric, centered) ---
        def triangle_pts(x, y, yaw):
            veh_len = 6.0
            veh_wid = 3.0

            # forward-facing triangle in +x
            pts = np.array([
                [ veh_len / 2,  0.0],
                [-veh_len / 2,  veh_wid / 2],
                [-veh_len / 2, -veh_wid / 2],
            ])

            c, s = np.cos(yaw), np.sin(yaw)
            R = np.array([[c, -s],
                        [s,  c]])
            pts = pts @ R.T
            pts[:, 0] += x
            pts[:, 1] += y
            return pts

        # --- overlay poses (EXACT extraction as working plot) ---
        if self.poses and plot_poses: 
            for p in self.poses:
                x = p[1]        # meters
                y = p[2]        # meters
                yaw = p[3]      # radians (DO NOT convert)

                # Flip y/yaw for correct orientation
                y = -y
                yaw = -yaw

                tri = Polygon(
                    triangle_pts(x, y, yaw),
                    closed=True,
                    facecolor="#00FF00",
                    edgecolor="black",
                    linewidth=1.2,
                    zorder=6,
                )
                ax.add_patch(tri)


        # Add scale bar
        add_scale_bar(ax, ix_min, ix_max, iy_min, iy_max, self.res, fraction=0.15)


        if global_extent is not None:
            ax.set_xlim(global_extent[0], global_extent[1])
            ax.set_ylim(global_extent[2], global_extent[3])
        else:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")
        ax.axis("off")
        plt.tight_layout(pad=0)

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)
        elif show:
            plt.show()
            plt.close(fig)


    def render_localization_video_frames(
        self,
        loc_results,
        out_dir,
        zoom_range=40.0,
        n_full=30,
        n_zoom=20,
        trail_N=25,        # <-- restored
        gt_window_N=10,        # N → total window = 2N+1 GT footprints
        veh_len=6.0,
        veh_wid=3.0,
        dpi=150,
        scan_path=None,
    ):
        """
        Video frames showing:
        - estimated pose (green triangle)
        - sliding window of GT footprints (solid grey outlines)
        """

        os.makedirs(out_dir, exist_ok=True)

        # ------------------------------------------------------------------
        # Rasterize voxel map
        # ------------------------------------------------------------------
        keys = np.asarray(list(self.voxels.keys()), dtype=np.int32)
        vals = np.asarray([v.intensity for v in self.voxels.values()], dtype=np.float32)

        vals = np.clip(vals, 0.0, 0.6)
        vals = (vals - 0.0) / (0.3 - 0.0)

        ix, iy = keys[:, 0], keys[:, 1]
        iy = -iy  # flip y for correct orientation
        ix_min, ix_max = ix.min(), ix.max()
        iy_min, iy_max = iy.min(), iy.max()

        img = np.full((ix_max - ix_min + 1, iy_max - iy_min + 1), np.nan)
        img[ix - ix_min, iy - iy_min] = vals

        extent = [
            ix_min * self.res,
            (ix_max + 1) * self.res,
            iy_min * self.res,
            (iy_max + 1) * self.res,
        ]

        view_w = 2 * zoom_range
        view_h = 2 * zoom_range
        view_aspect = view_w / view_h  # normally 1.0

        # ------------------------------------------------------------------
        # Transform poses to map frame
        # ------------------------------------------------------------------
        est_xs, est_ys, est_yaws = [], [], []
        gt_xs, gt_ys, gt_yaws = [], [], []

        for r in loc_results:
            map_id = r[0]
            pose_idx = self.pose_ids.index(map_id)
            map_node_pose = self.poses[pose_idx]

            ex, ey, eyaw = r[2], r[3], np.deg2rad(r[4])
            gx, gy, gyaw = r[5], r[6], np.deg2rad(r[7])

            map_pose = se3op.vec2tran(
                np.array([0, 0, 0, 0, 0, map_node_pose[3]]).reshape(-1, 1)
            )
            map_pose[0, 3] = map_node_pose[1]
            map_pose[1, 3] = map_node_pose[2]

            def to_map(x, y, yaw):
                T = se3op.vec2tran(
                    -np.array([0, 0, 0, 0, 0, -yaw]).reshape(-1, 1)
                )
                T[0, 3] = x
                T[1, 3] = y
                M = map_pose @ T
                return M[0, 3], M[1, 3], map_node_pose[3] + yaw

            ex, ey, eyaw = to_map(ex, ey, eyaw)
            gx, gy, gyaw = to_map(gx, gy, gyaw)

            est_xs.append(ex)
            est_ys.append(ey)
            est_yaws.append(eyaw)

            gt_xs.append(gx)
            gt_ys.append(gy)
            gt_yaws.append(gyaw)

        est_xs, est_ys, est_yaws = map(np.asarray, (est_xs, est_ys, est_yaws))
        gt_xs, gt_ys, gt_yaws = map(np.asarray, (gt_xs, gt_ys, gt_yaws))

        # Flip ys and yaw for correct orientation
        est_ys = -est_ys
        est_yaws = -est_yaws
        gt_ys = -gt_ys
        gt_yaws = -gt_yaws

        # ------------------------------------------------------------------
        # Scan setup
        # ------------------------------------------------------------------
        SCAN_RES = 0.1          # m / pixel
        SCAN_RADIUS_M = 100.0   # meters
        SCAN_RADIUS_PX = int(SCAN_RADIUS_M / SCAN_RES)  # 1000 px

        # --------------------------------------------------
        # Circular max-range mask (scan-local)
        # --------------------------------------------------
        size = 2 * SCAN_RADIUS_PX + 1
        yy, xx = np.ogrid[-SCAN_RADIUS_PX:SCAN_RADIUS_PX+1,
                        -SCAN_RADIUS_PX:SCAN_RADIUS_PX+1]

        circle_mask = (xx**2 + yy**2) <= (SCAN_RADIUS_PX**2)

        scan_files = None
        if scan_path is not None:
            scan_files = sorted(
                glob.glob(os.path.join(scan_path, "*.png"))
            )
            assert len(scan_files) >= len(loc_results), \
                "Not enough scans for number of poses"

        # ------------------------------------------------------------------
        # Plot setup
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 8), facecolor="white")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.imshow(
            img.T,
            origin="lower",
            cmap="magma_r",
            vmin=0.0,
            vmax=1.0,
            extent=extent,
            interpolation="nearest",
        )
        ax.set_aspect("equal")
        ax.axis("off")

        scan_im = None
        scan_tf = None
        scan_res = 0.1  # meters per pixel

        if scan_files is not None:
            scan0 = np.asarray(Image.open(scan_files[0]).convert("L"))

            H, W = scan0.shape

            # physical size in meters
            half_w_m = SCAN_RADIUS_M
            half_h_m = SCAN_RADIUS_M

            scan_im = ax.imshow(
                scan0,
                cmap="gray",
                origin="lower",
                alpha=0.8,
                zorder=2,              # above map, below glyphs
                extent=[
                    -half_w_m, half_w_m,
                    -half_h_m, half_h_m,
                ],
                visible=False,         # <-- hidden until phase 3
            )

            scan_tf = Affine2D()
            scan_im.set_transform(scan_tf + ax.transData)


        # ------------------------------------------------------------------
        # Geometry helper
        # ------------------------------------------------------------------
        def triangle_pts(x, y, yaw, scale=1.0):
            pts = np.array([
                [ veh_len / 2,  0.0],
                [-veh_len / 2,  veh_wid / 2],
                [-veh_len / 2, -veh_wid / 2],
            ]) * scale

            c, s = np.cos(yaw), np.sin(yaw)
            pts = pts @ np.array([[c, -s], [s, c]]).T
            pts[:, 0] += x
            pts[:, 1] += y
            return pts

        # ------------------------------------------------------------------
        # Estimated pose artist
        # ------------------------------------------------------------------
        est_tri = Polygon(
            np.zeros((3, 2)),
            closed=True,
            facecolor="#00FF00",
            edgecolor="black",
            linewidth=1.2,
            zorder=6,
        )
        ax.add_patch(est_tri)

        est_trail, = ax.plot(
            [],
            [],
            color="#00FF00",
            linewidth=4,
            alpha=0.6,
            zorder=5,
        )


        # --- single GT footprint (future target) ---
        gt_target = Polygon(
            np.zeros((3, 2)),
            closed=True,
            facecolor="none",
            edgecolor="0.6",
            linewidth=1.2,
            zorder=4,
        )
        ax.add_patch(gt_target)

        # --- GT trailing path ---
        gt_trail, = ax.plot(
            [],
            [],
            color="0.6",
            linewidth=3.0,
            alpha=0.8,
            zorder=3,
        )


        scan_cache = {}

        def get_scan(k):
            if k not in scan_cache:
                scan_cache[k] = np.asarray(
                    Image.open(scan_files[k]).convert("L")
                )
            return scan_cache[k]


        def aspect_correct_extent(xmin, xmax, ymin, ymax, target_aspect):
            w = xmax - xmin
            h = ymax - ymin
            cx = 0.5 * (xmin + xmax)
            cy = 0.5 * (ymin + ymax)

            if w / h > target_aspect:
                # too wide → expand height
                h_new = w / target_aspect
                w_new = w
            else:
                # too tall → expand width
                w_new = h * target_aspect
                h_new = h

            return (
                cx - w_new / 2,
                cx + w_new / 2,
                cy - h_new / 2,
                cy + h_new / 2,
            )


        # ------------------------------------------------------------------
        # Update function
        # ------------------------------------------------------------------
        def update_frame(k):
            # --------------------------------------------------
            # Estimated pose
            # --------------------------------------------------
            est_tri.set_xy(triangle_pts(est_xs[k], est_ys[k], est_yaws[k]))

            i0 = max(0, k - trail_N + 1)
            rear = veh_len / 2
            xe = est_xs[k] - rear * np.cos(est_yaws[k])
            ye = est_ys[k] - rear * np.sin(est_yaws[k])

            xs = list(est_xs[i0:k]) + [xe]
            ys = list(est_ys[i0:k]) + [ye]
            est_trail.set_data(xs, ys)

            # --------------------------------------------------
            # Current GT pose (same index k)
            # --------------------------------------------------
            gt_target.set_xy(
                triangle_pts(gt_xs[k], gt_ys[k], gt_yaws[k], scale=2.0)
            )

            # --------------------------------------------------
            # GT trailing path (same semantics as estimate)
            # --------------------------------------------------
            i_gt0 = max(0, k - trail_N + 1)

            rear_gt = veh_len * 0.6 / 2.0  # slightly larger than estimate
            xg = gt_xs[k] - rear_gt * np.cos(gt_yaws[k])
            yg = gt_ys[k] - rear_gt * np.sin(gt_yaws[k])

            xs_gt = list(gt_xs[i_gt0:k]) + [xg]
            ys_gt = list(gt_ys[i_gt0:k]) + [yg]

            gt_trail.set_data(xs_gt, ys_gt)

            # --------------------------------------------------
            # Scan overlay
            # --------------------------------------------------
            if scan_im is not None:
                scan_full = np.asarray(Image.open(scan_files[k]).convert("L"))

                # Invert color
                # scan_full = 255 - scan_full

                H, W = scan_full.shape
                cx, cy = W // 2, H // 2

                scan = scan_full[
                    cy - SCAN_RADIUS_PX : cy + SCAN_RADIUS_PX + 1,
                    cx - SCAN_RADIUS_PX : cx + SCAN_RADIUS_PX + 1,
                ]
                scan = np.ma.array(scan, mask=~circle_mask)

                scan_im.set_data(scan)

                scan_tf.clear()

                # 1) Flip scan about its up (Y) axis
                scan_tf.scale(1.0, -1.0)

                # 2) Rotate scan:
                #    scan +Y is vehicle forward, map +X is vehicle forward
                scan_tf.rotate(est_yaws[k] - np.pi / 2.0)

                # 3) Translate into map frame
                scan_tf.translate(est_xs[k], est_ys[k])

        # ------------------------------------------------------------------
        # Frame writing
        # ------------------------------------------------------------------
        frame_idx = 0

        def save_frame():
            nonlocal frame_idx
            fig.savefig(
                os.path.join(out_dir, f"frame_{frame_idx:05d}.png"),
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=0,
            )
            frame_idx += 1

        # ------------------------------------------------------------------
        # Phase 1: full map
        # ------------------------------------------------------------------
        full_xmin, full_xmax, full_ymin, full_ymax = aspect_correct_extent(
            extent[0], extent[1],
            extent[2], extent[3],
            view_aspect,
        )

        ax.set_xlim(full_xmin, full_xmax)
        ax.set_ylim(full_ymin, full_ymax)
        for _ in range(n_full):
            save_frame()

        # ------------------------------------------------------------------
        # Phase 2: zoom to first GT pose
        # ------------------------------------------------------------------
        # Aspect-correct full-map box (same as Phase 1)
        # First GT pose (zoom target)
        cx, cy = gt_xs[0], gt_ys[0]

        # Aspect-correct full-map box
        full_xlim = np.array([full_xmin, full_xmax])
        full_ylim = np.array([full_ymin, full_ymax])

        # Aspect-correct zoom box centered on first pose
        zoom_xmin, zoom_xmax, zoom_ymin, zoom_ymax = aspect_correct_extent(
            cx - zoom_range, cx + zoom_range,
            cy - zoom_range, cy + zoom_range,
            view_aspect,
        )
        zoom_xlim = np.array([zoom_xmin, zoom_xmax])
        zoom_ylim = np.array([zoom_ymin, zoom_ymax])

        for a in np.linspace(0, 1, n_zoom):
            ax.set_xlim((1 - a) * full_xlim + a * zoom_xlim)
            ax.set_ylim((1 - a) * full_ylim + a * zoom_ylim)
            update_frame(0)
            save_frame()

        # ------------------------------------------------------------------
        # Phase 3: follow estimate with GT context window
        # ------------------------------------------------------------------
        # --- legend proxy artists ---
        legend_est = Polygon(
            [[0, 0], [1, 0], [0.5, 1]],
            closed=True,
            facecolor="#00FF00",
            edgecolor="black",
            linewidth=1.2,
        )

        legend_gt = Polygon(
            [[0, 0], [1, 0], [0.5, 1]],
            closed=True,
            facecolor="0.6",
            edgecolor="0.6",
            linewidth=1.2,
        )

        legend = ax.legend(
            handles=[legend_est, legend_gt],
            labels=["DRL", "GT"],
            loc="upper right",
            frameon=True,
            framealpha=0.95,
            facecolor="white",
            edgecolor="0.8",
            fontsize=12,
        )
        legend.set_zorder(10)
        legend.set_bbox_to_anchor((1.0, 1.0), transform=ax.transAxes)

        if scan_im is not None:
            scan_im.set_visible(True)

        for k in range(len(gt_xs)):
            ax.set_xlim(est_xs[k] - zoom_range, est_xs[k] + zoom_range)
            ax.set_ylim(est_ys[k] - zoom_range, est_ys[k] + zoom_range)
            update_frame(k)
            save_frame()

        plt.close(fig)


def add_scale_bar(ax, ix_min, ix_max, iy_min, iy_max, res, fraction=0.3):
    """
    Add a scale bar to the bottom-left corner of a map.
    
    fraction: fraction of map width to use for scale bar length
    """
    map_width_m = (ix_max - ix_min) * res
    map_height_m = (iy_max - iy_min) * res

    # Choose a nice round number for the scale bar
    raw_length = map_width_m * fraction
    def round_to_nice(x):
        exp = 10 ** (len(str(int(x))) - 1)
        for n in [1, 2, 2.5, 3, 5, 10, 50]:
            if x <= n * exp:
                return n * exp
        return 10 * exp
    length_m = round_to_nice(raw_length)

    # Bar size & position
    bar_height = 0.01 * map_height_m
    bar_x = ix_min * res + 0.5 * map_width_m  # 5% padding from right
    bar_y = iy_min * res + 0.9 * map_height_m  # 5% padding from bottom

    # Draw rectangle
    ax.add_patch(plt.Rectangle(
        (bar_x, bar_y),
        length_m,
        bar_height,
        facecolor='white',
        edgecolor='black',
        linewidth=0.8
    ))

    # Add label above bar
    ax.text(
        bar_x + length_m / 2,
        bar_y + 1 * bar_height,
        f"{round(length_m)} m",
        ha='center',
        va='bottom',
        fontsize=42,
        color='black'
    )

