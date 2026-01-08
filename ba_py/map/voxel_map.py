import numpy as np
import random
import matplotlib.pyplot as plt

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

