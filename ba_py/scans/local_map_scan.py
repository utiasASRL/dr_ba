import numpy as np
from .base_scan import BaseScan
import matplotlib.pyplot as plt
from utils.se2_utils import circle_dot_operator
nan = float('nan')

class LocalMapScan(BaseScan):
    def __init__(self, pose, img, res, scan_id):
        super().__init__(pose, scan_id)
        self.res = res
        self.pixels = np.asarray(img)
        self.img_height, self.img_width = self.pixels.shape

    def coord_to_pixel(self, x, y, jac=False):
        # Convert world frame coordinates to pixel coordinates
        D = np.array([[0, 1/self.res, 0], [-1/self.res, 0, 0]])
        p = D @ np.linalg.inv(self.pose_2d) @ np.array([[x], [y], [1]]) + 0.5 * np.array([[self.img_width], [self.img_height]])

        if jac:
            return p[0, 0], p[1, 0], D @ np.linalg.inv(self.pose_2d) @ circle_dot_operator(np.array([[x], [y]]))

        return p[0, 0], p[1, 0]

    def get_root_pixel_coords(self, x, y):
        # Convert world frame coordinates to pixel coordinates
        u, v = self.coord_to_pixel(x, y)
        return int(np.floor(u)), int(np.floor(v))

    def check_coverage_at_point(self, x, y):
        # Check if the four pixels surrounding (x, y) are within image bounds
        a, b = self.get_root_pixel_coords(x, y)

        return (a >= 0 and a < self.img_width - 1 and
                b >= 0 and b < self.img_height - 1)

    def interpolate(self, x, y, jac=False):
        if not self.check_coverage_at_point(x, y):
            if jac:
                return nan, nan
            return nan

        # Get pixel coordinates
        u, v, d_g_d_T = self.coord_to_pixel(x, y, jac=True)
        a, b = self.get_root_pixel_coords(x, y)

        # Get intensities at the four corners
        int_ab = self.pixels[b, a]
        int_ab1 = self.pixels[b + 1, a]
        int_a1b = self.pixels[b, a + 1]
        int_a1b1 = self.pixels[b + 1, a + 1]

        # Get weights (measure offsets in meters within the voxel)
        u_tilde = u - a
        v_tilde = v - b
        w0 = (1 - u_tilde) * (1 - v_tilde)
        w1 = (1 - u_tilde) * v_tilde
        w2 = u_tilde * (1 - v_tilde)
        w3 = u_tilde * v_tilde

        # Bilinear interpolation
        int_xy = (w0 * int_ab + w1 * int_ab1 + w2 * int_a1b + w3 * int_a1b1)

        if jac:
            # Compute Jacobian
            d_B_d_u = (1 - v_tilde) * (int_a1b - int_ab) + v_tilde * (int_a1b1 - int_ab1)
            d_B_d_v = (1 - u_tilde) * (int_ab1 - int_ab) + u_tilde * (int_a1b1 - int_a1b)
            d_B_d_g = np.array([[d_B_d_u, d_B_d_v]])  # 1x2
            d_B_d_T = d_B_d_g @ d_g_d_T  # 1x3
            return int_xy, d_B_d_T

        return int_xy

    def visualize(self):
        plt.imshow(self.pixels, cmap='gray', origin='upper')
        plt.title(f"Scan ID: {self.id}")
        plt.xlabel("u (pixels)")
        plt.ylabel("v (pixels)")
        plt.colorbar(label="Intensity")
        plt.show()

    def visualize_with_xy(self, x, y):
        u, v = self.coord_to_pixel(x, y)
        print(f"World coords (x={x:.2f}, y={y:.2f}) -> Pixel coords (u={u:.2f}, v={v:.2f})")
        plt.plot(u, v, 'ro')
        plt.imshow(self.pixels, cmap='gray', origin='upper')
        plt.title(f"Scan ID: {self.id}")
        plt.xlabel("u (pixels)")
        plt.ylabel("v (pixels)")
        plt.colorbar(label="Intensity")
        plt.show()