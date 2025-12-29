import numpy as np
from scipy.spatial import Delaunay
from .base_scan import BaseScan
from utils.se2_utils import circle_dot_operator
nan = float('nan')

class PointScan(BaseScan):
    def __init__(self, pose, points, scan_id):
        super().__init__(pose, scan_id)
        self.points = np.asarray(points)
        self.xy = self.points[:, :2]
        self.Iv = self.points[:, 2]
        self.tri = Delaunay(self.xy)

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

    def check_coverage_at_point(self, x_map, y_map):
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