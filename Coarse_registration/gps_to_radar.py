#!/usr/bin/env python3
"""
gps_to_radar.py – Convert Boreas 200 Hz Applanix poses to radar-frame poses
(x, y, yaw) in the ENU world frame.
"""

import math, argparse, numpy as np, pandas as pd
from pathlib import Path
from pyboreas.utils.utils import yawPitchRollToRot

def build_T(R, t_xyz):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t_xyz
    return T

def yaw_from_R(R):
    return math.degrees(math.atan2(R[1, 0], R[0, 0]))

def main(calib_dir: Path, gps_csv: Path, out_csv: Path):
    # load extrinsics
    T_A_L = np.loadtxt(calib_dir / "T_applanix_lidar.txt") # Applanix ← LiDAR
    T_R_L = np.loadtxt(calib_dir / "T_radar_lidar.txt") # Radar ← LiDAR
    T_A_R = T_A_L @ np.linalg.inv(T_R_L) # Applanix ← Radar

    df = pd.read_csv(gps_csv)


    # transform each pose
    out = []
    for r in df.itertuples(index=False):
        # rotation ENU ← Applanix (Boreas util uses rad)
        R_E_A = yawPitchRollToRot(r.heading, r.pitch, r.roll)
        T_E_A = build_T(R_E_A, [r.easting, r.northing, r.altitude])

        # ENU ← Radar = (ENU ← Applanix) · (Applanix ← Radar)
        T_E_R = T_E_A @ T_A_R

        # extract
        x_r, y_r = T_E_R[0, 3], T_E_R[1, 3]
        yaw_r = yaw_from_R(T_E_R[:3, :3])

        out.append((r.GPSTime, x_r, y_r, yaw_r))

    pd.DataFrame(out, columns=["time_s", "easting_m", "northing_m", "yaw_deg"]
    ).to_csv(out_csv, index=False)

    print(f"wrote {len(out)} radar poses → {out_csv}")
    # print(f"static yaw in calibration (A←R) = {yaw_from_R(T_A_R[:3,:3]):.3f}°")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib_dir", required=True, help="…/calib/")
    ap.add_argument("--gps_csv", required=True, help="…/applanix/gps_post_process.csv")
    ap.add_argument("--out_csv", default="radar_frame_poses.csv")
    args = ap.parse_args()

    main(Path(args.calib_dir), Path(args.gps_csv), Path(args.out_csv))