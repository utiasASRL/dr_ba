#!/usr/bin/env python3
'''
outputs the translation and rotation error proposed by SIFT compared to ground truth translations
'''

import os
import cv2
import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def build_centered_warp(M, w, h):
    """Return a 2x3 warp that applies M around the image centre."""
    T1 = np.array([[1, 0, -w/2],
                   [0, 1, -h/2],
                   [0, 0,    1 ]], float)
    T2 = np.array([[1, 0,  w/2],
                   [0, 1,  h/2],
                   [0, 0,    1 ]], float)
    M3 = np.vstack([M, [0, 0, 1]])
    return (T2 @ M3 @ T1)[:2]


def affine_from_world(dx_m, dy_m, dyaw_deg, res_m_px, w, h):
    """
    World (ENU) → pixel-frame affine centred on the image.
    Handles axis flip (north = -y) and yaw sign.
    """
    dx_px =  dx_m / res_m_px
    dy_px = -dy_m / res_m_px          # y flip
    yaw   = -np.deg2rad(dyaw_deg)     # sign flip for image frame

    c, s = np.cos(yaw), np.sin(yaw)
    M    = np.array([[c, -s, dx_px],
                     [s,  c, dy_px]], float)
    return build_centered_warp(M, w, h)


def load_gt(gt_csv):
    df = pd.read_csv(gt_csv)
    return df[['time_s','easting_m','northing_m','yaw_deg']].sort_values('time_s')


def interp_pose(gt_df, ts):
    arr = gt_df.time_s.values
    idx = np.searchsorted(arr, ts)
    if idx == 0 or idx == len(arr):
        raise ValueError("timestamp outside GT range")
    alpha = (ts - arr[idx-1]) / (arr[idx] - arr[idx-1])
    p0, p1 = gt_df.iloc[idx-1], gt_df.iloc[idx]
    x  = p0.easting_m  * (1-alpha) + p1.easting_m  * alpha
    y  = p0.northing_m * (1-alpha) + p1.northing_m * alpha
    yaw = p0.yaw_deg   * (1-alpha) + p1.yaw_deg   * alpha
    return x, y, yaw


def plot_side_by_side(img1, img2, M_gt, M_est, out_png, trans_err_m=None, rot_err_deg=None, d_gt_m=None, alpha=0.5):
    h, w = img1.shape
    w_gt  = cv2.warpAffine(img2, M_gt,  (w, h), flags=cv2.INTER_LINEAR)
    w_est = cv2.warpAffine(img2, M_est, (w, h), flags=cv2.INTER_LINEAR)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(img1, cmap='gray'); axs[0].imshow(w_gt,  cmap='hot', alpha=alpha)
    axs[0].set_title("Ground-truth overlay");  axs[0].axis('off')
    if None not in (trans_err_m, rot_err_deg, d_gt_m):
        axs[1].set_title(
            f"SIFT overlay\n"
            f"gt distance = {d_gt_m:2f} m"
            f"trans error = {trans_err_m:2f} m"
            f"rot error = {rot_err_deg:2f} °"
        )
    axs[1].imshow(img1, cmap='gray'); axs[1].imshow(w_est, cmap='hot', alpha=alpha)
    # axs[1].set_title("SIFT overlay");          axs[1].axis('off')
    plt.tight_layout();  plt.savefig(out_png, dpi=200);  plt.close(fig)

def se2_from_xyyaw(x, y, yaw_deg):
    """Build a 3x3 homogeneous transform from (x, y, yaw)."""
    θ = np.deg2rad(yaw_deg)
    c, s = np.cos(θ), np.sin(θ)
    return np.array([
        [ c, -s,  x],
        [ s,  c,  y],
        [ 0,  0,  1],
    ], dtype=float)

def compute_errors_and_deltaT(xi, yi, yiaw, xj, yj, jyaw, M, res, img_w, img_h): 
    """
    img_w, img_h : local-map image size in pixels
    """
    # ground-truth relative SE(2)
    T_gt_S1 = se2_from_xyyaw(xi, yi, yiaw)
    T_gt_S2 = se2_from_xyyaw(xj, yj, jyaw)
    T_gt_rel = np.linalg.inv(T_gt_S1) @ T_gt_S2

    # SIFT affine → metric SE(2)
    R_px = M[:2, :2] # rotation in pixel coords
    t_px = M[:2, 2] # translation at top-left

    # shift translation from (0,0) to image-centre (w/2, h/2)
    c = np.array([img_w * 0.5, img_h * 0.5])
    t_centre_px = R_px @ c + t_px - c

    # px → metres (note: image Y down → ENU Y up ⇒ sign flip)
    tx_m = t_centre_px[0] * res
    ty_m = -t_centre_px[1] * res

    yaw_est = math.degrees(math.atan2(R_px[1, 0], R_px[0, 0]))
    T_est = se2_from_xyyaw(tx_m, ty_m, yaw_est)

    print("R_px =\n", R_px)
    print("t_px_topLeft =", t_px)
    print("t_px_centre =", t_centre_px)
    print("tx_m, ty_m =", tx_m, ty_m)

    # errors
    dx = T_est[0, 2] - T_gt_rel[0, 2]
    dy = T_est[1, 2] - T_gt_rel[1, 2]
    trans_err = math.hypot(dx, dy)

    rot_err = yaw_est - math.degrees(math.atan2(T_gt_rel[1, 0],
    T_gt_rel[0, 0]))
    rot_err = ((rot_err + 180) % 360) - 180

    deltaT = np.linalg.inv(T_gt_rel) @ T_est
    d_gt = math.hypot(T_gt_rel[0, 2], T_gt_rel[1, 2])

    return (T_gt_S1, T_gt_S2, trans_err, rot_err, d_gt, deltaT)

def parse_args():
    p = argparse.ArgumentParser(
        description="Batch GT-vs-SIFT overlay and error CSV")
    p.add_argument('--loops',   required=True,
                   help="CSV with columns 'scan_i','scan_j'")
    p.add_argument('--imgdir',  required=True,
                   help="Directory containing scan images")
    p.add_argument('--gt',      required=True,
                   help="global_pose_with_yaw.csv")
    p.add_argument('--res', type=float, default=0.2384,
                   help="Metres per pixel")
    p.add_argument('--ratio', type=float, default=0.85,
                   help="Lowe's ratio-test threshold")
    p.add_argument('--ransac_thresh', type=float, default=3.0,
                   help="RANSAC reprojection threshold (px)")
    p.add_argument('--start', type=int, default=0,
                   help="Zero-based row index to start from")
    p.add_argument('--outdir', default='vis_gt_vs_sift_with_error_his',
                   help="Folder to save PNGs and CSV")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    loops = pd.read_csv(args.loops)
    if args.start > 0:
        loops = loops.iloc[args.start:].reset_index(drop=True)

    gt_df  = load_gt(args.gt)
    sift   = cv2.SIFT_create()
    bf     = cv2.BFMatcher()

    records = []
    for _, row in loops.iterrows():
        si, sj = row.scan_i_name, row.scan_j_name

        file_i = f"{si}"
        file_j = f"{sj}"
        path_i = os.path.join(args.imgdir, file_i)
        path_j = os.path.join(args.imgdir, file_j)

        img1 = cv2.imread(path_i, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(path_j, cv2.IMREAD_GRAYSCALE)


        # if img1_src is None or img2_src is None:
        #     print(f"warning: missing scan {si} or {sj}");  continue
        
        # # Histogram Equalization
        img1 = cv2.equalizeHist(img1)   
        img2 = cv2.equalizeHist(img2)   

        # cv2.imshow('Source image', img1_src)
        # cv2.imshow('Equalized image', img1)


        # SIFT + RANSAC
        k1, d1 = sift.detectAndCompute(img1, None)
        k2, d2 = sift.detectAndCompute(img2, None)
        if d1 is None or d2 is None:  continue
        good = [m for m,n in bf.knnMatch(d1, d2, 2)
                if m.distance < args.ratio * n.distance]
        if len(good) < 4:  continue

        h, w = img1.shape
        centre = np.array([w/2, h/2], np.float32)
        pts1 = np.float32([k1[m.queryIdx].pt for m in good]) - centre
        pts2 = np.float32([k2[m.trainIdx].pt for m in good]) - centre
        M_c, _ = cv2.estimateAffinePartial2D(
            pts1, pts2, method=cv2.RANSAC,
            ransacReprojThreshold=args.ransac_thresh, confidence=0.99)
        if M_c is None:  continue

        # un-centre & flip to full pixel-frame (image coords)
        R_c, t_c = M_c[:, :2], M_c[:, 2]
        t_img = t_c + (np.eye(2) - R_c) @ centre
        M_est = np.column_stack([R_c, t_img])

        # Ground-truth affine
        ts_i = float(os.path.splitext(si)[0]) / 1e6
        ts_j = float(os.path.splitext(sj)[0]) / 1e6
        xi, yi, yawi = interp_pose(gt_df, ts_i)
        xj, yj, yawj = interp_pose(gt_df, ts_j)

        # 1. ENU translation (m) 
        dE, dN = xj - xi, yj - yi

        # 2. ENU  →  Applanix body (rotate by –yaw_i)
        θ = np.deg2rad(yawi)               # yaw_i: ENU → body
        c, s =  np.cos(θ), np.sin(θ)
        d_body = np.array([[ c, s],        # R_body_ENU.T  (–θ)
                        [-s, c]]) @ np.array([dE, dN])

        # Now d_body = [left(+), forward(+)] in VEHICLE frame
        dx_body, dy_body = d_body          # (m)

        # 3. Body  →  radar image axes 
        yaw_radar_in_body =  +np.pi/2          #  +90°
        cr, sr = np.cos(yaw_radar_in_body), np.sin(yaw_radar_in_body)
        d_radar = np.array([[ cr, -sr],
                            [ sr,  cr]]) @ np.array([dx_body, dy_body])
        dx_r, dy_r = d_radar                   # radar x = right, y = up


        # 4. Build pixel-frame affine (handles y-flip & centring) 
        dyaw = (yawj - yawi + 180) % 360 - 180
        M_gt = affine_from_world(dx_r, dy_r, dyaw,
                                args.res, w, h)


        # Visualisation 
        out_png = os.path.join(
            args.outdir, f"{os.path.splitext(si)[0]}__{os.path.splitext(sj)[0]}.png")
        img_h, img_w = img1.shape[:2]

        # 1. Recover the SIFT‐estimated translation in METRES
        R_px = M_est[:, :2] # 2×2
        t_px = M_est[:, 2] # translation at top‐left

        # shift origin to image‐center (w/2,h/2):
        c = np.array([img_w*0.5, img_h*0.5], float)
        t_ctr = R_px @ c + t_px - c # in pixels

        # pixel‐→metre, flipping Y:
        tx_est = t_ctr[0] * args.res
        ty_est = -t_ctr[1] * args.res

        # 2. Recover the SIFT‐estimated yaw (degrees) in IMAGE frame
        yaw_est = math.degrees(math.atan2(R_px[1,0], R_px[0,0]))

        # 3. Compare vectors in the same frame:
        # (dx_r, dy_r) is the ground‐truth translation in the same RADAR image frame
        trans_err = math.hypot(tx_est - dx_r, ty_est - dy_r)

        # 4. Rotation error is the signed difference:
        rot_err = ((yaw_est - dyaw + 180) % 360) - 180

        # 5. Pass these to the plot & CSV:
        plot_side_by_side(img1, img2, M_gt, M_est, out_png,
        trans_err_m=trans_err,
        rot_err_deg=rot_err,
        d_gt_m=math.hypot(dx_r, dy_r))

        records.append(dict(scan_i=si, scan_j=sj,
                            gt_dx_m=dx_r, gt_dy_m=dy_r, gt_dyaw_deg=dyaw,
                            trans_err_m=trans_err, rot_err_deg=rot_err))

    df = pd.DataFrame.from_records(records)
    df.to_csv(os.path.join(args.outdir, 'errors_debug.csv'), index=False)
    print(f"done – wrote {len(df)} rows and PNGs to {args.outdir}")


if __name__ == '__main__':
    main()
