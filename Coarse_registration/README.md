# Coarse Registration

Scripts in this folder **(i) convert high-rate Applanix GPS poses into the radar-image frame and (ii) compute pre-optimization error metrics for SIFT-based coarse registration**.  It outputs per-pair translation and rotation errors, and side-by-side visualisations.

## Steps
```bash
# step 1: convert Applanix poses to radar frame

python gps_to_radar.py \
  --calib_dir   /path/to/<sequence>/calib \
  --gps_csv     /path/to/<sequence>/applanix/gps_post_process.csv \
  --out_csv     radar_frame_poses.csv


# step 2: evaluate SIFT registration
python compare_est_gt.py \
  --loops   ../RaPlace/PYTHON/loop_pairs.csv \
  --imgdir  ../local_maps \
  --gt      radar_frame_poses.csv \
  --outdir  vis        # optional; defaults to vis_gt_vs_sift_with_error_his
```

## Scripts Reference
**gps_to_radar.py**

What it does:
1. Reads each 200 Hz Applanix record `(easting, northing, altitude, roll, pitch, heading)`.
2. Builds `T_ENU←Applanix`, then multiplies by calibration `T_A←R` to obtain `T_ENU←Radar`.
3. Extracts `(x, y, yaw)` in radar image axes (right-handed, x = right, y = up).

**compare_est_gt.py**

What it does:
1. Applies SIFT + ratio-test + RANSAC feature matching on image-center-normalised points.
2. Converts pixel affine to SE(2) in meters using `--res`.
3. Linearly interpolates radar-frame GT for each timestamp, then builds centered warp. For a 5 ms gap, straight-line interpolation is mathematically adequate and avoids introducing new assumptions. 
4. Outputs `trans_err_m`, the Euclidean difference in (x, y) translations, and `rot_err_deg`, the minimal signed yaw difference, and the overlay of the two scans using GT and SIFT. 

`errors_debug.csv` and the corresponding PNG overlays give the baseline translation/rotation errors that the DRO post-registration should reduce.
