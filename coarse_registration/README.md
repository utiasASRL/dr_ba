# Coarse Registration

**TO BE UPDATED !!!**


**coarse_registration.py**

What it does:
1. Applies SIFT + ratio-test + RANSAC feature matching on image-center-normalised points.
2. Converts pixel affine to SE(2) in meters using `--res`.
3. Linearly interpolates radar-frame GT for each timestamp, then builds centered warp. For a 5 ms gap, straight-line interpolation is mathematically adequate and avoids introducing new assumptions. 
4. Outputs `trans_err_m`, the Euclidean difference in (x, y) translations, and `rot_err_deg`, the minimal signed yaw difference, and the overlay of the two scans using GT and SIFT. 

`errors_debug.csv` and the corresponding PNG overlays give the baseline translation/rotation errors that the DRO post-registration should reduce.
