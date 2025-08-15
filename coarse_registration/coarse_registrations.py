import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import utils

import yaml
import numpy as np
import pandas as pd
import cv2


def main():
    # Get the data directory
    output_path = utils.getOutputDataDir()
    local_map_path = os.path.join(output_path, "local_maps")

    # Load coarse registrations parameters
    opts = yaml.safe_load(open(os.path.join("coarse_registration", "config.yaml"), 'r'))

    # Get the pixel resolution
    pix_res = utils.getPixelResolution()


    # Read the RaPlace matches
    raw_loops = pd.read_csv(os.path.join(output_path, "raplace_loops.csv"))
    print("Loaded", len(raw_loops), "RaPlace matches.")


    # Store the valid matches
    valid_matches = []


    # Create the cv tools for feature extraction and matching
    # nfeatures=0 means no limit (default), set to a specific number to limit features
    sift_extractor = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.02, edgeThreshold=20, sigma=2.5)
    sift_matcher = cv2.BFMatcher()

    # Loop through the matches
    for index, row in raw_loops.iterrows():
        # Read the images
        img1 = cv2.imread(os.path.join(local_map_path, row['scan_i_name']), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(os.path.join(local_map_path, row['scan_j_name']), cv2.IMREAD_GRAYSCALE)

        ## Perform histogram equalization
        #img1 = cv2.equalizeHist(img1)
        #img2 = cv2.equalizeHist(img2)

        # Extract features
        kp1, des1 = sift_extractor.detectAndCompute(img1, None)
        kp2, des2 = sift_extractor.detectAndCompute(img2, None)
        if des1 is None or des2 is None:
            print(f"Skipping match {index} due to missing descriptors.")
            continue

        # Match features
        matches = sift_matcher.knnMatch(des1, des2, k=2)

        # Apply simple rules to remove bad matches
        good_matches = []
        for m, n in matches:
            # Check if the match features have the same scale
            #if np.abs((kp1[m.queryIdx].octave & 255)/(kp2[n.trainIdx].octave & 255) -1) > 0.2:
            if (kp1[m.queryIdx].octave & 255) != (kp2[n.trainIdx].octave & 255):
                continue

            # Apply ratio test
            if m.distance < opts['lowe_ratio'] * n.distance:
                good_matches.append(m)
        if len(good_matches) < 4:
            print(f"Skipping match {index} due to insufficient good matches.")
            continue

        # Print the number of good matches
        print(f"Match {index} has {len(good_matches)} good matches.")



        # Perform RANSAC registration
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=opts['ransac_thr'])
        if M is None:
            print(f"Skipping match {index} due to failed registration.")
            continue

        # Get the pose and scale from the transformation matrix
        pose, scale = utils.affineToPoseAndScale(M, pix_res, img1.shape)


        # Reject if the scale is too far from 1
        if np.abs(scale - 1) > 0.05:
            print(f"Skipping match {index} due to scale {scale:.3f} being too far from 1.")
            continue


        # Add the valid match
        xy, theta = utils.poseToXYTheta(pose)
        valid_matches.append({
            'scan_i_name': row['scan_i_name'],
            'scan_j_name': row['scan_j_name'],
            'x': xy[0],
            'y': xy[1],
            'theta': theta
        })

        # Show the registration result
        img1_reg = cv2.warpAffine(img1, M, (img2.shape[1], img2.shape[0]))
        cv2.imshow("Match", np.hstack((img1_reg, img2)))
        cv2.waitKey(0)
        
    cv2.destroyAllWindows()


    # Save the valid matches
    valid_matches_df = pd.DataFrame(valid_matches)
    valid_matches_df.to_csv(os.path.join(output_path, "coarse_registrations.csv"), index=False)


if __name__ == "__main__":
    main()