import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import utils
import numpy as np
import cv2
import pandas as pd
import gp_doppler


def main():

    # Read the coarse registrations
    out_path = utils.getOutputDataDir()
    coarse_registrations = pd.read_csv(os.path.join(out_path, "coarse_registrations.csv"))

    fine_registrations = []

    if coarse_registrations.empty:
        print("No coarse registrations found.")
    else: 
        for loop in coarse_registrations.itertuples():
            img_i = cv2.imread(os.path.join(out_path, "local_maps", loop.scan_i_name), cv2.IMREAD_GRAYSCALE)
            img_j = cv2.imread(os.path.join(out_path, "local_maps", loop.scan_j_name), cv2.IMREAD_GRAYSCALE)
            res = utils.getPixelResolution()

            if img_i is None or img_j is None:
                print(f"Skipping registration for {loop.scan_i_name} and {loop.scan_j_name} due to missing images.")
                continue

            # Resize the images
            img_i_small = cv2.resize(img_i, (img_i.shape[1]//4 + 1, img_i.shape[0]//4 + 1))
            img_j_small = cv2.resize(img_j, (img_j.shape[1]//4 + 1, img_j.shape[0]//4 + 1))
            res_small = res * 4
            # Add Gaussian blur to the images
            img_i_small = cv2.GaussianBlur(img_i_small, (5, 5), 0)
            img_j_small = cv2.GaussianBlur(img_j_small, (5, 5), 0)


            # Perform fine registration using "gp_doppler"
            local_map_registrator = gp_doppler.LocalMapRegistrator(img_j_small, img_i_small, res_small, np.array([loop.x, loop.y, loop.theta]))

            #local_map_registrator.displayOverlay()
            state = local_map_registrator.register(nb_iter=100, verbose=False, step_tol=1e-4)
            x, y, theta = state

            print("Scan ", loop.scan_i_name, " registered to ", loop.scan_j_name, "results (init) : ", x, "(", loop.x, ")", y, "(", loop.y, ")", theta, "(", loop.theta, ")")

            #local_map_registrator.displayOverlay()
            reg_score = local_map_registrator.getRegistrationScore()
            print("Registration score:", reg_score)
            if reg_score > 0.5:
                fine_registrations.append({'scan_i_name': loop.scan_i_name, 'scan_j_name': loop.scan_j_name, 'x': x, 'y': y, 'theta': theta})
            else:
                print(f"Fine registration failed for {loop.scan_i_name} and {loop.scan_j_name}.")

    # Save the fine registrations to a CSV file
    fine_registrations = pd.DataFrame(fine_registrations, columns=['scan_i_name', 'scan_j_name', 'x', 'y', 'theta'])
    fine_registrations.to_csv(os.path.join(out_path, "fine_registrations.csv"), index=False)
    print(f"Saved {len(fine_registrations)} fine registrations to fine_registrations.csv")










if __name__ == "__main__":
    main()

    
    
