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

    fine_registrations = pd.DataFrame(columns=['scan_i_name', 'scan_j_name', 'x', 'y', 'theta'])

    if coarse_registrations.empty:
        print("No coarse registrations found.")
    else: 
        for loop in coarse_registrations.itertuples():
            img_i = cv2.imread(os.path.join(out_path, "local_maps", loop.scan_i_name), cv2.IMREAD_GRAYSCALE)
            img_j = cv2.imread(os.path.join(out_path, "local_maps", loop.scan_j_name), cv2.IMREAD_GRAYSCALE)

            if img_i is None or img_j is None:
                print(f"Skipping registration for {loop.scan_i_name} and {loop.scan_j_name} due to missing images.")
                continue

            # Perform fine registration using "gp_doppler"
            local_map_registrator = gp_doppler.LocalMapRegistrator(img_i, img_j, utils.getPixelResolution(), np.array([loop.x, loop.y, loop.theta]))
            local_map_registrator.testCostFunctionGrad()

            success, x, y, theta = local_map_registrator.register()

            if success:
                fine_registrations = fine_registrations.append({'scan_i_name': loop.scan_i_name, 'scan_j_name': loop.scan_j_name, 'x': x, 'y': y, 'theta': theta}, ignore_index=True)
            else:
                print(f"Fine registration failed for {loop.scan_i_name} and {loop.scan_j_name}.")

        fine_registrations.to_csv(os.path.join(out_path, "fine_registrations.csv"), index=False)
        print(f"Saved {len(fine_registrations)} fine registrations to fine_registrations.csv")










if __name__ == "__main__":
    main()

    
    
