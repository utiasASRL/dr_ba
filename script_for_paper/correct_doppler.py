import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import utils
import pyboreas as pb
import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')

kDataPath = "/media/ced/Extreme Pro/data/boreas/rss/test"

kVelToShift = 2.3070*0.5

kSubPathForOutput = "doppler_corrected_radar"

def main():

    dataset = pb.BoreasDataset(kDataPath)

    for seq in dataset.sequences:
        print("Root path:", seq.seq_root)
        if not (utils.getSeqType(seq.ID) in ['Glenshield', 'Commercial', 'Skyway']):
            print("Skip sequence type", utils.getSeqType(seq.ID))
            continue

        # Create a doppler_corrected_radar folder in seq.seq_root (remove if exists)
        corrected_folder = os.path.join(seq.seq_root, kSubPathForOutput)
        if os.path.exists(corrected_folder):
            temp_folder = corrected_folder.replace(" ", "\\ ")
            # Replace spaces with '\ '
            os.system("rm -r " + temp_folder)
        os.mkdir(corrected_folder)

        for i in range(len(seq.radar_frames)):
            radar_frame = seq.get_radar(i)
            # Print the x y body velocities
            print('Frame', i, ' / ', len(seq.radar_frames), end='\r')

            azimuths = radar_frame.azimuths.squeeze()

            dirs = np.empty((len(azimuths), 2))
            dirs[:, 0] = np.cos(azimuths)
            dirs[:, 1] = np.sin(azimuths)
            vec_bin_shift = kVelToShift * dirs @ (radar_frame.body_rate[:2]).reshape(2, 1)
            vec_bin_shift[::2] = -vec_bin_shift[::2]

            # Load the raw doppler image
            raw_doppler_img = cv2.imread(radar_frame.path, cv2.IMREAD_GRAYSCALE)
            up_chirp = raw_doppler_img[0,10]
            if up_chirp == 255:
                vec_bin_shift = -vec_bin_shift

            polar_img = raw_doppler_img[:,11:].astype(np.float32) / 255.0

            polar_shifted = np.zeros_like(polar_img)
            # Perform the shift
            for j in range(polar_img.shape[0]):
                polar_shifted[j] = np.roll(polar_img[j], np.round(vec_bin_shift[j]))

            ## Display in the same window the raw and corrected images in cartesian
            #raw_cart = pb.utils.radar.radar_polar_to_cartesian(azimuths, polar_img.astype(np.float32), radar_frame.resolution, 0.25, 800, False, True)
            #corrected_cart = pb.utils.radar.radar_polar_to_cartesian(azimuths, polar_shifted.astype(np.float32), radar_frame.resolution, 0.25, 800, False, True)

            #img_to_display = np.hstack((raw_cart, corrected_cart))
            #cv2.imshow("Doppler Correction", img_to_display)
            #cv2.waitKey(50)



            polar_shifted = (polar_shifted * 255.0).astype(np.uint8)

            output_data = raw_doppler_img[:, :11]
            output_data = np.concatenate((output_data, polar_shifted), axis=1)

            # Get the file name
            file_name = radar_frame.path.split('/')[-1]
            cv2.imwrite(os.path.join(corrected_folder, file_name), output_data)



            radar_frame.unload_data()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()