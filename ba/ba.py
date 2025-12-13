import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from pyboreas import BoreasDataset


kSeqId = 'boreas-2024-12-03-12-54'

class Map:
    def __init__(self, res=0.2):
        self.res = res
        self.voxels = {} # key: (x_idx, y_idx), value: intensity
        self.size = 0

    def index(self, x, y):
        x_idx = int(x / self.res)
        y_idx = int(y / self.res)
        return (x_idx, y_idx)
    
    def add_voxel(self, x, y, intensity):
        idx = self.index(x, y)
        if idx not in self.voxels:
            self.voxels[idx] = intensity
            self.size += 1

    def bilinear_interpolate(self, x, y):
        # Get voxel indices
        (a, b) = self.index(x, y)

        # Get intensities
        int_ab = self.voxels.get((a, b), 0)
        int_a1b = self.voxels.get((a + 1, b), 0)
        int_ab1 = self.voxels.get((a, b + 1), 0)
        int_a1b1 = self.voxels.get((a + 1, b + 1), 0)

        # If all surrounding voxels are empty, return 0
        if (int_ab == 0 and int_a1b == 0 and int_ab1 == 0 and int_a1b1 == 0):
            return 0
        
        # Get weights
        w0 = (1 - (x - a)/self.res) * (1 - (y - b)/self.res)
        w1 = (1 - (x - a)/self.res) * ((y - b)/self.res)
        w2 = ((x - a)/self.res) * (1 - (y - b)/self.res)
        w3 = ((x - a)/self.res) * ((y - b)/self.res)

        # Bilinear interpolation
        int_xy = (w0 * int_ab + w1 * int_ab1 + w2 * int_a1b + w3 * int_a1b1)

        return int_xy

    def get_intensity(self, x, y):
        idx = self.index(x, y)
        return self.voxels.get(idx, 0)




    def plot(self, save_path=None):
        if not self.voxels:
            return

        # --- convert sparse dict to dense raster ---
        keys = np.asarray(list(self.voxels.keys()), dtype=np.int32)
        vals = np.asarray(list(self.voxels.values()), dtype=np.float32)

        ix = keys[:, 0]
        iy = keys[:, 1]

        ix_min, ix_max = ix.min(), ix.max()
        iy_min, iy_max = iy.min(), iy.max()

        H = ix_max - ix_min + 1
        W = iy_max - iy_min + 1

        img = np.full((H, W), np.nan, dtype=np.float32)
        img[ix - ix_min, iy - iy_min] = vals

        # --- plotting ---
        fig, ax = plt.subplots(facecolor="black")
        ax.set_facecolor("black")

        im = ax.imshow(
            img.T,
            origin="lower",
            cmap="viridis",
            extent=[
                ix_min * self.res,
                (ix_max + 1) * self.res,
                iy_min * self.res,
                (iy_max + 1) * self.res,
            ],
            interpolation="nearest",
        )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Intensity", color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.get_yticklabels(), color="white")

        # Set x to be between -100 and +200
        ax.set_xlim(-100, 200)
        ax.set_ylim(-100, 150)


        ax.set_title("Voxel Map", color="white")
        ax.set_xlabel("X", color="white")
        ax.set_ylabel("Y", color="white")
        ax.tick_params(colors="white")
        ax.set_aspect("equal")

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="black")
        else:
            plt.show()
        plt.close(fig)

def main(seq_id):

    # Get the list of npy files in output/<seq_id>/scans
    scan_path = f'output/{seq_id}/scans/'
    data_path = '/home/dl/Documents/phd/data/boreas/'
    dataset = BoreasDataset(data_path, split=[[seq_id]])
    seq = dataset.sequences[0]  # Get the first sequence in the dataset

    voxel_img_output_path = f'output/{seq_id}/voxel_maps/'
    os.makedirs(voxel_img_output_path, exist_ok=True)

    sample_every_n = 15
    max_sample = 10

    # Form map
    frame_0_pose = seq.radar_frames[0].pose
    vox_map = Map(res=0.2)

    for idx, radar_frame in enumerate(seq.radar_frames):
        if idx % sample_every_n != 0:
            continue
        if idx // sample_every_n >= max_sample:
            break

        print(f'Processing frame {idx} / {len(seq.radar_frames)}')

        radar_frame.load_data()
        timestamp = radar_frame.timestamp_micro
        # TODO: should use same scan naming notation ideally... need to change DRO tho
        timestamp_scan = radar_frame.timestamps[0][0]
        radar_frame.unload_data()

        scan_file = f'{timestamp_scan}.npy'
        if not osp.exists(osp.join(scan_path, scan_file)):
            print(f'Scan file {scan_file} does not exist, skipping.')
            continue

        scan = np.load(osp.join(scan_path, f'{timestamp_scan}.npy'))

        # Transform scan to frame_0
        scan_pose = radar_frame.pose
        rel_pose = np.linalg.inv(frame_0_pose) @ scan_pose
        # print('Relative pose:\n', rel_pose)
        scan_hom = np.hstack((scan[:, :2], np.zeros((scan.shape[0], 1)), np.ones((scan.shape[0], 1)))).T  # (4, N)
        scan_transformed = (rel_pose @ scan_hom).T  # (N, 4)
        # Trim off fourth dimension
        scan_transformed = scan_transformed[:, :3]
        # Reload in intensities
        scan_transformed[:, 2] = scan[:, 2]

        for x, y, intensity in scan_transformed:
            if intensity > 0:
                vox_map.add_voxel(x, y, intensity)

        save_path = osp.join(voxel_img_output_path, f'{idx}.png')
        vox_map.plot(save_path)

    vox_map.plot()


if __name__ == '__main__':
    main(kSeqId)