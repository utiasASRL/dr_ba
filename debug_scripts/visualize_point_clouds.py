import numpy as np
import os
import polyscope as ps

kSeqId = 'boreas-2024-12-03-10-24'


def main(seq_id, iter = 10):

    # Get the list of npy files in output/<seq_id>/scans
    scan_path = f'output/{seq_id}/scans/'
    scan_files = [f for f in os.listdir(scan_path) if f.endswith('.npy')]

    # Random shuffle the list
    np.random.shuffle(scan_files)

    ps.init()

    for i in range(iter):
        # Load the scan
        scan_file = scan_files[i]
        scan = np.load(os.path.join(scan_path, scan_file))

        pts = np.array([scan[:, 0], scan[:, 1], np.zeros(scan.shape[0])]).T
        intensities = scan[:, 2]


        ps_cloud = ps.register_point_cloud(f'scan_{i}', pts)
        ps_cloud.add_scalar_quantity("intensities", intensities, enabled=True)

        # Add a big red sphere at origin, and a green sphere at (5,0,0)
        ps.register_point_cloud(f'origin_{i}', np.array([[0, 0, 0]]), radius=0.01, color=(1.0, 0.0, 0.0))
        ps.register_point_cloud(f'x_axis_{i}', np.array([[5, 0, 0]]), radius=0.01, color=(0.0, 1.0, 0.0))

        ps.show()

        # Clear all the scene
        ps.remove_all_structures()
        
        
    
    




if __name__ == "__main__":
    main(kSeqId)