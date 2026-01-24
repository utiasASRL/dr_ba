import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import utils
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy.spatial.transform import Rotation as R
import seaborn as sns
import pyboreas as pb
import cv2
import dirty_doppler as gp_doppler
import torch

kTypeColors = { 
            "Commercial": "green",
            "Glenshield": "orange",
            "Skyway": "blue",
            }
kTypeLabels = {
            "Commercial": "Commercial",
            "Glenshield": "Suburbs",
            "Skyway": "Skyway",
            }

kInlierPosThr = 1.5
kInlierRotThr = 1.0

def main():
    os.makedirs('figures', exist_ok=True)
    #plotLoopRegistrations()
    #teaserDataSample(frame_id=1150)
    #crossCorrelationExample()
    #plotPublicBoreas()

    #baOverlayFigure()
    #baOverlayFigure(seq_id='boreas-2024-12-04-11-45', tbv_path='/home/ced/Documents/data/boreas/for_tbv/TBV_Eval/boreas_tbv_model_8_2026-01-20_21-18/job_0')
    testOverlap(seq_id='boreas-2025-08-06-07-05', scan_ref = 1700, method='ba')
    testOverlap(seq_id='boreas-2025-08-06-07-05', scan_ref = 1700, method='gt')
    testOverlap(seq_id='boreas-2025-08-06-07-05', scan_ref = 1700, method='pogo')

    #plottrajectories()

    #timing()

def testOverlap(seq_id='boreas-2024-12-03-12-54', scan_ref = 1000, method='gt'):
    # Load the Dr-PoGO trajectory

    ref = scan_ref
    if method == 'gt':
        pogo_traj, times = utils.getPogoPosesAndTimes(seq_id, ouput_path='output')
        gt_traj, gt_times = utils.getGTRadarPosesAndTimes(seq_id)
        gt_times = gt_times*1e6
        traj = utils.getInterpolatedTrajectory(gt_traj, gt_times, times)
    elif method == 'pogo':
        traj, times = utils.getPogoPosesAndTimes(seq_id, ouput_path='output')
    elif method == 'ba':
        pogo_traj, times = utils.getPogoPosesAndTimes(seq_id, ouput_path='output')
        time_ref = times[scan_ref]
        traj, times = utils.getPogoPosesAndTimes(seq_id, ouput_path='output', file_name='ba_traj.csv', delimiter=',')
        # Find the closest time to time_ref
        ref = np.argmin(np.abs(times - time_ref))

    distances = [0]
    for i in range(1, traj.shape[0]):
        delta = traj[i, 0:3, 3] - traj[i-1, 0:3, 3]
        dist = np.linalg.norm(delta)
        distances.append(distances[-1] + dist)

    distances = np.array(distances)

    # Get the closest scan to scan_ref that is at least 300m apart in terms of travelled distance
    min_dist_gap = 300.0  # meters
    last_selected_range = 10000.0
    for i in range(traj.shape[0]):
        if np.abs(distances[i] - distances[ref]) < min_dist_gap:
            continue
        dist = np.linalg.norm(traj[i, 0:3, 3] - traj[ref, 0:3, 3])
        if dist < last_selected_range:
            last_selected_range = dist
            min_j = i
    print(f"Selected scan {min_j} at distance {last_selected_range} m from scan {ref}")
    os.makedirs('figures/overlays', exist_ok=True)

    score = overlapScans(traj[ref], times[ref], traj[min_j], times[min_j], scan_path=os.path.join('output', seq_id, 'local_maps'), labels=[f'Scan {ref}', f'Scan {min_j}'], visualize=True, output_path='figures/overlays/' + f'{method}_overlay_{seq_id}_scan_{ref}_{min_j}.pdf')
        






def baOverlayFigure(seq_id='boreas-2024-12-03-12-54', tbv_path='/home/ced/Documents/data/boreas/for_tbv/TBV_Eval/boreas_tbv_model_8_2026-01-20_08-06/job_0'):

    # Load the Dr-PoGO trajectory
    # Get the BA trajectory
    ba_traj, ba_times = utils.getPogoPosesAndTimes(seq_id, ouput_path='output', file_name='ba_traj.csv', delimiter=',')
    pogo_traj, pogo_times = utils.getPogoPosesAndTimes(seq_id, ouput_path='output')

    pogo_traj = utils.getInterpolatedTrajectory(pogo_traj, pogo_times, ba_times)
    pogo_times = ba_times

    # Get the TBV trajectory
    tbv_traj, tbv_times, _ = utils.readTBV2DTraj(tbv_path)
    # Interpolate to pogo times
    tbv_traj_interp = utils.getInterpolatedTrajectory(tbv_traj, tbv_times*1e6, pogo_times)

    distances = [0]
    for i in range(1, pogo_traj.shape[0]):
        delta = pogo_traj[i, 0:3, 3] - pogo_traj[i-1, 0:3, 3]
        dist = np.linalg.norm(delta)
        distances.append(distances[-1] + dist)
    distances = np.array(distances)

    min_dist_gap = 300.0  # meters
    max_range = 50.0  # meters
    min_step = 10  # meters
    last_selected_dist = -10000.0
    scan_pairs = []
    for i in range(pogo_traj.shape[0]):
        if distances[i] - last_selected_dist < min_step:
            continue
        min_scan_dist = 2*min_dist_gap
        for j in range(i + 1, pogo_traj.shape[0]):
            if distances[j] - distances[i] < min_dist_gap:
                continue
            dist_ij = np.linalg.norm(pogo_traj[i, 0:3, 3] - pogo_traj[j, 0:3, 3])
            if dist_ij < min_scan_dist:
                min_scan_dist = dist_ij
                min_j = j
        if min_scan_dist < max_range:
            scan_pairs.append((i, min_j))
            last_selected_dist = distances[i]


    # Scores for each pair
    scores = []
    for (i, j) in scan_pairs:
        scores.append(overlapScans(pogo_traj[i], pogo_times[i], pogo_traj[j], pogo_times[j], scan_path=os.path.join('output', seq_id, 'local_maps'), labels=[f'Scan {i}', f'Scan {j}']))

    # Get the n lowest scores that are at least 50m apart and display their overlays
    distance_threshold = 50.0  # meters
    n = 10
    sorted_indices = np.argsort(scores)
    selected_indices = []
    selected_positions = []
    for idx in sorted_indices:
        i, j = scan_pairs[idx]
        pos_i = pogo_traj[i, 0:2, 3]
        pos_j = pogo_traj[j, 0:2, 3]
        too_close = False
        for pos in selected_positions:
            if np.linalg.norm(pos - pos_i) < distance_threshold or np.linalg.norm(pos - pos_j) < distance_threshold:
                too_close = True
                break
        if not too_close:
            selected_indices.append(idx)
            selected_positions.append(pos_i)
            selected_positions.append(pos_j)
        if len(selected_indices) >= n:
            break


    # Empty the overlays folder
    overlay_path = os.path.join('figures', 'overlays', seq_id)
    os.makedirs(overlay_path, exist_ok=True)
    if os.path.exists(overlay_path):
        for file in os.listdir(overlay_path):
            os.remove(os.path.join(overlay_path, file))
    scan_path = os.path.join('output', seq_id, 'local_maps')
    for idx in selected_indices:
        i, j = scan_pairs[idx]
        output_path = os.path.join(overlay_path, f'pogo_overlay_scan_{i}_{j}.pdf')
        overlapScans(pogo_traj[i], pogo_times[i], pogo_traj[j], pogo_times[j], scan_path=scan_path, output_path=output_path, labels=[f'Scan {i}', f'Scan {j}'])

        output_path = os.path.join(overlay_path, f'tbv_overlay_scan_{i}_{j}.pdf')
        overlapScans(tbv_traj_interp[i], pogo_times[i], tbv_traj_interp[j], pogo_times[j], scan_path=scan_path, output_path=output_path, labels=[f'Scan {i}', f'Scan {j}'])

        output_path = os.path.join(overlay_path, f'ba_overlay_scan_{i}_{j}.pdf')
        overlapScans(ba_traj[i], pogo_times[i], ba_traj[j], pogo_times[j], scan_path=scan_path, output_path=output_path, labels=[f'Scan {i}', f'Scan {j}'])



    # Plot the scores a link colormap on the trajectory
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(pogo_traj[:, 0, 3], -pogo_traj[:, 1, 3], 'blue', label='PoGO', linewidth=1)
    norm = matplotlib.colors.Normalize(vmin=min(scores), vmax=max(scores))
    cmap = plt.get_cmap('hot')
    for idx, (i, j) in enumerate(scan_pairs):
        color = cmap(norm(scores[idx]))
        ax.plot([pogo_traj[i, 0, 3], pogo_traj[j, 0, 3]], [-pogo_traj[i, 1, 3], -pogo_traj[j, 1, 3]], color=color, linewidth=1.5)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Registration Score', rotation=270, labelpad=15)
    ax.set_title('Dr-PoGO Loop Closure Candidates with Registration Scores', {'fontweight': 'bold'})
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.axis('equal')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()





def overlapScans(pose1, time1, pose2, time2, scan_path, output_path = None, labels = None, visualize=False):
    res = utils.getPixelResolution()
    scan1 = cv2.imread(os.path.join(scan_path, utils.timeToName(time1)), cv2.IMREAD_GRAYSCALE)
    scan2 = cv2.imread(os.path.join(scan_path, utils.timeToName(time2)), cv2.IMREAD_GRAYSCALE)

    rel_pose = np.linalg.inv(pose1) @ pose2

    xy, theta = utils.poseToXYTheta(rel_pose)

    gp_doppler_reg = gp_doppler.LocalMapRegistrator(scan2, scan1, res, np.array([xy[0], xy[1], theta]))
    overlay = gp_doppler_reg.getOverlay()
    
    score = gp_doppler_reg.getRegistrationScore().detach().cpu().item()

    print(f"Registration score: {score}")


    if output_path is not None or visualize:
        print(f"Saving overlay to {output_path}...")
        print("Relative pose:", xy, np.rad2deg(theta))
        plt.figure(figsize=(12,12))
        plt.imshow(scan1, cmap='gray')
        plt.imshow(overlay, cmap='hot', alpha=0.7)
        if labels is not None:
            plt.title(f"{labels[0]} & {labels[1]}\nScore: {score:.2f}", {'fontweight': 'bold'})
        plt.axis('off')
        plt.tight_layout()
        if output_path is not None:
            plt.savefig(output_path)
        if visualize:
            plt.show()
        plt.close()

    return score


    #fig, ax = plt.subplots(1,2, figsize=(10,5))
    #ax[0].imshow(scan1, cmap='gray')
    #ax[1].imshow(scan2, cmap='gray')
    #if labels is not None:
    #    ax[0].set_title(labels[0])
    #    ax[1].set_title(labels[1])
    #plt.show()
    


def timing():
    result_folder = "output_dr_pogo_incremental"

    list_of_dirs = os.listdir(result_folder)

    dro_total_time = 0
    total_nb_frames = 0
    raplace_total_time = 0
    raplace_nb_queries = 0
    coarse_total_time = 0
    coarse_nb_queries = 0
    fine_total_time = []
    fine_nb_queries = []
    pogo_total_time = 0
    pogo_nb_queries = 0

    for dir_name in list_of_dirs:
        nb_frames = len(os.listdir(os.path.join(result_folder, dir_name, "local_maps")))
        total_nb_frames += nb_frames

        # Read the DRO time
        dro_time = np.loadtxt(os.path.join(result_folder, dir_name, "other_log/avg_time.txt"), delimiter=',', skiprows=1)
        dro_total_time += dro_time * nb_frames

        # Read the raplace time
        raplace_time_raw = np.loadtxt(os.path.join(result_folder, dir_name, "raplace_time.txt"), delimiter=',', skiprows=1)
        raplace_total_time += raplace_time_raw[0]
        raplace_nb_queries += raplace_time_raw[1]

        # Read the coarse registration time
        coarse_time_raw = np.loadtxt(os.path.join(result_folder, dir_name, "coarse_registration_time.txt"), delimiter=',', skiprows=1)
        coarse_total_time += coarse_time_raw[0]
        coarse_nb_queries += coarse_time_raw[1]

        # Read the fine registration time
        fine_time_raw = np.loadtxt(os.path.join(result_folder, dir_name, "fine_registration_time.txt"), delimiter=',', skiprows=1)
        fine_total_time.append(fine_time_raw[0])
        fine_nb_queries.append(fine_time_raw[1])

        # Read the pogo time
        pogo_time_raw = np.loadtxt(os.path.join(result_folder, dir_name, "pogo_time.txt"), delimiter=',')
        pogo_total_time += pogo_time_raw
        pogo_nb_queries += fine_time_raw[2]


    print("DRO average time per frame:", dro_total_time / total_nb_frames, "s")
    print("Raplace average time per query:", raplace_total_time / raplace_nb_queries, "s,   per frame:", raplace_total_time / total_nb_frames, "s")
    print("Coarse registration average time per query:", coarse_total_time / coarse_nb_queries, "s,   per frame:", coarse_total_time / total_nb_frames, "s")

    X = np.zeros((len(fine_total_time),2))
    b = np.zeros((len(fine_total_time),1))

    X[:,0] = np.array(fine_nb_queries)
    X[:,1] = 1
    b[:,0] = np.array(fine_total_time)

    w, _, _, _ = np.linalg.lstsq(X, b, rcond=None)
    print("Fine registration average time per query:", w[0,0], "s,   overhead:", w[1,0], "s,   per frame:", np.sum(fine_total_time) / total_nb_frames, "s")

    print("PoGO average time per query:", pogo_total_time / pogo_nb_queries, "s,   per frame:", pogo_total_time / total_nb_frames, "s")










def plottrajectories():
    sequences = ['boreas-2024-12-03-12-54', 'boreas-2024-12-23-17-18', 'boreas-2024-12-04-12-08']
    labels = ['Suburbs', 'Commercial', 'Skyway']
    rotations = [0, -60, 20] # Degrees
    zoom_regions = [(-150, 150, -150, 200), (-100, 100, -100, 100), (-200, 400, -100, 800)]
    zoom_regions_2 = [(-350, -100, 2050, 2300), (-520, -460, -1300, -1220), (3300, 3450, 3900, 4100)]
    insert_positions = [(0.05, 0.05, 0.35), (0.64, 0.45, 0.35), (0.6, 0.1, 0.35)]
    insert_positions_2 = [(0.65, 0.65, 0.3), (0.01, 0.01, 0.3), (0.01, 0.55, 0.3)]
    insert_locs = [(1,3), (1,3), (2,3)] # loc1, loc2 for mark_inset
    insert_locs_2 = [(1,3), (1,4), (2,4)] # loc1, loc2 for mark_inset

    fig, ax = plt.subplots(3, 1, figsize=(5, 13))
    for i, seq_id in enumerate(sequences):
        # Load the estimated poses
        dro_traj, times = utils.getDroPosesAndTimes(seq_id, ouput_path='output_dr_pogo')
        pogo_traj, times = utils.getPogoPosesAndTimes(seq_id, ouput_path='output_dr_pogo')

        navtech_traj, navtech_times = utils.readNavtechSLAM2DTraj(searchNavtechPath(seq_id))
        navtech_traj = np.linalg.inv(navtech_traj[0,:,:]) @ navtech_traj
        tbv_traj, tbv_times, _ = utils.readTBV2DTraj(searchTBVPath(seq_id))
        tbv_traj = np.linalg.inv(tbv_traj[0,:,:]) @ tbv_traj
        tbv_traj = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) @ tbv_traj


        gt_traj, gt_times = utils.getGTRadarPosesAndTimes(seq_id)
        gt_traj = np.linalg.inv(gt_traj[0,:,:]) @ gt_traj
        times = times * 1e-6

        ang_rot = np.deg2rad(rotations[i])
        rot_mat = np.array([[np.cos(ang_rot), -np.sin(ang_rot), 0, 0],
                            [np.sin(ang_rot), np.cos(ang_rot), 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        gt_traj = rot_mat @ gt_traj


        dro_aligned, _ ,_ = utils.align2DTrajectories(gt_traj, gt_times, dro_traj, times)
        pogo_aligned, _ ,_ = utils.align2DTrajectories(gt_traj, gt_times, pogo_traj, times)
        navtech_aligned, _ ,_ = utils.align2DTrajectories(gt_traj, gt_times, navtech_traj, navtech_times)
        tbv_aligned, _ ,_ = utils.align2DTrajectories(gt_traj, gt_times, tbv_traj, tbv_times)

        ax[i].plot(navtech_aligned[:, 0], -navtech_aligned[:, 1], 'lightgreen', label='Navtech-SLAM', linewidth=1)
        ax[i].plot(dro_aligned[:, 0], -dro_aligned[:, 1], 'orange', label='DRO', linewidth=1)
        ax[i].plot(tbv_aligned[:, 0], -tbv_aligned[:, 1], 'fuchsia', label='TBV-SLAM', linewidth=1)
        ax[i].plot(pogo_aligned[:, 0], -pogo_aligned[:, 1], 'blue', label='PoGO', linewidth=1)
        ax[i].plot(gt_traj[:, 0, 3], -gt_traj[:, 1, 3], 'r--', label='Groundtruth', linewidth=1)

        ## Insert a zoomed-in region
        #x_min, x_max, y_min, y_max = zoom_regions[i]
        #axin = ax[i].inset_axes(insert_positions[i], xlim=(x_min, x_max), ylim=(y_min, y_max))
        ##axin.plot(navtech_aligned[:, 0], -navtech_aligned[:, 1], 'lightgreen', linewidth=1)
        ##axin.plot(dro_aligned[:, 0], -dro_aligned[:, 1], 'orange', linewidth=1)
        ##axin.plot(tbv_aligned[:, 0], -tbv_aligned[:, 1], 'fuchsia', linewidth=1)
        ##axin.plot(pogo_aligned[:, 0], -pogo_aligned[:, 1], 'blue', linewidth=1)
        ##axin.plot(gt_traj[:, 0, 3], -gt_traj[:, 1, 3], 'r--', linewidth=1)
        #axin.axis('equal')
        #axin.set_xticks([])
        #axin.set_yticks([])
        #ax[i].indicate_inset_zoom(axin, edgecolor="black")

        #axin = inset_axes(ax[i], width="35%", height="35%", loc='upper right')
        ratio_x = zoom_regions[i][1] - zoom_regions[i][0]
        ratio_y = zoom_regions[i][3] - zoom_regions[i][2]
        axin = ax[i].inset_axes([insert_positions[i][0], insert_positions[i][1], insert_positions[i][2], insert_positions[i][2]*ratio_y/ratio_x])
        axin.plot(navtech_aligned[:, 0], -navtech_aligned[:, 1], 'lightgreen', linewidth=1)
        axin.plot(dro_aligned[:, 0], -dro_aligned[:, 1], 'orange', linewidth=1)
        axin.plot(tbv_aligned[:, 0], -tbv_aligned[:, 1], 'fuchsia', linewidth=1)
        axin.plot(pogo_aligned[:, 0], -pogo_aligned[:, 1], 'blue', linewidth=1)
        axin.plot(gt_traj[:, 0, 3], -gt_traj[:, 1, 3], 'r--', linewidth=1)
        x_min, x_max, y_min, y_max = zoom_regions[i]
        axin.set_xlim(x_min, x_max)
        axin.set_ylim(y_min, y_max)
        axin.set_xticks([])
        axin.set_yticks([])

        mark_inset(ax[i], axin, loc1=insert_locs[i][0], loc2=insert_locs[i][1], fc="none", ec="0.5", ls='--')
        #ax[i].indicate_inset_zoom(axin, edgecolor="black")

        ratio_x = zoom_regions_2[i][1] - zoom_regions_2[i][0]
        ratio_y = zoom_regions_2[i][3] - zoom_regions_2[i][2]
        axin2 = ax[i].inset_axes([insert_positions_2[i][0], insert_positions_2[i][1], insert_positions_2[i][2], insert_positions_2[i][2]*ratio_y/ratio_x])
        axin2.plot(navtech_aligned[:, 0], -navtech_aligned[:, 1], 'lightgreen', linewidth=1)
        axin2.plot(dro_aligned[:, 0], -dro_aligned[:, 1], 'orange', linewidth=1)
        axin2.plot(tbv_aligned[:, 0], -tbv_aligned[:, 1], 'fuchsia', linewidth=1)
        axin2.plot(pogo_aligned[:, 0], -pogo_aligned[:, 1], 'blue', linewidth=1)
        axin2.plot(gt_traj[:, 0, 3], -gt_traj[:, 1, 3], 'r--', linewidth=1)
        x_min, x_max, y_min, y_max = zoom_regions_2[i]
        axin2.set_xlim(x_min, x_max)
        axin2.set_ylim(y_min, y_max)
        axin2.set_xticks([])
        axin2.set_yticks([])
        mark_inset(ax[i], axin2, loc1=insert_locs_2[i][0], loc2=insert_locs_2[i][1], fc="none", ec="0.5", ls='--')



        ax[i].set_title(labels[i], {'fontweight': 'bold'})
        ax[i].set_ylabel('Y [m]')
        ax[i].axis('equal')

    ax[-1].set_xlabel('X [m]')
    ax[0].legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('figures/trajectories_examples.pdf')
    plt.show()


def searchTBVPath(seq_id):
    base_path = '/home/ced/Documents/data/boreas/for_tbv/rss/TBV_Eval'
    list_of_dirs = os.listdir(base_path)
    path = None
    for dir_name in list_of_dirs:
        param_path = os.path.join(base_path, dir_name, "job_0/pars.txt")
        if not os.path.exists(param_path):
            continue
        # Read line by line to find the sequence ID
        with open(param_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if line.startswith("sequence, "):
                parts = line.split(", ")
                if len(parts) >= 2 and seq_id in parts[1]:
                    path = os.path.join(base_path, dir_name, "job_0")
                    break

    if path is None:
        raise ValueError(f"Could not find TBV path for sequence {seq_id}")
    return path

def searchNavtechPath(seq_id):
    base_path = 'output_navtech_slam_rss'
    list_of_files = os.listdir(base_path)
    path = None
    for file_name in list_of_files:
        if file_name.startswith(seq_id) and file_name.endswith('_pgo.csv'):
            path = os.path.join(base_path, file_name)
            break

    if path is None:
        raise ValueError(f"Could not find Navtech path for sequence {seq_id}")
    return path

def plotPublicBoreas():
    pogo_path = "output_public"
    methods = {
        'Navtech-SLAM': 'output_navtech_slam',
        'TBV-SLAM': 'output_tbv',
        'Dr-PoGO': 'output_public',
    }

    paths = os.listdir(pogo_path)
    paths.sort()
    errors_ate = {}
    errors_ete = {}
    for method, path in methods.items():
        errors_ate[method] = []
        errors_ete[method] = []
        for seq_id in paths:
            if not seq_id.startswith('boreas-'):
                continue
            print(f"Processing sequence {seq_id} for method {method}...")

            if method == 'Dr-PoGO':
                error_path = os.path.join(path, seq_id, seq_id + "_errors.csv")
            else:
                error_path = os.path.join(path, seq_id + "_errors.csv")
            errors_seq = np.loadtxt(error_path, delimiter=',', skiprows=1)
            errors_ate[method].append(errors_seq[0])
            errors_ete[method].append(errors_seq[1])

    avg_ate = {method: np.mean(errors) for method, errors in errors_ate.items()}
    avg_ete = {method: np.mean(errors) for method, errors in errors_ete.items()}

    # Plot the errors for each type
    fig, ax = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    for method, errors in errors_ate.items():
        ax[0].plot(errors, label=method + f" (Avg: {avg_ate[method]:.2f} m)")
    #ax[0].set_xlabel("Sequence ID")
    ax[0].set_ylabel("Absolute Trajectory Error [m]")
    ax[0].legend(loc='upper right')
    # Remove xtick labels for the first plot
    ax[0].set_xticklabels([])
    # Set x limits
    ax[0].set_xlim(0, len(errors_ate[method]) - 1)
    # Set y limits
    ax[0].set_ylim(np.min([min(e) for e in errors_ate.values()]), np.max([max(e) for e in errors_ate.values()])+2500)

    for method, errors in errors_ete.items():
        ax[1].plot(errors, label=method + f" (Avg: {avg_ete[method]:.2f} m)")
    ax[1].set_xlabel("Sequence ID")
    ax[1].set_ylabel("End-Pose Error [m]")
    ax[1].legend(loc='upper right')
    ax[1].set_ylim(np.min([min(e) for e in errors_ete.values()]), np.max([max(e) for e in errors_ete.values()])+9000)
    # Set x limits
    ax[1].set_xlim(0, len(errors_ete[method]) - 1)
    # Log scale for y axis
    ax[0].set_yscale('log')
    ax[0].grid(True, which="both", ls="--")
    ax[1].set_yscale('log')
    ax[1].grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(f"figures/public_boreas_errors.pdf")
    plt.show()


def crossCorrelationExample():
    seq_id = 'boreas-2024-12-04-11-56'
    local_map_path = 'output_dr_pogo'

    # Get list of files in local_map_path
    files = os.listdir(os.path.join(local_map_path, seq_id, 'local_maps'))
    files.sort()

    res = utils.getPixelResolution()

    

    scan_high = 250#800#1100
    scan_high_next = scan_high + 2
    scan_low = 800#50
    scan_low_next = scan_low + 2

    high_img = cv2.imread(os.path.join(local_map_path, seq_id, 'local_maps', files[scan_high]), cv2.IMREAD_GRAYSCALE)
    low_img = cv2.imread(os.path.join(local_map_path, seq_id, 'local_maps', files[scan_low]), cv2.IMREAD_GRAYSCALE)
    high_img_next = cv2.imread(os.path.join(local_map_path, seq_id, 'local_maps', files[scan_high_next]), cv2.IMREAD_GRAYSCALE)
    low_img_next = cv2.imread(os.path.join(local_map_path, seq_id, 'local_maps', files[scan_low_next]), cv2.IMREAD_GRAYSCALE)


    t_high = utils.nameToTime(files[scan_high])
    t_high_next = utils.nameToTime(files[scan_high_next])
    t_low = utils.nameToTime(files[scan_low])
    t_low_next = utils.nameToTime(files[scan_low_next])

    gt_poses, gt_times = utils.getGTRadarPosesAndTimes(seq_id)
    pose_high = utils.getInterpolatedPose(gt_poses, gt_times, t_high)
    pose_high_next = utils.getInterpolatedPose(gt_poses, gt_times, t_high_next)
    pose_low = utils.getInterpolatedPose(gt_poses, gt_times, t_low)
    pose_low_next = utils.getInterpolatedPose(gt_poses, gt_times, t_low_next)

    rel_pose_high = np.linalg.inv(pose_high) @ pose_high_next
    rel_pose_low = np.linalg.inv(pose_low) @ pose_low_next

    xy_high, theta_high = utils.poseToXYTheta(rel_pose_high)
    xytheta_high = np.array([xy_high[0], xy_high[1], theta_high])
    xy_low, theta_low = utils.poseToXYTheta(rel_pose_low)
    xytheta_low = np.array([xy_low[0], xy_low[1], theta_low])

    gp_doppler_low = gp_doppler.LocalMapRegistrator(low_img_next, low_img, res, xytheta_low)
    overlay_low = gp_doppler_low.getOverlay()
    low_score = gp_doppler_low.getRegistrationScore().detach().cpu().item()
    low_cross_corr = np.sum(gp_doppler_low.costFunctionAndJacobian(torch.tensor(xytheta_low, device=gp_doppler_low.device).float(), with_jac=False).detach().cpu().numpy())

    gp_doppler_high = gp_doppler.LocalMapRegistrator(high_img_next, high_img, res, xytheta_high)
    overlay_high = gp_doppler_high.getOverlay()
    high_score = gp_doppler_high.getRegistrationScore().detach().cpu().item()
    high_cross_corr = np.sum(gp_doppler_high.costFunctionAndJacobian(torch.tensor(xytheta_high, device=gp_doppler_high.device).float(), with_jac=False).detach().cpu().numpy())
    
    gp_doppler_high_shifted = gp_doppler.LocalMapRegistrator(high_img_next, high_img, res, np.array([xy_high[0]+5.0, xy_high[1]+5.0, theta_high]))
    shifted_overlay_high = gp_doppler_high_shifted.getOverlay()
    shifted_high_score = gp_doppler_high_shifted.getRegistrationScore().detach().cpu().item()
    shifted_high_cross_corr = np.sum(gp_doppler_high_shifted.costFunctionAndJacobian(torch.tensor(np.array([xy_high[0]+5.0, xy_high[1]+5.0, theta_high]), device=gp_doppler_high_shifted.device).float(), with_jac=False).detach().cpu().numpy())

    kTextStep = 225
    kTextX = 50
    kTextY = 50
    kFontSize = 14
    kAlpha = 1.0

    ncolors = 256
    color_array = plt.get_cmap('hot')(range(ncolors))
    color_array[:, -1] = np.linspace(0, 1, ncolors)
    hot_alpha = matplotlib.colors.LinearSegmentedColormap.from_list('hot_alpha', color_array)
    plt.colormaps.register(cmap=hot_alpha)

    crop_ratio = 0.25
    extent = (high_img.shape[1]*crop_ratio, high_img.shape[1]*(1-crop_ratio), high_img.shape[0]*crop_ratio, high_img.shape[0]*(1-crop_ratio))
    kTextX = extent[0] + kTextX * 2*crop_ratio
    kTextY = extent[2] + kTextY * 2*crop_ratio
    kTextStep = kTextStep * 2*crop_ratio

    print("High: score:", high_score, "cross-corr:", high_cross_corr)
    print("Low: score:", low_score, "cross-corr:", low_cross_corr)
    fig, ax = plt.subplots(1, 3, figsize=(9, 4))
    ax[0].imshow(low_img, cmap='gray')
    ax[0].imshow(overlay_low, cmap='hot_alpha', alpha=kAlpha)
    #ax[0].text(kTextX, kTextY, 'Scene A, good reg.', color='white', fontsize=kFontSize, va='top', fontname='DejaVu Sans')
    ax[0].set_xlim(extent[0], extent[1])
    ax[0].set_ylim(extent[3], extent[2])
    #ax[0].text(kTextX, kTextY + kTextStep, 'Good registration', color='lightgreen', fontsize=kFontSize, va='top')
    #ax[0].text(kTextX, kTextY + 1 * kTextStep, 'g=' + "{0:.2e}".format(low_cross_corr), color='white', fontsize=kFontSize, va='top', fontname='DejaVu Sans')
    #ax[0].text(kTextX, kTextY + 2 * kTextStep, 's=' + str(np.round(low_score, 2)), color='lightgreen', fontsize=kFontSize, va='top', fontname='DejaVu Sans')
    ax[1].imshow(high_img, cmap='gray')
    ax[1].imshow(overlay_high, cmap='hot_alpha', alpha=kAlpha)
    #ax[1].text(kTextX, kTextY, 'Scene B, good reg.', color='white', fontsize=kFontSize, va='top', fontname='DejaVu Sans')
    ax[1].set_xlim(extent[0], extent[1])
    ax[1].set_ylim(extent[3], extent[2])
    #ax[1].text(kTextX, kTextY + kTextStep, 'Good registration', color='lightgreen', fontsize=kFontSize, va='top')
    #ax[1].text(kTextX, kTextY + 1 * kTextStep, 'g=' + "{0:.2e}".format(high_cross_corr), color='white', fontsize=kFontSize, va='top', fontname='DejaVu Sans')
    #ax[1].text(kTextX, kTextY + 2 * kTextStep, 's=' + str(np.round(high_score, 2)), color='lightgreen', fontsize=kFontSize, va='top', fontname='DejaVu Sans')
    ax[2].imshow(high_img, cmap='gray')
    ax[2].imshow(shifted_overlay_high, cmap='hot_alpha', alpha=kAlpha)
    #ax[2].text(kTextX, kTextY, 'Scene B, poor reg.', color='white', fontsize=kFontSize, va='top', fontname='DejaVu Sans')
    #ax[2].text(kTextX, kTextY + kTextStep, 'Poor registration', color='lightcoral', fontsize=kFontSize, va='top')
    #ax[2].text(kTextX, kTextY + 1 * kTextStep, 'g=' + "{0:.2e}".format(shifted_high_cross_corr), color='white', fontsize=kFontSize, va='top', fontname='DejaVu Sans')
    #ax[2].text(kTextX, kTextY + 2 * kTextStep, 's=' + str(np.round(shifted_high_score, 2)), color='lightcoral', fontsize=kFontSize, va='top', fontname='DejaVu Sans')
    ax[2].set_xlim(extent[0], extent[1])
    ax[2].set_ylim(extent[3], extent[2])
    # Remove the axis
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    plt.tight_layout()
    plt.savefig('figures/cross_correlation_score_example.pdf')
    plt.show()
    



def teaserDataSample(frame_id=150):
    seq_id = 'boreas-2024-12-03-10-24'
    dataset = pb.BoreasDataset(utils.getDataDir(seq_id))
    seq = dataset.get_seq_from_ID(seq_id)
    radar_frame = seq.get_radar(frame_id)

    # Save the polar data
    cv2.imwrite('figures/teaser_radar_polar.png', (radar_frame.polar*255).astype(np.uint8))
    # Convert to Cartesian
    cart_img = pb.utils.radar.radar_polar_to_cartesian(radar_frame.azimuths, radar_frame.polar.astype(np.float32), radar_frame.resolution, 0.25, 800, False, True)
    cv2.imwrite('figures/teaser_radar_cartesian.png', (cart_img*255).astype(np.uint8))




def plotLoopRegistrations():
    # Get the folders in the output directory
    output_paths = os.listdir("output")

    errors_coarse = {}
    errors_fine = {}


    for seq_id in output_paths:
        if not seq_id.startswith('boreas-'):
            continue

        print(f"Processing sequence {seq_id}...")

        try:
            # Load the coarse registrations
            loops_coarse = pd.read_csv(os.path.join("output", seq_id, "coarse_registrations.csv"))
            loops_fine = pd.read_csv(os.path.join("output", seq_id, "fine_registrations.csv"))
        except:
            print(f"Skipping sequence {seq_id} due to missing coarse registrations.")
            continue

        # Get the GT radar poses
        gt_poses, gt_times = utils.getGTRadarPosesAndTimes(seq_id)

        seq_type =  utils.getSeqType(seq_id)


        temp_errors_coarse = getRegistrationErrors(loops_coarse, gt_poses, gt_times)
        temp_errors_fine = getRegistrationErrors(loops_fine, gt_poses, gt_times)
        


        if(seq_type not in errors_coarse):
            errors_coarse[seq_type] = []
            errors_fine[seq_type] = []
        errors_coarse[seq_type].append({seq_id: temp_errors_coarse})
        errors_fine[seq_type].append({seq_id: temp_errors_fine})



    # Sort the sequences per alphabetical order
    errors_coarse = {k: sorted(v, key=lambda x: list(x.keys())[0]) for k, v in errors_coarse.items()}
    errors_fine = {k: sorted(v, key=lambda x: list(x.keys())[0]) for k, v in errors_fine.items()}


    # Get the number of matches and inliers per type
    for seq_type in errors_coarse.keys():
        inlier_matches_coarse = []
        for errors in errors_coarse[seq_type]:
            for seq_id, err in errors.items():
                for e in err:
                    if e[0] < kInlierPosThr and e[1] < kInlierRotThr:
                        inlier_matches_coarse.append(e)
        inlier_matches_fine = []
        for errors in errors_fine[seq_type]:
            for seq_id, err in errors.items():
                for e in err:
                    if e[0] < kInlierPosThr and e[1] < kInlierRotThr:
                        inlier_matches_fine.append(e)
        num_seq = len(errors_coarse[seq_type])
        num_inliers_coarse = len(inlier_matches_coarse)
        num_inliers_fine = len(inlier_matches_fine)
        num_coarse = 0
        for errors in errors_coarse[seq_type]:
            for seq_id, err in errors.items():
                num_coarse += len(err)
        num_fine = 0
        for errors in errors_fine[seq_type]:
            for seq_id, err in errors.items():
                num_fine += len(err)
        coarse_rmse_pos = np.sqrt(np.mean([e[0]**2 for e in inlier_matches_coarse]))
        fine_rmse_pos = np.sqrt(np.mean([e[0]**2 for e in inlier_matches_fine]))
        coarse_rmse_rot = np.sqrt(np.mean([e[1]**2 for e in inlier_matches_coarse]))
        fine_rmse_rot = np.sqrt(np.mean([e[1]**2 for e in inlier_matches_fine]))
        print("\n\n====== Sequence type", seq_type, "======")
        print("    Coarse:")
        print("        ", num_coarse / num_seq, "matches,", num_inliers_coarse / num_seq, "inliers ( ratio:", num_inliers_coarse / num_coarse, ")")
        print("        RMSE (trans):", coarse_rmse_pos)
        print("        RMSE (rot):", coarse_rmse_rot)
        print("    Fine:")
        print("        ", num_fine / num_seq, "matches,", num_inliers_fine / num_seq, "inliers ( ratio:", num_inliers_fine / num_fine, ")")
        print("        RMSE (trans):", fine_rmse_pos)
        print("        RMSE (rot):", fine_rmse_rot)

    # Plot the errors for each type
    fig, ax = plt.subplots(2, 2, figsize=(6, 4))
    # Similar as before but with seaborn's stripplot
    all_coarse_pos = []
    all_coarse_rot = []
    all_fine_pos = []
    all_fine_rot = []
    for seq_type in ['Glenshield', 'Commercial', 'Skyway']:
        for errors in errors_coarse[seq_type]:
            for seq_id, err in errors.items():
                for e in err:
                    all_coarse_pos.append({
                        'seq_type': seq_type,
                        'seq_id': seq_id,
                        'trans_err': e[0],
                        'rot_err': e[1]
                    })
                    all_coarse_rot.append({
                        'seq_type': seq_type,
                        'seq_id': seq_id,
                        'rot_err': e[1]
                    })
        for errors in errors_fine[seq_type]:
            for seq_id, err in errors.items():
                for e in err:
                    all_fine_pos.append({
                        'seq_type': seq_type,
                        'seq_id': seq_id,
                        'trans_err': e[0],
                        'rot_err': e[1]
                    })
                    all_fine_rot.append({
                        'seq_type': seq_type,
                        'seq_id': seq_id,
                        'rot_err': e[1]
                    })

    # --- Plot with seaborn stripplot using hue for color ---
    sns.stripplot(data=pd.DataFrame(all_coarse_pos), x='seq_id', y='trans_err', hue='seq_type',
                  palette=kTypeColors, dodge=True, ax=ax[0, 0], alpha=0.5)
    sns.stripplot(data=pd.DataFrame(all_coarse_rot), x='seq_id', y='rot_err', hue='seq_type',
                  palette=kTypeColors, dodge=True, ax=ax[0, 1], alpha=0.5)
    sns.stripplot(data=pd.DataFrame(all_fine_pos), x='seq_id', y='trans_err', hue='seq_type',
                  palette=kTypeColors, dodge=True, ax=ax[1, 0], alpha=0.5)
    sns.stripplot(data=pd.DataFrame(all_fine_rot), x='seq_id', y='rot_err', hue='seq_type',
                  palette=kTypeColors, dodge=True, ax=ax[1, 1], alpha=0.5)
    seq_types = ['Glenshield', 'Commercial', 'Skyway']
    # Replace with the labels
    seqs_per_type = [len(errors_coarse[stype]) for stype in seq_types]
    seq_types = [kTypeLabels[stype] for stype in seq_types]

    # Compute the center position for each group
    group_centers = np.cumsum([0] + seqs_per_type)[:-1] + np.array(seqs_per_type) / 2 - 0.5

    # Set the x-ticks at the group centers and label them with the sequence type
    ax[0, 0].set_xticks(group_centers)
    ax[0, 0].set_xticklabels(seq_types, fontsize=9)
    ax[0, 1].set_xticks(group_centers)
    ax[0, 1].set_xticklabels(seq_types, fontsize=9)
    ax[1, 0].set_xticks(group_centers)
    ax[1, 0].set_xticklabels(seq_types, fontsize=9)
    ax[1, 1].set_xticks(group_centers)
    ax[1, 1].set_xticklabels(seq_types, fontsize=9)

    # Remove the legends
    ax[0, 0].legend_.remove()
    ax[0, 1].legend_.remove()
    ax[1, 0].legend_.remove()
    ax[1, 1].legend_.remove()

    ax[0, 0].set_title('Coarse reg. - Pos. error', {'fontweight': 'bold'})
    ax[0, 1].set_title('Coarse reg. - Rot. error', {'fontweight': 'bold'})
    ax[1, 0].set_title('Fine reg. - Pos. error', {'fontweight': 'bold'})
    ax[1, 1].set_title('Fine reg. - Rot. error', {'fontweight': 'bold'})
    ax[0, 0].set_ylabel('Pos. Error [m]')
    ax[0, 1].set_ylabel('Rot. Error [deg]')
    ax[1, 0].set_ylabel('Pos. Error [m]')
    ax[1, 1].set_ylabel('Rot. Error [deg]')
    ax[0, 0].set_xlabel('')
    ax[0, 1].set_xlabel('')
    ax[1, 0].set_xlabel('')
    ax[1, 1].set_xlabel('')
    plt.tight_layout()
    plt.savefig('figures/registration_errors_per_sequence.pdf')
    plt.show()


def getRegistrationErrors(loops, gt_poses, gt_times):
    errors = []
    for loop in loops.itertuples():
        time_i = utils.nameToTime(loop.scan_i_name)
        time_j = utils.nameToTime(loop.scan_j_name)

        pose_i = utils.getInterpolatedPose(gt_poses, gt_times, time_i)
        pose_j = utils.getInterpolatedPose(gt_poses, gt_times, time_j)

        # Compute the relative pose
        rel_pose_gt = np.linalg.inv(pose_i) @ pose_j
        xy = np.array([loop.x, loop.y])
        theta = loop.theta
        rel_pose_est = utils.XYThetaToPose(xy, theta)

        # Compute the error
        rel_pose_err = np.linalg.inv(rel_pose_est) @ rel_pose_gt
        trans_err = np.linalg.norm(rel_pose_err[:2, 3])
        rot_err = np.linalg.norm(R.from_matrix(rel_pose_err[:3, :3]).as_rotvec())*180.0/np.pi  # Convert to degrees
        errors.append((trans_err, rot_err))
    return errors


if __name__ == "__main__":
    main()