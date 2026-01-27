import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp

# ----------------------------
# Config helpers
# ----------------------------

def get_by_path(cfg, key_path):
    keys = key_path.split(".")
    d = cfg
    for k in keys:
        d = d[k]
    return d


def format_label(cfg, label_fields):
    """
    Build a legend label from selected config fields.

    label_fields: dict[path -> short_name]
    """
    parts = []
    for path, name in label_fields.items():
        val = get_by_path(cfg, path)
        parts.append(f"{name}={val}")
    return ", ".join(parts)


# ----------------------------
# Run discovery
# ----------------------------

def find_runs(ablation_root):
    ablation_root = Path(ablation_root)
    return sorted(
        d for d in ablation_root.iterdir()
        if d.is_dir() and d.name.startswith("set_")
    )


# ----------------------------
# Visualization
# ----------------------------

def plot_ablation_comparison(
    ablation_root,
    label_fields
):
    ablation_root = Path(ablation_root)
    run_dirs = sorted(
        (d for d in ablation_root.iterdir() if d.is_dir() and d.name.startswith("map_loc_")),
        key=lambda d: int(d.name.split("_")[2])
    )

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 8))

    # labels = ['glen', 'skyway', 'farm', 'forest']

    for run_dir in run_dirs:
        print(f"Processing {run_dir.name}...")
        cfg_path = run_dir / 'map_loc_config.yaml'
        if not cfg_path.exists():
            print(f"Skipping {run_dir.name} (no config found)")
            continue

        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

        label = format_label(cfg, label_fields)
        # label = labels[int(run_dir.name.split("_")[2]) - 1]

        # map_id,scan_id,est_x,est_y,est_yaw,gt_x,gt_y,gt_yaw
        loc_result_path = osp.join(run_dir, "loc_results.csv")
        if not osp.exists(loc_result_path):
            print(f"Skipping {run_dir.name} (no loc_results.csv)")
            continue

        frame_ids = []
        x_errs = []
        y_errs = []
        yaw_errs = []
        scan_id_0 = None
        with open(loc_result_path, "r") as f:
            next(f)  # skip header
            for line in f:
                tokens = line.strip().split(",")
                if scan_id_0 is None:
                    scan_id_0 = int(tokens[1])

                scan_id = int(tokens[1]) - scan_id_0

                est_x = float(tokens[2])
                est_y = float(tokens[3])
                est_yaw = float(tokens[4]) * (180.0 / np.pi)

                gt_x = float(tokens[5])
                gt_y = float(tokens[6])
                gt_yaw = float(tokens[7]) * (180.0 / np.pi)

                # wrap yaw error to [-180, 180]
                yaw_err = est_yaw - gt_yaw
                if yaw_err > 180.0:
                    yaw_err -= 360.0
                elif yaw_err < -180.0:
                    yaw_err += 360.0

                frame_ids.append(scan_id)
                x_errs.append(est_x - gt_x)
                y_errs.append(est_y - gt_y)
                yaw_errs.append(yaw_err)

        print("RMSE (m), (m), (deg):")
        x_errs = np.array(x_errs)
        y_errs = np.array(y_errs)
        yaw_errs = np.array(yaw_errs)
        rmse_x = np.sqrt(np.mean(x_errs**2))
        rmse_y = np.sqrt(np.mean(y_errs**2))
        rmse_yaw = np.sqrt(np.mean(yaw_errs**2))
        print(f"{rmse_x:.3f}, {rmse_y:.3f}, {rmse_yaw:.3f}")
        label += f" (RMSE: {rmse_x:.2f}m, {rmse_y:.2f}m, {rmse_yaw:.2f}°)"

        # ---- plotting ----
        axs[0].plot(frame_ids, x_errs, label=label)
        axs[1].plot(frame_ids, y_errs, label=label)
        axs[2].plot(frame_ids, yaw_errs, label=label)



    # ---- formatting ----
    axs[0].set_ylabel("x error [m]")
    axs[1].set_ylabel("y error [m]")
    axs[2].set_ylabel("yaw error [deg]")
    axs[2].set_xlabel("frame")

    for ax in axs:
        ax.grid(True)

    # single shared legend
    axs[0].legend()

    plt.tight_layout()
    plt.show()



# ----------------------------
# Entry point
# ----------------------------

if __name__ == "__main__":
    plot_ablation_comparison(
        ablation_root="/home/dl/Documents/phd/dev/dr_ba/output/ablation_map_loc/set_50",
        label_fields={
            "map.voxel_res": "vox",
            "input.dist_field_preproc": "DF-scan",
            "input.gauss_blur_sigma": "σ-scan",
            "mapping.dist_field_preproc": "DF-map",
            "mapping.gauss_blur_sigma": "σ-map",
        },
    )
