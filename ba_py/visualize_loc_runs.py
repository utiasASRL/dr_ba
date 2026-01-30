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
    print(ablation_root)
    for d in ablation_root.iterdir():
        print(d)
    run_dirs = sorted(
        (d for d in ablation_root.iterdir() if d.is_dir() and d.name.startswith("loc_")),
        key=lambda d: int(d.name.split("_")[1])
    )

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 8))

    labels = ['glen_1', 'glen_2', 'glen_3']

    avg_rmse_x = 0.0
    avg_rmse_y = 0.0
    avg_rmse_yaw = 0.0
    avg_rmse_translation = 0.0
    count = 0
    for run_dir in run_dirs:
        print(f"Processing {run_dir.name}...")
        cfg_path = run_dir / 'loc_config.yaml'
        if not cfg_path.exists():
            print(f"Skipping {run_dir.name} (no config found)")
            continue

        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

        # label = format_label(cfg, label_fields)
        label = labels[int(run_dir.name.split("_")[1]) - 1]

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

        # Check that sequence didnt diverge
        if (x_errs[-1]**2 + y_errs[-1]**2) > 5.0**2:
            print(f"  Sequence {run_dir.name} diverged! Ignoring it for RMSE computations")
        else:
            print("RMSE (m), (m), (deg):")
            x_errs = np.array(x_errs)
            y_errs = np.array(y_errs)
            yaw_errs = np.array(yaw_errs)
            rmse_x = np.sqrt(np.mean(x_errs**2))
            rmse_y = np.sqrt(np.mean(y_errs**2))
            rmse_yaw = np.sqrt(np.mean(yaw_errs**2))
            print(f"{rmse_x:.3f}, {rmse_y:.3f}, {rmse_yaw:.3f}")
            label += f" (RMSE: {rmse_x:.2f}m, {rmse_y:.2f}m, {rmse_yaw:.2f}°)"

            avg_rmse_x += rmse_x
            avg_rmse_y += rmse_y
            avg_rmse_yaw += rmse_yaw
            avg_rmse_translation += np.sqrt(rmse_x**2 + rmse_y**2)
            count += 1

        # ---- plotting ----
        axs[0].plot(frame_ids, x_errs, label=label)
        axs[1].plot(frame_ids, y_errs, label=label)
        axs[2].plot(frame_ids, yaw_errs, label=label)


    if count > 0:
        avg_rmse_x /= count
        avg_rmse_y /= count
        avg_rmse_yaw /= count
        avg_rmse_translation /= count
        print(f"Average RMSE over {count} runs:")
        print(f"  x: {avg_rmse_x:.3f} m")
        print(f"  y: {avg_rmse_y:.3f} m")
        print(f"  yaw: {avg_rmse_yaw:.3f} deg")
        print(f"  translation: {avg_rmse_translation:.3f} m")

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
        ablation_root="/home/dl/Documents/phd/dev/dr_ba/output/aa_paper_ablation/no_map_glen/loc_set/",
        # ablation_root="/home/dl/Documents/phd/dev/dr_ba/output/aaa_paper_results/loc/ba/boreas-2024-12-03-12-54",
        label_fields={
            "map.voxel_res": "vox",
            "input.dist_field_preproc": "DF-scan",
            "input.gauss_blur_sigma": "σ-scan",
            "mapping.dist_field_preproc": "DF-map",
            "mapping.gauss_blur_sigma": "σ-map",
        },
    )
