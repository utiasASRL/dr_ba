import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys

# ----------------------------
# Config helpers
# ----------------------------

def get_by_path(cfg, key_path):
    """Retrieve nested config value via dotted path."""
    keys = key_path.split(".")
    d = cfg
    for k in keys:
        d = d[k]
    return d


def format_label(cfg, label_fields):
    """
    Build a legend label from selected config fields.

    Example:
      label_fields = {
          "input.gauss_blur_sigma": "σ",
          "input.dist_field_preproc": "DF",
      }
    """
    parts = []
    for path, name in label_fields.items():
        val = get_by_path(cfg, path)
        parts.append(f"{name}={val}")
    return ", ".join(parts)


# ----------------------------
# Data loading (placeholder)
# ----------------------------

def load_run_data(run_dir):
    """
    Load whatever data you want to visualize for a single run.

    Replace this with:
      - reading numpy arrays
      - parsing CSVs
      - loading trajectory errors
      - etc.
    """
    csv_path = run_dir / 'rmse_cost_history.csv'
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)

    # Check number of entries in each row
    if data.shape[1] == 5:
        cost = data[:, 0]
        ate = data[:, 1]
        rmse_x = data[:, 2]
        rmse_y = data[:, 3]
        rmse_yaw = data[:, 4]
    elif data.shape[1] == 6:
        cost = data[:, 0]
        ate = data[:, 1]
        epe = data[:, 2]
        rmse_x = data[:, 3]
        rmse_y = data[:, 4]
        rmse_yaw = data[:, 5]
    else:
        print("Unexpected number of columns in CSV file:", data.shape[1])
        sys.exit(1)


# ----------------------------
# Main visualization
# ----------------------------

def visualize_ablation(
    ablation_root,
    label_fields
):
    ablation_root = Path(ablation_root)
    run_dirs = sorted(
        d for d in ablation_root.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    )

    fig_epe, ax_epe = plt.subplots()
    fig_ate, ax_ate = plt.subplots()

    for run_dir in run_dirs:
        cfg_path = run_dir / 'ba_config.yaml'
        if not cfg_path.exists():
            print(f"Skipping {run_dir} (no config)")
            continue

        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

        label = format_label(cfg, label_fields)

        csv_path = run_dir / 'rmse_cost_history.csv'
        if not csv_path.exists():
            print(f"Skipping {run_dir} (no CSV data)")
            continue

        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)

        # Check number of entries in each row
        if data.shape[1] == 6:
            cost = data[:, 0]
            ate = data[:, 1]
            epe = data[:, 2]
        else:
            print("Unexpected number of columns in CSV file:", data.shape[1])
            sys.exit(1)

        ax_epe.plot(epe, marker='o', label=label)
        ax_ate.plot(ate, marker='o', label=label)


    for ax, fig, ylabel, title in [
        (ax_ate, fig_ate, "ATE (m)", "ATE over Iterations"),
        (ax_epe, fig_epe, "EPE (m)", "EPE over Iterations"),
    ]:
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.tight_layout()

    plt.show()


# ----------------------------
# Entry point
# ----------------------------

if __name__ == "__main__":
    visualize_ablation(
        ablation_root="/home/dl/Documents/phd/dev/dr_ba/output/ba_ablation/set_00",
        label_fields={
            "input.gauss_blur_sigma": "σ",
            "input.dist_field_preproc": "DF",
        },
    )
