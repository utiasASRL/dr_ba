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
      }
    """
    parts = []
    for path, name in label_fields.items():
        val = get_by_path(cfg, path)
        parts.append(f"{name}={val}")
    return ", ".join(parts)


# ----------------------------
# Main visualization
# ----------------------------

def visualize_ablation(
    ablation_root,
    label_fields
):
    ablation_root = Path(ablation_root)
    run_dirs = sorted(
        (d for d in ablation_root.iterdir() if d.is_dir()),
        key=lambda d: d.name
    )
    print(run_dirs)

    fig_epe, ax_epe = plt.subplots()
    fig_ate, ax_ate = plt.subplots()

    avg_ate = 0.0
    avg_epe = 0.0
    count = 0

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

        avg_ate += ate[-1]
        avg_epe += epe[-1]
        count += 1
    if count > 0:
        avg_ate /= count
        avg_epe /= count
        print(f"Average final ATE over {count} runs: {avg_ate:.4f} m")
        print(f"Average final EPE over {count} runs: {avg_epe:.4f} m")


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
        # ablation_root="/home/dl/Documents/phd/dev/dr_ba/output/aaa_paper_results/ba/skyway",
        ablation_root="/home/dl/Documents/phd/dev/dr_ba/output/aa_paper_ablation/drba_skyway",
        label_fields={
            "input.gauss_blur_sigma": "σ",
        },
    )
