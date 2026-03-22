import yaml
import copy
from pathlib import Path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import subprocess
import os.path as osp

def set_by_path(cfg, key_path, value):
    keys = key_path.split(".")
    d = cfg
    for k in keys[:-1]:
        if k not in d:
            raise KeyError(f"Invalid config key: {key_path}")
        d = d[k]
    d[keys[-1]] = value


if __name__ == "__main__":
    common_overrides = {
        "input.input_type": "local_maps",
        "optimization.range_factor": 0.0005,
        "input.adaptive_blur": False,
        "input.gauss_blur_sigma": 3.0,
        "input.min_int_val_tol": 0.1,
        "input.min_percent_nonzero": 3.0,

        "localization.map_location": "/home/dl/Documents/phd/dev/dr_ba/output/aa_paper_ablation/no_map_skyway/run_1",

        # "localization.map_location": "/home/dl/Documents/phd/dev/dr_ba/output/aaa_paper_results/ba/forest/boreas-2025-07-18-10-33",
        "mapping.map_seq": "boreas-2024-12-04-11-45",
    }

    # Glen (boreas-2024-12-03-12-54)
    # param_sets = [
    #     {
    #         "input.seq_id": "boreas-2025-01-08-10-59",
    #     },
    #     {
    #         "input.seq_id": "boreas-2025-01-08-11-22",
    #     },
    #     {
    #         "input.seq_id": "boreas-2025-01-08-12-28",
    #     },
    # ]

    # Industrial (boreas-2024-12-05-14-12)
    # param_sets = [
    #     {
    #         "input.seq_id": "boreas-2024-12-23-16-27",
    #     },
    #     {
    #         "input.seq_id": "boreas-2024-12-23-16-44",
    #     },
    #     {
    #         "input.seq_id": "boreas-2024-12-23-17-01",
    #     },
    # ]

    # Skyway (boreas-2024-12-04-11-45)
    param_sets = [
        {
            "input.seq_id": "boreas-2024-12-04-11-56",
        },
        {
            "input.seq_id": "boreas-2024-12-04-12-08",
        },
        {
            "input.seq_id": "boreas-2024-12-04-12-19",
        },
    ]

    # Forest (boreas-2025-07-18-10-33)
    # param_sets = [
    #     {
    #         "input.seq_id": "boreas-2025-07-18-11-00",
    #     },
    #     {
    #         "input.seq_id": "boreas-2025-07-18-11-25",
    #     },
    #     {
    #         "input.seq_id": "boreas-2025-07-18-11-53",
    #     },
    # ]

    # Farm (boreas-2025-07-18-14-55)
    # param_sets = [
    #     {
    #         "input.seq_id": "boreas-2025-07-18-15-12",
    #         "localization.start_frame": 620,
    #     },
    #     {
    #         "input.seq_id": "boreas-2025-07-18-15-30",
    #     },
    #     {
    #         "input.seq_id": "boreas-2025-07-18-15-48",
    #     },
    # ]

    # Create new subfolder within ablation based on number of existing folders
    base_output_path = Path("/home/dl/Documents/phd/dev/dr_ba/output/ablation_loc")
    if (not osp.exists(base_output_path)):
        base_output_path.mkdir(parents=True, exist_ok=True)
    existing_runs = [d for d in base_output_path.iterdir() if d.is_dir() and d.name.startswith("set_")]
    run_id = len(existing_runs)
    output_path = base_output_path / f"set_{run_id:02d}"
    output_path.mkdir(parents=True, exist_ok=True)

    base_config_path = Path("/home/dl/Documents/phd/dev/dr_ba/ba/config/ablation/")
    with open(base_config_path / "base_loc_config.yaml", "r") as f:
        base_cfg = yaml.safe_load(f)

    for i, overrides in enumerate(param_sets):
        cfg = copy.deepcopy(base_cfg)

        # Update output.output_path in config to unique folder
        cfg["output"]["output_path"] = str(output_path)

        # Merge common + per-run overrides (per-run wins)
        merged_overrides = {**common_overrides, **overrides}

        for key_path, value in merged_overrides.items():
            set_by_path(cfg, key_path, value)

        out_path = base_config_path / f"temp_loc_config.yaml"
        with open(out_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        print(f"Wrote {out_path}")

        # Run BA with this config
        subprocess.call(["build/app/dr_loc", str(out_path)])

    print("All results saved to:", output_path)



