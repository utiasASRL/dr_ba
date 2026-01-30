import yaml
import copy
from pathlib import Path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import subprocess

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
        "optimization.alpha": 2.0,
        "map.voxel_res": 1.0,
        "input.adaptive_blur": True,
        "optimization.use_cumul_thresh": True,
        "keyframing.max_kf_dist": 5.0,
        "input.input_type": "scans",
    }

    param_sets = [
        # Glen
        # {
        #     "input.seq_id": "boreas-2024-12-03-12-54",
        #     # "mapping.frame_ranges": [[300, 700], [3300, 3600]],
        # },
        # {
        #     "input.seq_id": "boreas-2025-01-08-10-59",
        #     "mapping.frame_ranges": [[300, 700], [3250, 3600]],
        # },
        # {
        #     "input.seq_id": "boreas-2025-01-08-11-22",
        #     "mapping.frame_ranges": [[300, 700], [3350, 3600]],
        # },
        # {
        #     "input.seq_id": "boreas-2025-01-08-12-28",
        #     "mapping.frame_ranges": [[300, 700], [3350, 3600]],
        # },
        # Industrial
        # {
        #     "input.seq_id": "boreas-2024-12-05-14-12",
        #     "mapping.frame_ranges": [[0, 600], [2000, -1]],
        # }
        # Skyway
        {
            "input.seq_id": "boreas-2024-12-04-11-45",
            # "mapping.frame_ranges": [[550, 730], [1500, 1660]],
            # "mapping.frame_ranges": [[490, 510], [1690, 1710]],
        },
        # {
        #     "input.seq_id": "boreas-2024-12-04-11-56",
        #     # "mapping.frame_ranges": [[400, 570], [1800, 1970]],
        # },
        # {
        #     "input.seq_id": "boreas-2024-12-04-12-08",
        #     # "mapping.frame_ranges": [[620, 800], [1330, 1520]],
        # },
        # {
        #     "input.seq_id": "boreas-2024-12-04-12-19",
        #     "mapping.frame_ranges": [[370, 540], [1700, 1850]],
        # },
        # Forest
        # {
        #     "input.seq_id": "boreas-2025-07-18-10-33",
        #     "mapping.frame_ranges": [[0, 600], [3900, -1]],
        # },
        # {
        #     "input.seq_id": "boreas-2025-07-18-11-00",
        #     "mapping.frame_ranges": [[0, 600], [3900, -1]],
        # },
        # {
        #     "input.seq_id": "boreas-2025-07-18-11-25",
        #     "mapping.frame_ranges": [[0, 600], [3900, -1]],
        # },
        # {
        #     "input.seq_id": "boreas-2025-07-18-11-53",
        #     "mapping.frame_ranges": [[0, 600], [3900, -1]],
        # },
    ]

    # Create new subfolder within ablation based on number of existing folders
    base_output_path = Path("/home/dl/Documents/phd/dev/dr_ba/output/ba_ablation")
    existing_runs = [d for d in base_output_path.iterdir() if d.is_dir() and d.name.startswith("set_")]
    run_id = len(existing_runs)
    output_path = base_output_path / f"set_{run_id:02d}"
    output_path.mkdir(parents=True, exist_ok=True)

    # Copy this file to output folder for record-keeping
    this_script_path = Path(__file__)
    subprocess.call(["cp", this_script_path, output_path / this_script_path.name])

    base_config_path = Path("/home/dl/Documents/phd/dev/dr_ba/ba/config/ablation/")
    with open(base_config_path / "base_ba_config.yaml", "r") as f:
        base_cfg = yaml.safe_load(f)

    for i, overrides in enumerate(param_sets):
        cfg = copy.deepcopy(base_cfg)

        # Update output.output_path in config to unique folder
        cfg["output"]["output_path"] = str(output_path)

        # Merge common + per-run overrides (per-run wins)
        merged_overrides = {**common_overrides, **overrides}

        for key_path, value in merged_overrides.items():
            set_by_path(cfg, key_path, value)

        out_path = base_config_path / "temp_ba_config.yaml"
        with open(out_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        print(f"Wrote {out_path}")

        # Run BA with this config
        subprocess.call(["build/app/dr_ba", str(out_path)])

    print("All results saved to:", output_path)