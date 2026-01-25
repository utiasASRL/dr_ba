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
    param_sets = [
        {
            "map.voxel_res": 1.0,
            "input.dist_field_preproc": False,
            "input.gauss_blur_sigma": 3.0,
            "mapping.dist_field_preproc": False,
            "mapping.gauss_blur_sigma": 3.0,
            "input.seq_id": "boreas-2024-12-04-11-56",
            "input.map_id": "boreas-2024-12-04-11-45",
        },
        {
            "map.voxel_res": 1.0,
            "input.dist_field_preproc": True,
            "input.gauss_blur_sigma": 9.0,
            "mapping.dist_field_preproc": False,
            "mapping.gauss_blur_sigma": 3.0,
            "input.seq_id": "boreas-2024-12-04-11-56",
            "input.map_id": "boreas-2024-12-04-11-45",
        },
        {
            "map.voxel_res": 1.0,
            "input.dist_field_preproc": True,
            "input.gauss_blur_sigma": 9.0,
            "mapping.dist_field_preproc": True,
            "mapping.gauss_blur_sigma": 9.0,
            "input.seq_id": "boreas-2024-12-04-11-56",
            "input.map_id": "boreas-2024-12-04-11-45",
        },
        {
            "map.voxel_res": 1.0,
            "input.dist_field_preproc": True,
            "input.gauss_blur_sigma": 15.0,
            "mapping.dist_field_preproc": True,
            "mapping.gauss_blur_sigma": 15.0,
            "input.seq_id": "boreas-2024-12-04-11-56",
            "input.map_id": "boreas-2024-12-04-11-45",
        },
    ]

    # Create new subfolder within ablation based on number of existing folders
    base_output_path = Path("/home/dl/Documents/phd/dev/dr_ba/output/ablation_map_loc")
    existing_runs = [d for d in base_output_path.iterdir() if d.is_dir() and d.name.startswith("set_")]
    run_id = len(existing_runs)
    output_path = base_output_path / f"set_{run_id:02d}"
    output_path.mkdir(parents=True, exist_ok=True)

    base_config_path = Path("/home/dl/Documents/phd/dev/dr_ba/ba/config/ablation/")
    with open(base_config_path / "base_map_loc_config.yaml", "r") as f:
        base_cfg = yaml.safe_load(f)

    for i, overrides in enumerate(param_sets):
        cfg = copy.deepcopy(base_cfg)

        # Update output.output_path in config to unique folder
        cfg["output"]["output_path"] = str(output_path)

        for key_path, value in overrides.items():
            set_by_path(cfg, key_path, value)

        out_path = base_config_path / f"temp_map_loc_config.yaml"
        with open(out_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        print(f"Wrote {out_path}")

        # Run BA with this config
        subprocess.call(["build/app/dr_map_loc_test", str(out_path)])