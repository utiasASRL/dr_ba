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
        "map.voxel_res": 1.0,
        "optimization.range_factor": 0.02,

        "input.adaptive_blur": False,
        "input.gauss_blur_sigma": 3.0,
        "input.min_int_val_tol": 0.1,
        "input.min_percent_nonzero": 3.0,

        "mapping.adaptive_blur": False,
        "mapping.gauss_blur_sigma": 3.0,
        "mapping.min_int_val_tol": 0.1,
        "mapping.min_percent_nonzero": 3.0,

        "mapping.pose_source": "gt",
        "mapping.estimate_location": "/home/dl/Documents/phd/dev/dr_ba/output/aaa_paper_results/boreas-2024-12-03-12-54"
    }

    param_sets = [
        {
            "input.seq_id": "boreas-2025-01-08-10-59",
            "mapping.map_seq": "boreas-2024-12-03-12-54",
            # "mapping.frame_ranges": [[300, 700], [3300, 3600]],
            # "localization.start_frame": 300,
            # "localization.end_frame": 600,
        },
        {
            "input.seq_id": "boreas-2025-01-08-11-22",
            "mapping.map_seq": "boreas-2024-12-03-12-54",
            # "mapping.frame_ranges": [[300, 700], [3300, 3600]],
            # "localization.start_frame": 300,
            # "localization.end_frame": 600,
        },
        {
            "input.seq_id": "boreas-2025-01-08-12-28",
            "mapping.map_seq": "boreas-2024-12-03-12-54",
            # "mapping.frame_ranges": [[300, 700], [3300, 3600]],
            # "localization.start_frame": 300,
            # "localization.end_frame": 600,
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

        # Merge common + per-run overrides (per-run wins)
        merged_overrides = {**common_overrides, **overrides}

        for key_path, value in merged_overrides.items():
            set_by_path(cfg, key_path, value)

        out_path = base_config_path / f"temp_map_loc_config.yaml"
        with open(out_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        print(f"Wrote {out_path}")

        # Run BA with this config
        subprocess.call(["build/app/dr_map_loc_test", str(out_path)])



# For ablation localization
    # param_sets = [
    #     {
    #         "input.seq_id": "boreas-2025-01-08-10-59",
    #         "mapping.map_seq": "boreas-2024-12-03-12-54",
    #         "mapping.frame_ranges": [[300, 700], [3300, 3600]],
    #         "localization.start_frame": 300,
    #         "localization.end_frame": 600,
    #     },
    #     {
    #         "input.seq_id": "boreas-2025-01-08-11-22",
    #         "mapping.map_seq": "boreas-2024-12-03-12-54",
    #         "mapping.frame_ranges": [[300, 700], [3300, 3600]],
    #         "localization.start_frame": 300,
    #         "localization.end_frame": 600,
    #     },
    #     {
    #         "input.seq_id": "boreas-2025-01-08-12-28",
    #         "mapping.map_seq": "boreas-2024-12-03-12-54",
    #         "mapping.frame_ranges": [[300, 700], [3300, 3600]],
    #         "localization.start_frame": 300,
    #         "localization.end_frame": 600,
    #     },
    # ]


# For testing troublesome regions
# {
#     "map.voxel_res": 1.0,
#     "optimization.range_factor": 0.001,
#     "input.gauss_blur_sigma": 3.0,
#     "mapping.gauss_blur_sigma": 3.0,
#     "localization.odometry_prior.use_odometry_prior": True,
#     "localization.odometry_prior.translation_std": 0.1,
#     "localization.odometry_prior.rotation_std": 0.1,
#     "input.seq_id": "boreas-2025-01-08-10-59",
#     "mapping.map_seq": "boreas-2024-12-03-12-54",
#     "mapping.frame_ranges": [[500, 750], [3320, 3550]],
#     "localization.start_frame": 500,
#     "localization.end_frame": 600,
# },
# {
#     "map.voxel_res": 1.0,
#     "optimization.range_factor": 0.001,
#     "input.gauss_blur_sigma": 3.0,
#     "mapping.gauss_blur_sigma": 3.0,
#     "localization.odometry_prior.use_odometry_prior": True,
#     "localization.odometry_prior.translation_std": 0.1,
#     "localization.odometry_prior.rotation_std": 0.1,
#     "input.seq_id": "boreas-2024-12-04-11-56",
#     "mapping.map_seq": "boreas-2024-12-04-11-45",
#     "mapping.frame_ranges": [[400, 550], [1650, 1800]],
#     "localization.start_frame": 500,
#     "localization.end_frame": 600,
# },
# {
#     "map.voxel_res": 1.0,
#     "optimization.range_factor": 0.001,
#     "input.gauss_blur_sigma": 3.0,
#     "mapping.gauss_blur_sigma": 3.0,
#     "localization.odometry_prior.use_odometry_prior": True,
#     "localization.odometry_prior.translation_std": 0.1,
#     "localization.odometry_prior.rotation_std": 0.1,
#     "input.seq_id": "boreas-2025-07-18-15-12",
#     "mapping.map_seq": "boreas-2025-07-18-14-55",
#     "mapping.frame_ranges": [[2050, 2200]],
#     "localization.start_frame": 2100,
#     "localization.end_frame": 2200,
# },
# {
#     "map.voxel_res": 1.0,
#     "optimization.range_factor": 0.001,
#     "input.gauss_blur_sigma": 3.0,
#     "mapping.gauss_blur_sigma": 3.0,
#     "localization.odometry_prior.use_odometry_prior": True,
#     "localization.odometry_prior.translation_std": 0.1,
#     "localization.odometry_prior.rotation_std": 0.1,
#     "input.seq_id": "boreas-2025-07-18-11-00",
#     "mapping.map_seq": "boreas-2025-07-18-10-33",
#     "mapping.frame_ranges": [[2250, 2425]],
#     "localization.start_frame": 2300,
#     "localization.end_frame": 2400,
# },