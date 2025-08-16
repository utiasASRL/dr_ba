# DRPoGO: Direct Radar Map Alighnment for Loop Closure in Pose Graph Optimization
This repository provides the codebase for radar-based coarse registration and evaluation against ground truth. The pipeline includes local map generation using DRO, loop closure proposal using RaPlace, and pose graph optimization.

## Dependencies

All dependencies should be in `requirements.txt`. Please install them in you virtual environment with
```
pip install -r requirements.txt
```


## Run DRO
First, download data from the Boreas dataset [here](https://www.boreas.utias.utoronto.ca/#/download). DRO generates local maps that accounts for motion distortion of the radar scans.

Then copy the example config file `DRO/config_example.yaml` to `DRO/config.yaml` and modify the parameters as needed, especially the `data_path` as follows.
```yaml
  data:
    data_path: /absolute/path/to/Boreas/<sequence>
```

In the root of the repository, run the following command to generate local maps:
```bash
python dro/radar_gp_state_estimation.py
```

It will output the local maps in the `output/<SEQ-NAME>/local_maps` folder, each names with the radar scan's first timestamp. 

## Run RaPlace
Simply run RaPlace as follows in the root of the repository (all the paths should be autonomatically using what was specified in the DRO config file):
```bash
python raplace/raplace.py
```

It will generate a CSV of proposed scan pairs in the `output/<SEQ-NAME>/raplace_loops.csv` folder.
Each row contains the following columns:
- `time_i`: Timestamp of the first scan in the pair \[s\].
- `time_j`: Timestamp of the second scan in the pair \[s\].
- `scan_i_name`: Name of the first scan in the pair.
- `scan_j_name`: Name of the second scan in the pair.
- `score`: The score of the proposed loop closure as defined in RaPlace (not used)
- `min_dist`: The minimum dist between the scores as defined in RaPlace (not used)

## Run the feature-based coarse registration

In the root of the repository, run the following command:
```bash
python coarse_registration/coarse_registration.py
```

It will generate a CSV of coarse registration results in the `output/<SEQ-NAME>/coarse_registration.csv` folder.
Each row contains the following columns:
- `scan_i_name`: Name of the first scan in the pair.
- `scan_j_name`: Name of the second scan in the pair.
- `x`, `y`, `theta`: The estimated transformation from scan i to scan j.

## For paper and evaluation

To plot the coarse registration errors, run the following command:
```bash
python script_for_paper/plot_coarse_registration_error.py
```

