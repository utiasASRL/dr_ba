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
In `RaPlace/PYTHON/RaPlace.py`, set:
```python
radar_data_dir = "../../local_maps"  # path to PNGs
gtpose = ""                          # unused for this pipeline
```

In the `RaPlace/PYTHON/` folder, run:

```bash
python RaPlace.py
```

It will generate a CSV of proposed scan pairs in the `RaPlace/PYTHON/` folder. 

## Run Coarse_registration

In `Coarse_registration/gps_to_radar.py`, add the argument for the boreas's calibration directory and the path to the gps_post_process.csv. This file converts the Boreas 200 Hz Applanix poses to radar-frame poses, later used to interpolate ground truth poses in the radar frame. 
In the `Coarse_registration/` folder, run:

```bash
python gps_to_radar.py \
  --calib_dir  <sequence>/calib \
  --gps_csv    <sequence>/applanix/gps_post_process.csv
```
Produces `radar_frame_poses.csv` with timestamp-aligned SE(2) ground-truth.

Then, run:
```bash
python compare_est_gt.py \
  --loops  ../RaPlace/PYTHON/loop_pairs.csv \
  --imgdir ../local_maps \
  --gt     radar_frame_poses.csv
```
It will visualize the PNGs with the RaPlace proposed loop closure radar scans, the GT distance of the two scans, the tranlation error and rotational error between the SIFT proposed R and t matrices with the GT R and t matrices. It will also output a CSV containing the errors and matrices. 
