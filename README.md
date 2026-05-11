# Dr-BA: Separable Optimization for Direct Radar Bundle Adjustment & Localization
This repository provides the codebase for radar-based bundle adjustment and localization. The pipeline includes local map generation using DRO, loop closure proposal using RaPlace, pose graph optimization using Dr-PoGO, bundle adjustment using Dr-BA, and direct localization using DRL. Please refer to our paper for full details: https://arxiv.org/abs/2605.07041.

## Dependencies

All dependencies should be in `requirements.txt`. Please install them in you virtual environment with
```
pip install -r requirements.txt
```

For Dr-PoGO, a required part of this repo, you will need Eigen3 and Ceres Solver:
```bash
sudo apt-get install python3-tk
sudo apt-get install python3-dev
sudo apt-get install build-essential
sudo apt-get install cmake
sudo apt-get install libeigen3-dev
sudo apt-get install libceres-dev
sudo apt-get install libyaml-cpp-dev
```

Also clone the submodules
```bash
git submodule update --init --recursive
```

## Compile

1) Compile Dr-PoGO:
```bash
cd pogo
mkdir -p build
cd build
cmake ..
make -j8
cd ../..
```
2) Compile Dr-BA
```bash
bash ba/scripts/build_package.sh
```

## Download data
First, download data from the Boreas dataset [here](https://www.boreas.utias.utoronto.ca/#/download). The pipeline has thus far only been tested on Boreas RT data and the config files are populated with good values for this data. To run everything including localization you will need to download at least two sequences from the same route. The full set of sequences used in the paper can be found in `script_for_paper/run_dr_pogo.py`.

## Run the full pipeline

You need to run the following steps in order:

1. Run pose graph optimization for local maps, cumulative images, and initial guess generation (Dr-PoGO)
2. Run bundle adjustment for refined map poses (Dr-BA)
3. (Optional) Run mapping step to create a map based on the BA results. Only needed if the mapping parameters are different from those used in BA.
4. Run direct localization based on a created map (DRL)

### Run Dr-PoGO
Modify the `data_path` in the `DRO/config.yaml` file to point to one of the downloaded sequences:
```yaml
  data:
    data_path: /absolute/path/to/Boreas/<seq-name>
    multi_sequence: false
```

It does not matter what sequence name you put here as it will be swapped out automatically later. The main runfile that will be used is `script_for_paper/run_dr_pogo.py`. Edit the set of sequences that you want Dr-PoGO to be run on.
Then, simply run
```bash
python script_for_paper/run_dr_pogo.py
```

### Run Dr-BA
Modify the `ba/config/ba_config.yaml` file with at minimum an udpate to:

- `data/data_path` to point to where Boreas data is downloaded
- `data/meas_path` to point to the output of Dr-PoGO (automatically located under `this/repo/output`)
- `output/output_path` to point to where you want to save Dr-BA results to (if saving is enabled)
- `output/visualize` and `output/save_result` for self-explanatory behaviour. You must save results if you want to run localization.
- `ba/seq_id` must be set to the sequence name

If you are running out of RAM to run BA, you can also lower the `ba/optimization/max_loaded_scans` parameter. The lower this is the slower the optimization will be, but the less RAM it will require.

Once the config file is good, run
```bash
bash ba/scripts/run_ba.sh
```

### Run Dr-BA Mapping
This step is optional in the case that you wish to generate a map with parameters different from those used to run BA. To generate a new map on the base of the BA-optimized poses, modify the `ba/config/map_config.yaml` file with at minimum an udpate to:
- `mapping/seq_id` with the same sequence ID as what was used to generate the BA solution
- `mapping/estimate_location` with `output/output_path` from the BA solve

Once the config file is good, run
```bash
bash ba/scripts/run_map.sh
```

### Run DRL (localization)
Modify the `ba/config/loc_config.yaml` file with at minimum an udpate to:

- `localization/seq_id` with the localizing sequence ID
- `localization/map_seq_id` with the sequence ID of the mapping sequence (the one that BA was run on)
- `localization/map_location` with the location of the BA or mapping step output

Once the config file is good, run
```bash
bash ba/scripts/run_loc.sh
```

The script will automatically print out the localization RMSE for the sequence.

### Visualization
A number of self-explanatory visualization scripts are available under `ba_py`.
These get automatically triggered when the `output/visualize` options are enabled under any part of the Dr-BA pipeline.
They can also be manually called with a specific output directory.