import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import utils
import subprocess
import yaml


# RSS data
sequences = [
             'boreas-2024-12-05-14-12', # Commercial
             'boreas-2024-12-23-16-27', # Commercial
             'boreas-2024-12-23-16-44', # Commercial
             'boreas-2024-12-23-17-01', # Commercial
             'boreas-2024-12-23-17-18', # Commercial


             'boreas-2024-12-03-10-24', # Glenshield
             'boreas-2024-12-03-12-54', # Glenshield
             'boreas-2025-01-08-10-59', # Glenshield
             'boreas-2025-01-08-11-22', # Glenshield
             'boreas-2025-01-08-12-28', # Glenshield

             'boreas-2024-12-04-11-45', # Skyway
             'boreas-2024-12-04-11-56', # Skyway
             'boreas-2024-12-04-12-08', # Skyway
             'boreas-2024-12-04-12-19', # Skyway
             'boreas-2024-12-04-12-34', # Skyway
            ]

### Original Boreas data
#sequences = [
#            'boreas-2020-11-26-13-58',
#            'boreas-2020-12-18-13-44',
#            'boreas-2021-01-26-11-22',
#            'boreas-2021-02-02-14-07',
#            'boreas-2021-03-02-13-38',
#            'boreas-2021-03-30-14-23',
#            'boreas-2021-04-20-14-11',
#            'boreas-2021-05-13-16-11',
#            'boreas-2021-07-20-17-33',
#            'boreas-2021-09-02-11-42',
#            'boreas-2021-10-15-12-35',
#            'boreas-2021-11-14-09-47',
#            'boreas-2021-11-23-14-27',
#            ]

data_path = utils.getDataDir()


for seq in sequences:

    # Change the data.data_path in dro/config.yaml to the current sequence
    with open(os.path.join("dro", "config.yaml"), 'r') as f:
        opts = yaml.safe_load(f)
    opts['data']['data_path'] = os.path.join(data_path, seq)
    with open(os.path.join("dro", "config.yaml"), 'w') as f:
        yaml.dump(opts, f)

    try:
        subprocess.call(["python3","dro/radar_gp_state_estimation.py"])
        subprocess.call(["python3","raplace/raplace.py"])
        subprocess.call(["python3","coarse_registration/coarse_registrations.py"])
        subprocess.call(["python3","dro/fine_registration.py"])
        subprocess.call(["pogo/build/pogo"])
    except:
        print(f"An error occurred while processing sequence {seq}.")
        continue
