import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import utils
import subprocess
import yaml



sequences = [
             #'boreas-2024-12-05-14-12', # Commercial
             #'boreas-2024-12-23-16-27', # Commercial
             #'boreas-2024-12-23-16-44', # Commercial
             #'boreas-2024-12-23-17-01', # Commercial
             #'boreas-2024-12-23-17-18', # Commercial


             'boreas-2024-12-03-10-24', # Glenshield
             #'boreas-2024-12-03-12-54', # Glenshield
             #'boreas-2025-01-08-10-59', # Glenshield
             #'boreas-2025-01-08-11-22', # Glenshield
             #'boreas-2025-01-08-12-28', # Glenshield

             #'boreas-2024-12-04-11-45', # Skyway
             #'boreas-2024-12-04-11-56', # Skyway
             #'boreas-2024-12-04-12-08', # Skyway
             #'boreas-2024-12-04-12-19', # Skyway
             #'boreas-2024-12-04-12-34', # Skyway
            ]

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
