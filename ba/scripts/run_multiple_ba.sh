#!/bin/bash

SCRIPT_PATH=$(dirname "$(realpath "$0")")
CONFIG_FILE="$SCRIPT_PATH/../config/dr_ba_config.yaml"

sequences=(
    'boreas-2024-12-03-12-54' # Glenshield
    # 'boreas-2025-01-08-10-59' # Glenshield
    # 'boreas-2025-01-08-11-22' # Glenshield
    # 'boreas-2025-01-08-12-28' # Glenshield

    'boreas-2024-12-04-11-45' # Skyway
    # 'boreas-2024-12-04-11-56' # Skyway
    # 'boreas-2024-12-04-12-08' # Skyway
    # 'boreas-2024-12-04-12-19' # Skyway

    'boreas-2024-12-05-14-12' # Industrial
    # 'boreas-2024-12-23-16-27' # Industrial
    # 'boreas-2024-12-23-16-44' # Industrial
    # 'boreas-2024-12-23-17-01' # Industrial

    'boreas-2025-07-18-10-33' # Forest
    # 'boreas-2025-07-18-11-00' # Forest
    # 'boreas-2025-07-18-11-25' # Forest
    # 'boreas-2025-07-18-11-53' # Forest

    'boreas-2025-07-18-14-55' # Farm
    # 'boreas-2025-07-18-15-12' # Farm
    # 'boreas-2025-07-18-15-30' # Farm
    # 'boreas-2025-07-18-15-48' # Farm

    # 'boreas-2025-08-06-06-33' # Urban
    # 'boreas-2025-08-06-07-05' # Urban
    # 'boreas-2025-08-06-07-41' # Urban
    # 'boreas-2025-08-06-08-35' # Urban

    'boreas-2025-01-08-10-59' # Glenshield
    'boreas-2024-12-04-11-56' # Skyway
    'boreas-2024-12-23-16-27' # Industrial
    'boreas-2025-07-18-11-00' # Forest
    'boreas-2025-07-18-15-12' # Farm
)

for seq in "${sequences[@]}"; do
    echo "Processing sequence: $seq"
    
    # Update seq_ids in config file - replace the entire seq_ids list with just the current sequence
    sed -i "/seq_ids:/,/^  [^ ]/ { /seq_ids:/! { /^  [^ ]/! d } }" "$CONFIG_FILE"
    sed -i "/seq_ids:/a\    - '$seq'" "$CONFIG_FILE"
    
    # Run the BA script
    bash "$SCRIPT_PATH/run_ba.sh"
    
    if [ $? -ne 0 ]; then
        echo "Error processing sequence $seq"
        continue
    fi
done

echo "All sequences processed"