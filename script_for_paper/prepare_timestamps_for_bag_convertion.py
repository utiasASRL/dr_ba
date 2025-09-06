import os
import numpy as np

kDataPath = "/media/ced/Extreme Pro/data/boreas/rss_doppler_corrected"

def main():
    # Get the list of sequences
    sequences = [d for d in os.listdir(kDataPath) if os.path.isdir(os.path.join(kDataPath, d))]
    sequences.sort()

    for seq in sequences:
        # Get the list of radar frames
        radar_path = os.path.join(kDataPath, seq, 'radar')
        if not os.path.exists(radar_path):
            print(f"Radar path does not exist for sequence {seq}, skipping...")
            continue
        radar_frames = [f for f in os.listdir(radar_path) if f.endswith('.png')]
        radar_frames.sort()

        # Prepare timestamps for each radar frame
        timestamps = []
        for frame in radar_frames:
            # Get the timestamp from the filename as int64
            timestamp = int(frame.split('.')[0])
            timestamps.append(timestamp)

        timestamps = np.array(timestamps, dtype=np.int64)
        # Save to a file
        np.savetxt(os.path.join(kDataPath, seq, 'radar.timestamps'), timestamps, fmt='%d')
        print(f"Saved radar timestamps for sequence {seq} to radar.timestamps")

if __name__ == "__main__":
    main()
