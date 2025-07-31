import os, sys, pandas as pd
from RaPlace import RaPlace  # or however you import your RaPlace functions

def export_loops(scan_folder, out_csv, score_thresh=0.0):
    rp = Raplace()
    scans = sorted([f for f in os.listdir(scan_folder) if f.endswith('.png') or f.endswith('.bin')])
    records = []
    N = len(scans)
    for i in range(N):
        scan_i = rp.load_scan(os.path.join(scan_folder, scans[i]))
        for j in range(i+1, N):
            scan_j = rp.load_scan(os.path.join(scan_folder, scans[j]))
            score = rp.compute_similarity(scan_i, scan_j)
            if score >= score_thresh:
                records.append({'scan_i': scans[i], 'scan_j': scans[j], 'score': float(score)})

    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"Wrote {len(records)} loop candidates to {out_csv}")

if __name__ == '__main__':
    export_loops('/home/liweican-2025-research/root/RaPlace', 'loops.csv', score_thresh=0.85)
