import pandas as pd
from pathlib import Path
import csv

def failure_conveerter(failure_number):
    if failure_number == 0:
        return "No failure"
    elif failure_number == 1:
        return "Executional"
    elif failure_number == 2:
        return "Decisional"
    elif failure_number == 3: 
        return "Mechanical"
    
def failure_verbal_conveerter(failure_number):
    if failure_number == 0:
        return "No reaction"
    elif failure_number == 1:
        return "Acknowledgement"
    elif failure_number == 2:
        return "Apology Repair"


def expand_gaze_by_state(ee_path: Path, gaze_path: Path, output_path: Path):
    """
    Reads EE state and gaze CSVs, then writes one output row per gaze sample,
    annotated with the EE interval data in which it falls (skipping state==0 intervals).
    """
    # Load data
    ee = pd.read_csv(ee_path).sort_values("timestamp").reset_index(drop=True)
    gaze = pd.read_csv(gaze_path).sort_values("timestamp_ns").reset_index(drop=True)

    # Normalize gaze timestamps to seconds
    gaze['timestamp_s'] = gaze['timestamp_ns'] / 1e9

    # Build EE intervals: [timestamp, next_timestamp)
    ee['next_timestamp'] = ee['timestamp'].shift(-1)
    ee['next_state'] = ee['state'].shift(-1)
    ee_intervals = ee.dropna(subset=['next_timestamp']).reset_index(drop=True)

    # Build IntervalIndex for assignment
    intervals = pd.IntervalIndex.from_arrays(
        ee_intervals['timestamp'],
        ee_intervals['next_timestamp'],
        closed='right'
    )
    gaze['ee_idx'] = intervals.get_indexer(gaze['timestamp_s'])

    # Prepare output CSV
    with open(output_path, 'w', newline='') as fout:
        writer = csv.writer(fout)
        # Header: adjust names as needed
        writer.writerow([
            'timestamp', 'Participant_id',
            'Puzzle_number', 'object_number', 'failure_number',
            'Verbal_number', 'state', 'next_state', 'gaze_value'
        ])

        # Iterate over gaze samples
        for _, g_row in gaze.iterrows():
            idx = int(g_row['ee_idx'])
            # Skip if no interval or next_state == 0
            if idx < 0 or ee_intervals.at[idx, 'next_state'] == 0:
                continue

            # Extract EE fields
            ee_row = ee_intervals.loc[idx]
            participant = ee_row.get('Participant_id', '')
            puzzle = ee_row.get('Puzzle_number', '')
            obj_num = ee_row.get('object_number', '')
            failure = failure_conveerter(ee_row.get('failure_number', ''))
            verbal = failure_verbal_conveerter(ee_row.get('Verbal_number', ''))
            state = ee_row.get('state', '')
            Nstate = ee_row.get('next_state', '')

            # Extract gaze value (adjust column name as needed)
            gaze_val = g_row['Place of Interest']

            # Write combined row
            writer.writerow([
                g_row['timestamp_s'],
                participant,
                puzzle,
                obj_num,
                failure,
                verbal,
                state,
                Nstate,
                gaze_val
            ])

    print(f"Expanded gaze data written to {output_path}")
if __name__ == "__main__":
    for i in range(1,55):
        if i==3 or i == 9 or i==47:
            continue
        print(i)
        ee_file    = f'/Volumes/R@mtin/Year 2/Experiment/EyeTracker/Data/Participant {i}/ee_state_data.csv'
        gaze_file  = f'/Volumes/R@mtin/Year 2/Experiment/EyeTracker/Data/Participant {i}/region_checks_filled.csv'
        output_csv = f'/Volumes/R@mtin/Year 2/Experiment/EyeTracker/Data/Participant {i}/merged_gaze_state.csv'

        expand_gaze_by_state(ee_file, gaze_file, output_csv)
