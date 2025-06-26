import pandas as pd
from pathlib import Path
import os

# Constants
MAX_GAP = 10
MISSING_LABELS = {'No detected places'}

def fill_short_gaps(series: pd.Series, max_gap: int = MAX_GAP) -> pd.Series:
    """
    Fill short runs of missing labels (<= max_gap) in a gaze-label Series
    if the label before and after the gap are the same.
    """
    filled = series.copy()
    n = len(filled)

    for idx in range(1, n):
        current = filled.iat[idx]
        prev = filled.iat[idx - 1]

        # We're interested when we transition from missing -> valid label
        if current not in MISSING_LABELS and prev in MISSING_LABELS:
            # Measure gap length
            gap_end = idx - 1
            gap_start = gap_end
            while gap_start > 0 and filled.iat[gap_start - 1] in MISSING_LABELS:
                gap_start -= 1
            gap_length = gap_end - gap_start + 1

            # Only fill if gap is short enough and surrounding labels match
            if gap_length <= max_gap:
                before_label = filled.iat[gap_start - 1] if gap_start > 0 else None
                after_label = current
                if before_label == after_label:
                    filled.iloc[gap_start: idx] = after_label

    return filled

def process_session_folder(session_path: Path):
    """Load GazeString.csv, fill gaps, and save to GazeString2.csv."""
    gaze_csv = os.path.join(session_path, "region_checks.csv")
    output_csv = os.path.join(session_path, "region_checks_filled.csv")

    df = pd.read_csv(gaze_csv)
    df['Place of Interest'] = fill_short_gaps(df['Place of Interest'])

    df.to_csv(output_csv, index=False, columns=['timestamp_ns', 'Video_time', 'Place of Interest'])
    # print(f"Processed {session_path.name}: wrote {output_csv.name}")
    print(f"Processed Done")

if __name__ == "__main__":
    for i in range(1,55):
        if i == 3 or i == 9 or i ==47:
            continue
        base_dir = f'/Volumes/R@mtin/Year 2/Experiment/EyeTracker/Data/Participant {i}'
        process_session_folder(base_dir)
