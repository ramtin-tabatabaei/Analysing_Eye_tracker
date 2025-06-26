import pandas as pd
import numpy as np
from pathlib import Path

# --- Helper functions ---

def remove_consecutive_duplicates(seq):
    """Remove consecutive duplicates from a list."""
    if not seq:
        return []
    seq = [x for x in seq if x != 'No detected places']
    result = [seq[0]]
    for item in seq[1:]:
        if item != result[-1]:
            result.append(item)
    return result

def build_transition_matrix(states, categories):
    """Build a Markov transition probability matrix for given states."""
    counts = pd.DataFrame(0, index=categories, columns=categories)
    for a, b in zip(states, states[1:]):
        counts.at[a, b] += 1
    # convert to probabilities
    prob = counts.div(counts.sum(axis=1).replace(0,1), axis=0).fillna(0)
    return prob

def stationary_entropy(P):
    """
    Compute the Shannon entropy of the stationary distribution for transition
    matrix P, ignoring any zero-probability states.
    """
    # 1) Find stationary distribution π by the leading eigenvector of Pᵀ
    vals, vecs = np.linalg.eig(P.T)
    idx = np.argmax(np.isclose(vals, 1.0))
    pi = np.real(vecs[:, idx])
    pi = pi / pi.sum()

    # 2) Keep only positive entries (drop zeros)
    pi_pos = pi[pi > 0]

    # 3) Compute entropy H_s = -∑ π_i log2(π_i)
    Hs = -np.sum(pi_pos * np.log2(pi_pos))

    return Hs

def transition_entropy(P):
    """Compute transition entropy H_t for matrix P."""
    # stationary distribution
    vals, vecs = np.linalg.eig(P.T)
    idx = np.argmax(np.isclose(vals, 1.0))
    pi = np.real(vecs[:, idx])
    pi = pi / pi.sum()
    P_nonzero = P.copy()
    P_nonzero[P_nonzero==0] = 1  # avoid log2(0)
    Ht = -np.sum(pi[:,None] * P * np.log2(P_nonzero))
    return Ht

# def mean_repeat_length(seq, target):
#     """Mean length of consecutive runs of target in seq."""
#     lengths = []
#     count = 0
#     for item in seq:
#         if item == target:
#             count += 1
#         else:
#             if count:
#                 lengths.append(count)
#                 count = 0
#     if count:
#         lengths.append(count)
#     return np.mean(lengths) if lengths else 0

def probability_of_state(seq, target):
    """Proportion of occurrences of target in seq."""
    return seq.count(target) / len(seq) if seq else 0


def gaze_shift_counter(seq, numebr_of_puzzles, Target):
    if Target == "all":
        return (len(seq)-numebr_of_puzzles)/numebr_of_puzzles
    else:
        return (seq.count(Target))/numebr_of_puzzles
        

def Calculate_time(sub):
    """
    Given sub-DataFrame filtered for one puzzle & (no-)failure,
    compute the average duration per object (object_number).
    
    Returns:
        float: mean duration across all object_number groups.
    """
    # Ensure sub is sorted by time
    # sub = sub.sort_values("timestamp")

    durations = []
    # Group by object_number
    for obj_num, grp in sub.groupby("object_number"):
        times = grp["timestamp"].values
        if len(times) >= 2:
            # duration for this object: last minus first
            durations.append(times[-1] - times[0])
    if not durations:
        return 0.0
    # Return the mean duration (or sum, or whatever you need)
    return float(np.mean(durations))
                

# --- Main processing ---

def process_sessions():
    categories = ['end_effector',
                  'tangram',
                'robot_head',
                'robot_pieces',
                'robot_body',
                'user_pieces']
    records = []

    output_csv = f'/Volumes/R@mtin/Year 2/Experiment/EyeTracker/Data/participants_gaze_summary.csv'


    for i in range(1,55):
        if i==3 or i == 9 or i==47 or i==38:
            continue
        gaze_file = f'/Volumes/R@mtin/Year 2/Experiment/EyeTracker/Data/Participant {i}/merged_gaze_state.csv'

        df = pd.read_csv(gaze_file)

        uniques = list(set(df["Puzzle_number"]))

        # for each puzzle 1-6
        for puzzle in range(1,7):
            if puzzle not in uniques:
                continue
            
            for object_number in range(1,5):
                if object_number not in df["object_number"]:
                    continue

                
                sub = df[
                            (df["Puzzle_number"] == puzzle) &
                            (df["object_number"] == object_number)
                        ]
                
                sub = sub[
                            (sub["state"] < 10)
                        ]
                            

                numebr_of_puzzles = 1

                seq = sub["gaze_value"].tolist()
                seq = remove_consecutive_duplicates(seq)
                if not seq:
                    continue

                P = build_transition_matrix(seq, categories).values

                shift_count = gaze_shift_counter(seq, numebr_of_puzzles, "all")
                Ht = transition_entropy(P)
                Hs = stationary_entropy(P)

                rec = {
                    "Participant_id" : df["Participant_id"][0],
                    "Puzzle_number": puzzle,
                    "Object_number": object_number,
                    "Failure_type": sub["failure_number"].iloc[0],
                    "Duration": Calculate_time(sub),
                    "num_gaze_shifts": shift_count,
                    "robot_head_gaze_shifts": gaze_shift_counter(seq, numebr_of_puzzles, "robot_head"),
                    "robot_pieces_gaze_shifts": gaze_shift_counter(seq, numebr_of_puzzles, "robot_pieces"),
                    "EndEffector_gaze_shifts": gaze_shift_counter(seq, numebr_of_puzzles, "end_effector"),
                    "prob_end_eff": probability_of_state(seq, "end_effector"),
                    "prob_tangram": probability_of_state(seq, "tangram"),
                    "prob_robot_pieces": probability_of_state(seq, "robot_pieces"),
                    "prob_robot_head": probability_of_state(seq, "robot_head"),
                    "transition_entropy": Ht,
                    "stationary_entropy": Hs
                }

                if shift_count < 6 or (Ht == 0 and Hs == 0):
                    continue

                records.append(rec)

    out_df = pd.DataFrame(records)
    out_df.to_csv(output_csv, index=False)
    print(f"Saved summary to {output_csv}")

if __name__ == "__main__":

        process_sessions()
