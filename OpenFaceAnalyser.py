import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os

# ----------- Parameters ------------
video_number = 3
participant = 16
video_path = f"/Volumes/R@mtin/Year 2/Experiment/EyeTracker/Data/Participant {participant}/user reaction/video_part_{video_number}.mp4"
output_dir = f"/Volumes/R@mtin/Year 2/Experiment/EyeTracker/Data/Participant {participant}/AU_frames_part_{video_number}"
os.makedirs(output_dir, exist_ok=True)
au_threshold = 1.5
confidence_threshold = 0.9

# ----------- Load Data ------------
old_df = pd.read_csv(f'/Volumes/R@mtin/Year 2/Experiment/EyeTracker/Data/Participant {participant}/user reaction/video_part_{video_number}.csv')
df_2 = pd.read_csv(f'/Volumes/R@mtin/Year 2/Experiment/EyeTracker/Data/Participant {participant}/user reaction/video_part_{video_number}_frametime.csv')
original_df = pd.read_csv(f'/Volumes/R@mtin/Year 2/Experiment/EyeTracker/Data/Participant {participant}/ee_state_data.csv')

# ----------- Filter Time Range Based on Failure ------------
new_df = original_df[(original_df["Puzzle_number"] == video_number * 2) & (original_df["failure_number"] > 0)]

if new_df.empty:
    raise ValueError("No matching failure found for this video number.")

failure_num = int(new_df["failure_number"].iloc[0])

if failure_num == 1:
    start_time = new_df[new_df["state"] == 5]["timestamp"].iloc[0]
    end_time = new_df[new_df["state"] == 6]["timestamp"].iloc[0]
elif failure_num in [2, 3]:
    start_time = new_df[new_df["state"] == 5]["timestamp"].iloc[0]
    end_time = new_df[new_df["state"] == 8]["timestamp"].iloc[0]
else:
    raise ValueError("Unexpected failure number.")

df_3 = df_2[(df_2["timestamp"] > start_time * 1e9) & (df_2["timestamp"] < end_time * 1e9)]
start_frame = df_3["frame_number"].iloc[0]
end_frame = df_3["frame_number"].iloc[-1]

df = old_df[(old_df["frame"] > start_frame) & (old_df["frame"] < end_frame)]

# ----------- Filter & Prepare AU Columns ------------
au_cols = [col for col in df.columns if col.startswith("AU") and col.endswith("_r")]
eye_related = {"AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU45_r"}
filtered_au_cols = [c for c in au_cols if c not in eye_related]

df_confident = df[df['confidence'] > confidence_threshold].copy()

# ----------- Load Video and Annotate Frames ------------
cap = cv2.VideoCapture(video_path)
for _, row in df_confident.iterrows():
    active_aus = [au for au in filtered_au_cols if row[au] > au_threshold]
    if active_aus:
        frame_num = int(row['frame'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            text = ", ".join(active_aus)
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2, cv2.LINE_AA)
            output_path = os.path.join(output_dir, f"frame_{frame_num}.jpg")
            cv2.imwrite(output_path, frame)

cap.release()

output_dir  # Path where the annotated frames are saved




# # 2) Identify all AU intensity columns ("_r" suffix)
# au_cols = [col for col in df.columns if col.startswith("AU") and col.endswith("_r")]

# # 3) Define eye/brow/eyelid AUs to drop
# eye_related = {
#     "AU01_r",  # Inner Brow Raiser
#     "AU02_r",  # Outer Brow Raiser
#     "AU04_r",  # Brow Lowerer
#     "AU05_r",  # Upper Lid Raiser
#     "AU06_r",  # Cheek Raiser
#     "AU07_r",  # Lid Tightener
#     "AU45_r",  # Blink
# }

# # 4) Keep only non-eye AUs
# filtered_au_cols = [c for c in au_cols if c not in eye_related]
# print("Keeping these AU columns (non-eye):", filtered_au_cols)

# # 5) Filter frames where OpenFace confidence > 0.8
# df_confident = df[df['confidence'] > 0.85].copy()

# # 6) Choose x-axis: use 'frame' column if present, else use DataFrame index
# if 'frame' in df_confident.columns:
#     x_vals = df_confident['frame']
# else:
#     x_vals = df_confident.index

# # 7) Plot each non-eye AU only if its max > 1
# plt.figure(figsize=(12, 8))
# for au in filtered_au_cols:
#     series = df_confident[au].iloc[1:]       # ignore the first value
#     max_val = series.max()
#     if max_val > 1.5:
#         x = x_vals.iloc[1:]
#         plt.plot(x, series, label=au)

# plt.xlabel('Frame Number')
# plt.ylabel('AU Intensity')
# plt.title('Non-Eye Action Units (confidence > 0.8, max > 1)')
# plt.legend(fontsize='small', ncol=2, loc='upper right')
# plt.tight_layout()
# plt.show()
