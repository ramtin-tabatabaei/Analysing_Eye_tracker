import shutil
import os

# 1) Configure your source and destination directories
src_dir  = '/Users/stabatabaeim/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/University/Year 2/Experiment/EyeTracker/Data'
dst_dir  = '/Volumes/R@mtin/Year 2/Experiment/EyeTracker/Data'

for i in range(10, 55):
    # prepare participant subfolders
    src_dir_p = os.path.join(src_dir,  f'Participant {i}')
    dst_dir_p = os.path.join(dst_dir,  f'Participant {i}')
    os.makedirs(dst_dir_p, exist_ok=True)

    # find the “2025…” scene folder
    folders_2025 = [
        name for name in os.listdir(src_dir_p)
        if os.path.isdir(os.path.join(src_dir_p, name)) and name.startswith('2025')
    ]
    folder1_src = os.path.join(src_dir_p, folders_2025[0])

    # find the “Timeseries Data + Scene Video” parent, then its own “2025…” subfolder
    folders_time = [
        name for name in os.listdir(src_dir_p)
        if os.path.isdir(os.path.join(src_dir_p, name)) and name.startswith('Time')
    ]
    folder2_base = os.path.join(src_dir_p, folders_time[0])
    folders_2025_2 = [
        name for name in os.listdir(folder2_base)
        if os.path.isdir(os.path.join(folder2_base, name)) and name.startswith('2025')
    ]
    folder2_src = os.path.join(folder2_base, folders_2025_2[0])

    # list of full paths to copy
    files_to_copy = [
        # os.path.join(folder1_src, "Neon Scene Camera v1 ps1.mp4"),
        os.path.join(folder1_src, "Video_Timestamp.csv"),
        os.path.join(folder2_src,  "gaze.csv"),
        os.path.join(folder2_src,  "scene_camera.json"),
    ]


    # copy each file into dst_dir_p
    for src_path in files_to_copy:
        fname = os.path.basename(src_path)
        dst_path = os.path.join(dst_dir_p, fname)
        if os.path.exists(src_path):
            try:
                shutil.copy2(src_path, dst_path)
                print(f"[OK]  {src_path} → {dst_path}")
            except Exception as e:
                print(f"[ERR] copying {src_path}: {e}")
        else:
            print(f"[MISS] source not found: {src_path}")
