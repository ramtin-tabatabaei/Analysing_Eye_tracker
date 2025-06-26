import pandas as pd
import cv2
import numpy as np
import cv2.aruco as aruco
import json
import os


def read_number_from_csv(file_path, column_name, row_index):
    """
    Reads a number from a given CSV file at a specific column and row index.
    :param file_path: Path to the CSV file
    :param column_name: Name of the column to read from
    :param row_index: Index of the row to read
    :return: The number read from the CSV file
    """
    try:
        df = pd.read_csv(file_path)
        if column_name not in df.columns:
            print(f"Error: Column '{column_name}' not found in the CSV file.")
            return None
        
        if row_index >= len(df):
            print("Error: Row index out of range.")
            return None
        
        number = df.at[row_index, column_name]
        return number
    except Exception as e:
        print(f"Error: {e}")
        return None
    
def get_video_info(video_path):
    """
    Returns the total number of frames and frame rate of a given video file.
    :param video_path: Path to the video file
    :return: Total number of frames and frame rate
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return -1, -1
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return total_frames, frame_rate

def find_closest_timestamp(df, target_value):
    """
    Finds the closest timestamp to a given number.
    :param df: DataFrame containing the timestamp column
    :param target_value: The number to compare against the timestamps
    :return: Closest timestamp
    """
    closest_index = (df["timestamp [ns]"].sub(target_value)).abs().idxmin()
    closest_value = df["timestamp [ns]"].iloc[closest_index]
    return closest_value, closest_index




def draw_gaze_points_and_aruco_on_video(video_path, gaze_data, Video_Time, camera_matrix, dist_coeffs, out_csv):
    """
    Draws gaze points and detects ArUco markers on each frame of the video.
    :param video_path: Path to the video file
    :param gaze_data: DataFrame containing gaze x and y coordinates with timestamps
    :param camera_matrix: Camera matrix for distortion correction
    :param dist_coeffs: Distortion coefficients
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    Video_StartTime = Video_Time.iloc[0,0]
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    
    aruco_dict = aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    aruco_params = aruco.DetectorParameters()

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # output_path = '/Users/stabatabaeim/Univerisity/Year 2/Experiment/EyeTracker/Recorded_Files/MobilePhone/Yushan/output_video.mp4'
    # out = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height))

    rows = []   # will hold one list per frame
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
        
        # if ids is not None:
        #     aruco.drawDetectedMarkers(frame, corners, ids)
        
        frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Convert ms to seconds
        

        elapsed_time = timestamp
        # cv2.putText(frame, f"Time: {elapsed_time:.2f} sec", (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        target_value = Video_StartTime +(10**9)*timestamp
        # target_value = Video_Time.iloc[frame_index-1,0]

        # print(Video_Time.iloc[frame_index-1,0]*10**(-6)-target_value*10**(-6))

        closest_timestamp, closest_index = find_closest_timestamp(Gaze_data, target_value)
        # print(f"The closest timestamp to {target_value} is {closest_timestamp} at index {closest_index}")
        # print(gaze_data[["gaze x [px]", "gaze y [px]"]].iloc[closest_index])

        gaze_x = int(gaze_data["gaze x [px]"].iloc[closest_index])
        gaze_y = int(gaze_data["gaze y [px]"].iloc[closest_index])

        pt = np.array([[[gaze_x, gaze_y]]], dtype=np.float32)

        # undistortPoints maps it into normalized coordinates; by providing
        # P=camera_matrix we ask for pixel output in the undistorted image:
        undist_pt = cv2.undistortPoints(pt, camera_matrix, dist_coeffs, P=camera_matrix)

        # Extract back to ints
        ux, uy  = undist_pt[0,0]
        gaze_x, gaze_y = int(round(ux)), int(round(uy))

        # start building this row
        row = [target_value, elapsed_time, gaze_x, gaze_y]

        # append each detected marker’s id + center‐x + center‐y
        if ids is not None:
            for idx, mid in enumerate(ids.flatten()):
                pts = corners[idx].reshape(-1, 2)
                cx  = int(pts[:,0].mean())
                cy  = int(pts[:,1].mean())
                row += [int(mid), cx, cy]

        rows.append(row)

        # pt = np.array([[[gaze_x, gaze_y]]], dtype=np.float32)

        # undistortPoints maps it into normalized coordinates; by providing
        # P=camera_matrix we ask for pixel output in the undistorted image:
        # undist_pt = cv2.undistortPoints(pt, camera_matrix, dist_coeffs, P=camera_matrix)

        # Extract back to ints
        # ux, uy  = undist_pt[0,0]
        # ux, uy = int(round(ux)), int(round(uy))

        # Now draw using the undistorted coordinates
        # cv2.circle(frame, (ux, uy), 20, (255, 255, 255), 10)

        
        # # Ensure gaze coordinates are within frame boundaries
        # if 0 <= gaze_x < frame_width and 0 <= gaze_y < frame_height:
        #     cv2.circle(frame, (int(gaze_x), int(gaze_y)), 20, (0, 0, 255), 10)

        # out.write(frame)
        # cv2.imshow('Video with Gaze Points and ArUco Markers', frame)
        # if elapsed_time > 40:
        #     break
        print(frame_index/total_frames)
        # if cv2.waitKey(int(1000/frame_rate)) & 0xFF == ord('q'):
        #     break
    
    cap.release()

    # now turn rows into a DataFrame and write CSV
    # figure out the maximum number of markers seen in any frame
    max_markers = max((len(r)-4)//3 for r in rows) if rows else 0

    # build header: ts, gaze_x, gaze_y, then for each marker slot: id, x, y
    cols = ["timestamp_ns", "Video_time", "gaze_x_px", "gaze_y_px"]
    for i in range(1, max_markers+1):
        cols += [f"aruco_{i}_id", f"aruco_{i}_x_px", f"aruco_{i}_y_px"]

    # pad each row to same length
    full_width = 4 + 3*max_markers
    for r in rows:
        r += [None] * (full_width - len(r))

    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} rows to {out_csv!r}")


    # out.release()
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    path = '/Volumes/R@mtin/Year 2/Experiment/EyeTracker/Data'

    folders_Paricipants = [name for name in os.listdir(path)
    if os.path.isdir(os.path.join(path, name)) and name.startswith('Participant')]
    
    for participant_number in folders_Paricipants:
        print(participant_number)

        main_path = path + "/" + participant_number
        video_path = main_path + "/"+ "Neon Scene Camera v1 ps1.mp4"
        total_frames, frame_rate = get_video_info(video_path)

        # Video_StartTime_file_path = '/Users/stabatabaeim/Univerisity/Year 2/Experiment/EyeTracker/Recorded_Files/MobilePhone/Yushan/Timeseries Data + Scene Video/2025-02-20_16-07-41-e75e47d1/events.csv'
        # Video_StartTime = read_number_from_csv(Video_StartTime_file_path, "timestamp [ns]", 0)
        Video_StartTime_file = main_path + "/"+ "Video_Timestamp.csv"
        Video_Time_file_data = pd.read_csv(Video_StartTime_file)
        # print(Video_StartTime_file_data.iloc[0,0])

        Gaze_file_path = main_path + "/"+ "gaze.csv"
        Gaze_data = pd.read_csv(Gaze_file_path, usecols=["timestamp [ns]", "gaze x [px]", "gaze y [px]"])

        # Path to your JSON file
        json_path = main_path + "/"+ "scene_camera.json"

        # 1. Load the JSON
        with open(json_path, 'r') as f:
            data = json.load(f)

        # 2. Extract and convert
        camera_matrix        = np.array(data['camera_matrix'])
        distortion_coefficients    = np.array(data['distortion_coefficients'])

        draw_gaze_points_and_aruco_on_video(video_path, Gaze_data, Video_Time_file_data, camera_matrix, distortion_coefficients, out_csv='/Users/stabatabaeim/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/University/Year 2/Experiment/EyeTracker/Data/Participant 10/Video_Gaze_Aruco.csv')



