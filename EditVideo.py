# import pandas as pd
# import cv2
# import numpy as np
# import cv2.aruco as aruco


# def read_number_from_csv(file_path, column_name, row_index):
#     """
#     Reads a number from a given CSV file at a specific column and row index.
#     :param file_path: Path to the CSV file
#     :param column_name: Name of the column to read from
#     :param row_index: Index of the row to read
#     :return: The number read from the CSV file
#     """
#     try:
#         df = pd.read_csv(file_path)
#         if column_name not in df.columns:
#             print(f"Error: Column '{column_name}' not found in the CSV file.")
#             return None
        
#         if row_index >= len(df):
#             print("Error: Row index out of range.")
#             return None
        
#         number = df.at[row_index, column_name]
#         return number
#     except Exception as e:
#         print(f"Error: {e}")
#         return None
    
# def get_video_info(video_path):
#     """
#     Returns the total number of frames and frame rate of a given video file.
#     :param video_path: Path to the video file
#     :return: Total number of frames and frame rate
#     """
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print("Error: Could not open video.")
#         return -1, -1
    
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     frame_rate = cap.get(cv2.CAP_PROP_FPS)
#     cap.release()
#     return total_frames, frame_rate

# def find_closest_timestamp(df, target_value):
#     """
#     Finds the closest timestamp to a given number.
#     :param df: DataFrame containing the timestamp column
#     :param target_value: The number to compare against the timestamps
#     :return: Closest timestamp
#     """
#     closest_index = (df["timestamp [ns]"].sub(target_value)).abs().idxmin()
#     closest_value = df["timestamp [ns]"].iloc[closest_index]
#     return closest_value, closest_index




# def draw_gaze_points_and_aruco_on_video(video_path, gaze_data, Video_StartTime, camera_matrix, dist_coeffs):
#     """
#     Draws gaze points and detects ArUco markers on each frame of the video.
#     :param video_path: Path to the video file
#     :param gaze_data: DataFrame containing gaze x and y coordinates with timestamps
#     :param camera_matrix: Camera matrix for distortion correction
#     :param dist_coeffs: Distortion coefficients
#     """
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print("Error: Could not open video.")
#         return
    
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     frame_rate = cap.get(cv2.CAP_PROP_FPS)
    
#     aruco_dict = aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
#     aruco_params = aruco.DetectorParameters()

#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     output_path = '/Users/stabatabaeim/Univerisity/Year 2/Experiment/EyeTracker/Recorded_Files/MobilePhone/Yushan/output_video.mp4'
#     out = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height))
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
        
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
        
#         if ids is not None:
#             aruco.drawDetectedMarkers(frame, corners, ids)
        
#         frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
#         timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Convert ms to seconds

#         elapsed_time = timestamp
#         cv2.putText(frame, f"Time: {elapsed_time:.2f} sec", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
#         target_value = Video_StartTime +(10**9)*timestamp
#         closest_timestamp, closest_index = find_closest_timestamp(Gaze_data, target_value)
#         # print(f"The closest timestamp to {target_value} is {closest_timestamp} at index {closest_index}")
#         print(gaze_data[["gaze x [px]", "gaze y [px]"]].iloc[closest_index])

#         gaze_x = int(gaze_data["gaze x [px]"].iloc[closest_index])
#         gaze_y = int(gaze_data["gaze y [px]"].iloc[closest_index])
        
#         # Ensure gaze coordinates are within frame boundaries
#         if 0 <= gaze_x < frame_width and 0 <= gaze_y < frame_height:
#             cv2.circle(frame, (int(gaze_x), int(gaze_y)), 20, (0, 0, 255), 10)

#         out.write(frame)
#         cv2.imshow('Video with Gaze Points and ArUco Markers', frame)
#         if elapsed_time > 40:
#             break
#         if cv2.waitKey(int(1000/frame_rate)) & 0xFF == ord('q'):
#             break
    
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()




# if __name__ == "__main__":
#     video_path = '/Users/stabatabaeim/Univerisity/Year 2/Experiment/EyeTracker/Recorded_Files/MobilePhone/Yushan/Timeseries Data + Scene Video/2025-02-20_16-07-41-e75e47d1/e2336cc7_0.0-3142.377.mp4'
#     total_frames, frame_rate = get_video_info(video_path)
#     # if total_frames != -1:
#     #     print(f"Total number of frames in the video: {total_frames}")
#     #     print(f"Frame rate (FPS) of the video: {frame_rate}")

#     Video_StartTime_file_path = '/Users/stabatabaeim/Univerisity/Year 2/Experiment/EyeTracker/Recorded_Files/MobilePhone/Yushan/Timeseries Data + Scene Video/2025-02-20_16-07-41-e75e47d1/events.csv'
#     Video_StartTime = read_number_from_csv(Video_StartTime_file_path, "timestamp [ns]", 0)

#     # if Video_StartTime is not None:
#     #     print(f"The number read from the CSV file is: {Video_StartTime}")

#     Gaze_file_path = '/Users/stabatabaeim/Univerisity/Year 2/Experiment/EyeTracker/Recorded_Files/MobilePhone/Yushan/Timeseries Data + Scene Video/2025-02-20_16-07-41-e75e47d1/gaze.csv'
#     Gaze_data = pd.read_csv(Gaze_file_path, usecols=["timestamp [ns]", "gaze x [px]", "gaze y [px]"])
#     # print(Gaze_data)

#     # target_value = 1740028061204000000 +94227*(10**9)*1/frame_rate
#     # closest_timestamp, closest_index = find_closest_timestamp(Gaze_data, target_value)
#     # print(f"The closest timestamp to {target_value} is {closest_timestamp} at index {closest_index}")
#     # print(Gaze_data[["gaze x [px]", "gaze y [px]"]].iloc[closest_index])
#     # print(int(Gaze_data["gaze x [px]"].iloc[closest_index]))

#     camera_matrix = np.array([[882.9359938422125, 0.0, 811.9703347396303], [0.0, 881.8842459552434, 599.9935777910313], [0.0, 0.0, 1.0]])
#     distortion_coefficients = np.array([-0.1304563693410277, 0.10867382696023291, -0.00021961653898858047, -0.0002471077823448382, -6.211241152523251e-08, 0.17090923197472804, 0.05044953820912044, 0.025482393605275936])

#     draw_gaze_points_and_aruco_on_video(video_path, Gaze_data, Video_StartTime, camera_matrix, distortion_coefficients)





    
