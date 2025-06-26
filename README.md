# Analysing_Eye_tracker

"Analysis_gaze_summary copy 3.py" is the last file of gaze data and the output is gaze features for each participant, like probability of gaze for each areas of interest and these kind of features. 

For each participant I have 14 files. 
1. Audio_summary.csv file which is the output of Whisper and anotehr model that i used to detect the laugher for each participant.

2. ee_state_data.csv which is a csv file, an dit is the output from rosbag, there i have Participant_id, Puzzle_number, object_number, failure_number, Verbal_number, state based on the timestamp.

3. gaze.csv file which is the output of eye tracker. i extractet it from the mobile phone. 

4. merged_gaze_state.csv it is a merged file which there i have the gaze value based on the robot state and the timestamp.

5. Neon Scene Camera v1 ps1.mp4 this is the output of eye tracker. i extractet it from the mobile phone.

6. region_checks.csv its a file where I have the Place of Interest based on the timestamp_ns and Video_time

7. scene_camera.json this is a file which i didnt used in my analysis, but there i have camera matrix.

8. Video_Gaze_Aruco.csv which is a file containing timestamp_ns, Video_time, gaze_x_px, gaze_y_px, aruco_1_id, aruco_1_x_px, aruco_1_y_px.

9. Video_Timestamp.csv which there i have a timestamp for each frame of the video.

So to start analysing, first we need to have the timestamp file for the each video (by knowing the exact timestamp of the first frame of the video we can also do the analysis, because there is a python code that based on the video frame per second can find the timestamp of other frames of the video. it is also important to know the video does not have an exact frame per second.)
so the first step is to run "FindFrameTimes.py" file.





