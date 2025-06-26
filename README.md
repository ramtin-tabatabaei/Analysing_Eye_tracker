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

Second we need to run the VideoFindArucos.py to find the arucomarker positions and make the Video_Gaze_Aruco.csv file. 

state_checker.py is just a file to check if any value from the robot states are missing or not. 

"ImageAurcodetecter.py" I used this file to annotate how the aruco makers are related to the gaze.

Then I used the AOI_detecter.py file to use the Video_Gaze_Aruco.csv and make a new file which tells me where the participants are looking at.

Then I used the file fill_gaze_gaps.py to put a threshold, meaning if for example if a value is missing by mistake dont consider that as a gaze shift, fill it with the same value of the previous and after one. 

"VideoAnalysis.py" The I run this file to check if the areas of interests are detected correctly, there i show the gaze and areas of interest and also i print the place that they are looking at.

Then I used the file "merge_gaze_with_state.py" to merged the gaze data and the states that i have. the output is "merged_gaze_state.csv'" 


In the end we can run "Analysis_gaze_summary copy.py" that it will make the gaze featues and save it for each participant, or we can run "Analysis_gaze_summary copy3.py" which it saves all data for all participant in one csv file. 



## 🧠 **Eye Tracker Analysis Pipeline Overview**

For each **participant**, you have **14 files** used throughout the analysis. Below is a breakdown of what each file is, what it contains, and how the overall analysis proceeds.

---

### 📁 **Participant Files Explained**

| #     | File Name                                              | Description                                                                                                                                                   |
| ----- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | `Audio_summary.csv`                                    | Output from Whisper and another model used to detect **laughter** timestamps.                                                                                 |
| 2     | `ee_state_data.csv`                                    | ROS-generated CSV containing: `Participant_id`, `Puzzle_number`, `Object_number`, `Failure_number`, `Verbal_number`, and **robot states** (all time-stamped). |
| 3     | `gaze.csv`                                             | Raw eye-tracker output (gaze data) extracted from the mobile phone.                                                                                           |
| 4     | `merged_gaze_state.csv`                                | Merged file aligning **gaze data with robot state** using timestamps.                                                                                         |
| 5     | `Neon Scene Camera v1 ps1.mp4`                         | Raw **video recording** from the eye tracker (mobile phone).                                                                                                  |
| 6     | `region_checks.csv`                                    | Contains **Places of Interest (AOIs)** with `timestamp_ns` and `Video_time`.                                                                                  |
| 7     | `scene_camera.json`                                    | Contains **camera calibration parameters** (`camera_matrix`, distortion); not used directly in analysis.                                                      |
| 8     | `Video_Gaze_Aruco.csv`                                 | Contains **gaze data with ArUco marker detection**: timestamps, gaze positions, marker IDs and their positions.                                               |
| 9     | `Video_Timestamp.csv`                                  | Stores the **timestamp of each video frame**, crucial for syncing video with gaze.                                                                            |
| 10–14 | (Other scripts & intermediate data used in processing) |                                                                                                                                                               |

---

### 🔄 **Analysis Workflow Steps**

#### 🔹 **Step 1: Generate Video Frame Timestamps**

* **Script**: `FindFrameTimes.py`
* **Purpose**: Determines the exact timestamp for each frame using the known timestamp of the **first frame** and the (non-uniform) frame rate.

---

#### 🔹 **Step 2: Detect ArUco Markers in Video**

* **Script**: `VideoFindArucos.py`
* **Purpose**: Detects **ArUco markers** frame-by-frame and creates `Video_Gaze_Aruco.csv`, combining:

  * Gaze coordinates
  * Marker IDs and positions
  * Timestamps

---

#### 🔹 **Step 3: Check State Data**

* **Script**: `state_checker.py`
* **Purpose**: Verifies the **completeness of robot state data** (no missing values).

---

#### 🔹 **Step 4: Manually Annotate Marker Meaning**

* **Script**: `ImageAurcodetecter.py`
* **Purpose**: Manually annotate how each **ArUco marker relates to gaze** or areas of interest (AOIs).

---

#### 🔹 **Step 5: Detect Areas of Interest (AOIs)**

* **Script**: `AOI_detecter.py`
* **Purpose**: Uses the `Video_Gaze_Aruco.csv` to determine **where participants are looking** and associates it with predefined AOIs.

---

#### 🔹 **Step 6: Fill Gaps in Gaze Data**

* **Script**: `fill_gaze_gaps.py`
* **Purpose**: Applies a **temporal threshold** to fill short missing values in gaze data (treats short gaps as continuity, not shifts).

---

#### 🔹 **Step 7: Visually Verify AOI Detection**

* **Script**: `VideoAnalysis.py`
* **Purpose**: Visual inspection tool showing:

  * Video frames with **gaze points and AOIs**
  * **Prints the AOI** the participant is looking at
* Useful for debugging and validation.

---

#### 🔹 **Step 8: Merge Gaze with Robot State**

* **Script**: `merge_gaze_with_state.py`
* **Purpose**: Creates `merged_gaze_state.csv` by aligning **gaze data with robot states** using timestamps.

---

#### 🔹 **Step 9: Extract Gaze Features**

* **Scripts**:

  * `Analysis_gaze_summary copy.py`: Outputs **individual CSV files per participant** with extracted gaze features (e.g., AOI probabilities).
  * `Analysis_gaze_summary copy3.py`: Outputs **one combined CSV** with features for **all participants**.











