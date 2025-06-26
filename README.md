# Analysing_Eye_tracker

## 🧠 **Eye Tracker Analysis Pipeline Overview**

For each **participant**, you have **9 files** used throughout the analysis. Below is a breakdown of what each file is, what it contains, and how the overall analysis proceeds.

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
| 9     | `Video_Timestamp.csv`                                  | Stores the **timestamp of each video frame**, crucial for syncing video with gaze.                                                                                                                                                                                                                                     |

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











