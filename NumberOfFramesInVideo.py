import cv2

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

if __name__ == "__main__":
    video_path = '/Users/stabatabaeim/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/University/Year 2/Experiment/EyeTracker/Data/Participant 2/2025-03-19-17-04-21/Neon Scene Camera v1 ps1.mp4'
    total_frames, frame_rate = get_video_info(video_path)
    if total_frames != -1:
        print(f"Total number of frames in the video: {total_frames}")
        print(f"Frame rate (FPS) of the video: {frame_rate}")
