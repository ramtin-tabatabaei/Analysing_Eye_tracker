from moviepy.editor import VideoFileClip
import pandas as pd

def extract_audio_by_timestamps(
    video_path: str,
    video_start_ts: float,
    event_start_ts: float,
    event_end_ts: float,
    output_audio_path: str
):
    """
    Extract the audio between event_start_ts and event_end_ts,
    given the absolute start timestamp of the video itself.
    """
    # Load video
    clip = VideoFileClip(video_path)

    # Compute relative times (in seconds)
    t_start = event_start_ts - video_start_ts
    t_end   = event_end_ts   - video_start_ts

    print(t_start)
    print(t_end)

    # Guard against out‐of‐bounds
    t_start = max(t_start, 0)
    t_end   = min(t_end, clip.duration)

    # Extract that audio segment
    audio_segment = clip.audio.subclip(t_start, t_end)

    # Write to file
    audio_segment.write_audiofile(output_audio_path)

    clip.close()

def failure_type_founder(failure):
    if failure == 1:
        return "Executional"
    elif failure == 2:
        return "Decisional"
    elif failure == 3:
        return "Mechanical"

# Example usage:
if __name__ == "__main__":
    for i in range(13,55):
        if i==3 or i == 9 or i==47:
            continue
        main_path = f'/Volumes/R@mtin/Year 2/Experiment/EyeTracker/Data/Participant {i}'
        video_file  = main_path + "/"+ 'Neon Scene Camera v1 ps1.mp4'
        # Suppose the video file was recorded at Unix time 1,700,000,000.0
        Video_StartTime_file = main_path + "/"+ "Video_Timestamp.csv"
        Robot_state_file = main_path + "/" + 'ee_state_data.csv'

        Video_Time_file_data = pd.read_csv(Video_StartTime_file)
        Robot_state_data = pd.read_csv(Robot_state_file)

        # print(Robot_state_data)

        Video_StartTime = Video_Time_file_data.iloc[0,0]
        Video_endTime = Video_Time_file_data.iloc[-1,0]


        
        video_start_ts  = Video_StartTime/10**9
        video_end_ts  = Video_endTime/10**9

        for failure_number in (2, 4, 6):
            failure = Robot_state_data[
                        (Robot_state_data["Puzzle_number"] == failure_number) &
                        (Robot_state_data["failure_number"] >0)
                    ]
            
            failure_type = failure_type_founder(failure["failure_number"].iloc[0])

            # And your failure event ran from these absolute timestamps:
            event_start_ts  = failure["timestamp"].iloc[0]
            event_end_ts    = failure["timestamp"].iloc[-1]+15

            if event_start_ts > video_start_ts and event_start_ts < video_end_ts:

                extract_audio_by_timestamps(
                    video_file,
                    video_start_ts,
                    event_start_ts,
                    event_end_ts,
                    main_path + "/failure" + str(failure_number/2) +"_"+ failure_type + "_audio.wav"
                )
