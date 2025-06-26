# import openai

# openai.api_key = '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'

# response = openai.audio.speech.create(
#     model="tts-1",  # or "tts-1-hd" for higher quality
#     voice="onyx",   # voices: "nova", "echo", "onyx", "fable", "shimmer", "alloy"
#     input="Hello! My name is tiago, What is your name? I am a humanoid robot."
# )

# # Save audio to file
# with open("output.mp3", "wb") as f:
#     f.write(response.content)



import cv2

# Path to your video file
video_path = '/Volumes/R@mtin/Year 2/Experiment/EyeTracker/Data/Participant 1/user reaction/video_part_2.mp4'

# Open the video
cap = cv2.VideoCapture(video_path)

# Frame you want to access
frame_number = 380
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

# Read the frame
ret, frame = cap.read()

if ret:
    # Get the time in milliseconds
    timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    timestamp_sec = timestamp_ms / 1000.0
    print(f"Time of frame {frame_number}: {timestamp_sec:.2f} seconds")

    # Show the frame
    cv2.imshow(f"Frame {frame_number}", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"Could not retrieve frame {frame_number}.")

cap.release()