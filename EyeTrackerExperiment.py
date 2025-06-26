import asyncio
import contextlib
import typing as T
import numpy as np
import cv2
import csv
from datetime import datetime
from os import makedirs


# Set up the camera matrix and distortion coefficients from your data
camera_matrix = np.array([
    [882.9359938422125, 0.0, 811.9703347396303],
    [0.0, 881.8842459552434, 599.9935777910313],
    [0.0, 0.0, 1.0]
])

dist_coeffs = np.array([
    -0.1304563693410277,
    0.10867382696023291,
    -0.00021961653898858047,
    -0.0002471077823448382,
    -6.211241152523251e-08,
    0.17090923197472804,
    0.05044953820912044,
    0.025482393605275936
])

from pupil_labs.realtime_api import (
    Device,
    Network,
    receive_gaze_data,
    receive_video_frames,
)


async def main():
    async with Network() as network:
        dev_info = await network.wait_for_new_device(timeout_seconds=5)
    if dev_info is None:
        print("No device could be found! Abort")
        return

    async with Device.from_discovered_device(dev_info) as device:
        print(f"Getting status information from {device}")
        status = await device.get_status()

        sensor_gaze = status.direct_gaze_sensor()
        if not sensor_gaze.connected:
            print(f"Gaze sensor is not connected to {device}")
            return

        sensor_world = status.direct_world_sensor()
        if not sensor_world.connected:
            print(f"Scene camera is not connected to {device}")
            return

        restart_on_disconnect = True

        queue_video = asyncio.Queue()
        queue_gaze = asyncio.Queue()

        process_video = asyncio.create_task(
            enqueue_sensor_data(
                receive_video_frames(sensor_world.url, run_loop=restart_on_disconnect),
                queue_video,
            )
        )
        process_gaze = asyncio.create_task(
            enqueue_sensor_data(
                receive_gaze_data(sensor_gaze.url, run_loop=restart_on_disconnect),
                queue_gaze,
            )
        )
        try:
            await match_and_draw(queue_video, queue_gaze)
        finally:
            process_video.cancel()
            process_gaze.cancel()

# def detect_markers(frame, aruco_dict, parameters):
#     # Convert to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # Detect markers
#     corners, ids, rejectedCandidates = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
#     Aruco_array = []
#     if ids is not None:
#         # print(np.array(ids).shape[0])
#         ids_array = np.array(ids)
#         Aruco_array = np.zeros(ids_array.shape[0]*3)
#         # Aruco_array = np.zeros(30)
#         for i in range(ids_array.shape[0]):
#             X_avg = int((corners[i][0][0][0]+corners[i][0][1][0])/2)
#             Y_avg = int((corners[i][0][0][1]+corners[i][0][2][1])/2)
#             Aruco_array[3*i] = ids_array[i,0]
#             Aruco_array[3*i+1] = X_avg
#             Aruco_array[3*i+2] = Y_avg 
#             # print(f"ID: {ids_array[i,0]}, Location (x, y): ({X_avg}, {Y_avg})")


#     frame_marked = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, ids)
#     return frame_marked, ids, Aruco_array


def detect_markers(frame, aruco_dict, parameters, marker_length=0.05):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect markers
    corners, ids, rejectedCandidates = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    Aruco_array = []

    if ids is not None:
        # Draw markers on frame
        frame_marked = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, ids)

        # Estimate pose of each marker
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

        # Loop through each detected marker
        for i, id_ in enumerate(ids):
            X_avg = int((corners[i][0][0][0] + corners[i][0][1][0]) / 2)
            Y_avg = int((corners[i][0][0][1] + corners[i][0][2][1]) / 2)
            depth = tvecs[i][0][2]  # Z-axis in tvec represents the distance (depth) from the camera to the marker

            Aruco_array.append([id_[0], X_avg, Y_avg, depth])  # Save ID, X, Y, and depth

            # Draw depth info on the frame
            cv2.putText(frame_marked, f"Depth: {depth:.2f}m", (X_avg, Y_avg+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            #cv2.aruco.drawAxis(frame_marked, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.1)

        return frame_marked, ids, np.array(Aruco_array)

    return frame, ids, Aruco_array

async def enqueue_sensor_data(sensor: T.AsyncIterator, queue: asyncio.Queue) -> None:
    async for datum in sensor:
        try:
            queue.put_nowait((datum.datetime, datum))
        except asyncio.QueueFull:
            print(f"Queue is full, dropping {datum}")


async def match_and_draw(queue_video, queue_gaze):

    # Format the current datetime into a string that is safe for filenames
    current_time_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"csv_files/aruco_data_{current_time_str}.csv"
    base_save_path = f"Frames_{current_time_str}"
    makedirs(base_save_path, exist_ok=True)
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        headers = ['Time', 'Timestamp unix Second', 'Gaze_X', 'Gaze_Y']
        for i in range(1,81):
            headers.append(f"Aruco{i}_ID")
            headers.append(f'Aruco{i}_X')
            headers.append(f'Aruco{i}_Y')
            headers.append(f'Aruco{i}_Z')
        writer.writerow(headers)
   
        while True:
            video_datetime, video_frame = await get_most_recent_item(queue_video)
            #print(video_datetime)
            _, gaze_datum = await get_closest_item(queue_gaze, video_datetime)

            bgr_buffer = video_frame.to_ndarray(format="bgr24")

            # Detect ArUco markers
            bgr_buffer , ids, Aruco_array = detect_markers(bgr_buffer , aruco_dict, parameters)

            draw_time(bgr_buffer, video_datetime)


            cv2.circle(
                bgr_buffer,
                (int(gaze_datum.x), int(gaze_datum.y)),
                radius=40,
                color=(0, 0, 255),
                thickness=15,
            )

            cv2.imshow("Scene camera with gaze overlay", bgr_buffer)
            cv2.waitKey(1)


            frame_save_path = f"{base_save_path}/{video_datetime.strftime('%Y-%m-%d_%H-%M-%S-%f')}.jpg"
            cv2.imwrite(frame_save_path, bgr_buffer)

            # aruco_data = list(map(float, Aruco_array.split(',')))
            Aruco_array = np.array(Aruco_array).flatten()

            print(video_datetime.timestamp())

            # Write the current frame's datetime and each element of Aruco_array in its own column
            writer.writerow([video_datetime.strftime('%H:%M:%S.%f'), video_datetime.timestamp(), int(gaze_datum.x), int(gaze_datum.y), *Aruco_array])

            # If you are running other async tasks, it's good to yield with asyncio.sleep(0)
            await asyncio.sleep(0)



async def get_most_recent_item(queue):
    item = await queue.get()
    while True:
        try:
            next_item = queue.get_nowait()
        except asyncio.QueueEmpty:
            return item
        else:
            item = next_item


async def get_closest_item(queue, timestamp):
    item_ts, item = await queue.get()
    # assumes monotonically increasing timestamps
    if item_ts > timestamp:
        return item_ts, item
    while True:
        try:
            next_item_ts, next_item = queue.get_nowait()
        except asyncio.QueueEmpty:
            return item_ts, item
        else:
            if next_item_ts > timestamp:
                return next_item_ts, next_item
            item_ts, item = next_item_ts, next_item

def draw_time(frame, time):
    frame_txt_font_name = cv2.FONT_HERSHEY_SIMPLEX
    frame_txt_font_scale = 1.0
    frame_txt_thickness = 3

    # first line: frame index
    frame_txt = str(time)

    cv2.putText(
        frame,
        frame_txt,
        (20, 50),
        frame_txt_font_name,
        frame_txt_font_scale,
        (0, 0, 0),
        thickness=frame_txt_thickness,
        lineType=cv2.LINE_8,
    )


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        # Define the ArUco dictionary
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        parameters = cv2.aruco.DetectorParameters()
        asyncio.run(main())