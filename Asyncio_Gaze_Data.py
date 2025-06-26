import asyncio
import contextlib
from datetime import datetime
from os import makedirs
from pupil_labs.realtime_api import Device, Network, receive_gaze_data
import csv
import pytz

def Converter(timestamp):

    # Create a UTC datetime object from the timestamp
    utc_dt = datetime.utcfromtimestamp(timestamp).replace(tzinfo=pytz.utc)

    # Convert the UTC datetime object to Melbourne's time zone
    melbourne_tz = pytz.timezone('Australia/Melbourne')
    melbourne_dt = utc_dt.astimezone(melbourne_tz)

    # Format the datetime object as a string
    dt_formatted = melbourne_dt.strftime('%H:%M:%S.%f')
    return dt_formatted


async def main():
    async with Network() as network:
        dev_info = await network.wait_for_new_device(timeout_seconds=5)
    if dev_info is None:
        print("No device could be found! Abort")
        return
    
    async with Device.from_discovered_device(dev_info) as device:
        print(f"Getting status information from {device}")
        status = await device.get_status()

    async with Device.from_discovered_device(dev_info) as device:
        status = await device.get_status()
        sensor_gaze = status.direct_gaze_sensor()
        if not sensor_gaze.connected:
            print(f"Gaze sensor is not connected to {device}")
            return

        restart_on_disconnect = True

        current_time_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"csv_files/gaze_data_{current_time_str}.csv"
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            headers = ['Timestamp unix Second', 'Gaze_X', 'Gaze_Y', "Time"]
            writer.writerow(headers)

            async for gaze in receive_gaze_data(
                sensor_gaze.url, run_loop=restart_on_disconnect
            ):
                # print(gaze.x)

                # Write the current frame's datetime and each element of Aruco_array in its own column
                writer.writerow([float(gaze.timestamp_unix_seconds), float(gaze.x), float(gaze.y), Converter(float(gaze.timestamp_unix_seconds))])

                # If you are running other async tasks, it's good to yield with asyncio.sleep(0)
                await asyncio.sleep(0)


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())