
import asyncio
import signal
from pupil_labs.realtime_api import Device, Network, StatusUpdateNotifier
from pupil_labs.realtime_api.models import Recording

async def print_recording(component):
    if isinstance(component, Recording):
        print(f"Update: {component.message}")

async def recording_stop_and_save(device):
    print("Stopping recording and saving...")
    await device.recording_stop_and_save()
    print("Recording stopped and saved.")

async def main():
    stop_event = asyncio.Event()

    def signal_handler(sig, frame):
        print("SIGINT received, stopping recording and saving...")
        stop_event.set()  # Set the stop event to break the main wait loop.

    signal.signal(signal.SIGINT, signal_handler)

    async with Network() as network:
        dev_info = await network.wait_for_new_device(timeout_seconds=5)

    if dev_info is None:
        print("No device could be found! Abort")
        return

    async with Device.from_discovered_device(dev_info) as device:
        notifier = StatusUpdateNotifier(device, callbacks=[print_recording])
        await notifier.receive_updates_start()
        recording_id = await device.recording_start()
        print(f"Initiated recording with id {recording_id}")

        # Wait for the stop event to be set.
        await stop_event.wait()
        await recording_stop_and_save(device)

if __name__ == "__main__":
    asyncio.run(main())
