import asyncio
import subprocess

async def run_script(script_name):
    # Run a script using subprocess and await its completion
    process = await asyncio.create_subprocess_exec(
        'python3', script_name,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

    # Wait for the script to complete and capture stdout and stderr
    stdout, stderr = await process.communicate()

    if process.returncode == 0:
        print(f"{script_name}: Completed successfully")
        if stdout:
            print(f"Output:\n{stdout.decode()}")
    else:
        print(f"{script_name}: Error")
        if stderr:
            print(f"Error Message:\n{stderr.decode()}")

async def main():
    # List of script names to run
    scripts = ['EyeTrackerExperiment.py', 'Asyncio_Gaze_Data.py', 'Asyncio_Recording.py']

    # Create a list of tasks to run the scripts asynchronously
    tasks = [run_script(script) for script in scripts]

    # Run the tasks concurrently
    await asyncio.gather(*tasks)

if __name__ == '__main__':
    asyncio.run(main())
