import shutil
from pathlib import Path
import os

def sync_directories(src_dir: Path, dst_dir: Path):
    """
    Compare two directories and copy files that exist in src_dir
    but are missing in dst_dir into dst_dir.
    """

    # Ensure source and destination directories exist
    if not src_dir.is_dir():
        raise ValueError(f"Source directory {src_dir} does not exist or is not a directory.")
    if not dst_dir.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)

    # List all files in source (non-recursive)
    src_files = {
        f.name for f in src_dir.iterdir()
        if f.is_file() and not f.name.startswith("._") and not f.name.startswith(".")
    }
    dst_files = {
        f.name for f in dst_dir.iterdir()
        if f.is_file() and not f.name.startswith("._") and not f.name.startswith(".")
    }

    # Determine which files are missing in destination
    missing_files = src_files - dst_files

    if not missing_files:
        print("No files to copy; destination is already up to date.")
        return

    # Copy missing files
    for filename in missing_files:
        src_file = src_dir / filename
        dst_file = dst_dir / filename
        shutil.copy2(src_file, dst_file)
        print(f"Copied: {src_file} -> {dst_file}")



def sync_directories_2(src_dir: Path, dst_dir: Path):
    """
    Compare two directories and copy files that exist in src_dir
    but are missing in dst_dir into dst_dir.
    """
    entries_2 = [
        name for name in os.listdir(src_dir)
        if not name.startswith("._") and name.startswith("2024")
    ]

    path_src_2 = src_dir / entries_2[0]
    path_dst_2 = dst_dir / entries_2[0]

    # Ensure source and destination directories exist
    if not path_src_2.is_dir():
        raise ValueError(f"Source directory {path_src_2} does not exist or is not a directory.")
    if not path_dst_2.exists():
        path_dst_2.mkdir(parents=True, exist_ok=True)

    # List all files in source (non-recursive)
    src_files = {
        f.name for f in path_src_2.iterdir()
        if f.is_file() and not f.name.startswith("._") and not f.name.startswith(".")
    }
    dst_files = {
        f.name for f in path_dst_2.iterdir()
        if f.is_file() and not f.name.startswith("._") and not f.name.startswith(".")
    }

    # Determine which files are missing in destination
    missing_files = src_files - dst_files

    if not missing_files:
        print("No files to copy; destination is already up to date.")
        return

    # Copy missing files
    for filename in missing_files:
        src_file = path_src_2 / filename
        dst_file = path_dst_2 / filename
        shutil.copy2(src_file, dst_file)
        print(f"Copied: {src_file} -> {dst_file}")

if __name__ == "__main__":

    path = "/Volumes/R@mtin/User Study/"
    entries = [
        name for name in os.listdir(path)
        if not name.startswith("._") and name.startswith("2024")
    ]
    for dir in entries:
        source = Path(f'/Volumes/R@mtin/User Study/{dir}')
        destination = Path(f'/Volumes/4180-human-robot-interaction/Ramtin/Study 1/{dir}')
        sync_directories(source, destination)
        sync_directories_2(source, destination)
        print(source)