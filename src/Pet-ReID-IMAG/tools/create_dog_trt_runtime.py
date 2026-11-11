"""Create an isolated executable name for optional NVIDIA application settings."""
import os
from pathlib import Path
import shutil
import sys


def main():
    if os.name != 'nt' or sys.version_info[:2] != (3, 12):
        raise SystemExit('This launcher setup expects Windows and Python 3.12')
    directory = Path(sys.executable).parent
    base = Path(sys._base_executable)
    files = [(base, directory / 'dog_capture_trt.exe')]
    files += [(base.parent / name, directory / name) for name in ('python312.dll', 'python3.dll')]
    for source, destination in files:
        if not source.is_file():
            raise FileNotFoundError(source)
        if destination.exists() and destination.read_bytes() == source.read_bytes():
            continue
        shutil.copy2(source, destination)
    print(directory / 'dog_capture_trt.exe')


if __name__ == '__main__':
    main()
