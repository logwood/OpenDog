"""Load a dedicated Windows Python executable without changing system PATH."""
import os
from pathlib import Path
import runpy
import sys

handles = []
if os.name == 'nt':
    for path in (Path(sys.base_prefix), Path(sys.base_prefix) / 'Library/bin', Path(sys.base_prefix) / 'DLLs'):
        if path.is_dir():
            handles.append(os.add_dll_directory(str(path)))
if len(sys.argv) < 2:
    raise SystemExit('Usage: dog_capture_trt.exe bootstrap_dog_trt.py SCRIPT [ARGS]')
script = sys.argv.pop(1)
sys.argv[0] = script
runpy.run_path(script, run_name='__main__')
