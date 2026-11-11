"""Run independently of FastReID and the existing recognition service."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dog_capture.cli import main

if __name__ == "__main__":
    main()
