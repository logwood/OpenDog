"""``python -m dog_capture`` -> the v1 engine CLI (legacy v0.x tools stay in tools/)."""
import sys

from dog_capture.engine.cli import main

if __name__ == '__main__':
    sys.exit(main())
