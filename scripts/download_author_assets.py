#!/usr/bin/env python3
"""Helper for a normal local machine. The author's Google Drive folder is dynamic.
Run from a network/browser environment where Google Drive is accessible.
"""
from pathlib import Path
import argparse

FOLDER_URL = "https://drive.google.com/drive/folders/1_7pdSRTvD_XdTu8z0MxrM9PDoEuX-tjf?usp=drive_link"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="author_assets")
    args = ap.parse_args()
    try:
        import gdown
    except ImportError as e:
        raise SystemExit("Install gdown first: pip install gdown") from e
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    print("Downloading author folder:", FOLDER_URL)
    print("Destination:", out.resolve())
    result = gdown.download_folder(url=FOLDER_URL, output=str(out), quiet=False, use_cookies=True, remaining_ok=True)
    print("Downloaded entries:", len(result or []))
    if not result:
        print("No entries were returned. Open the Drive folder in a browser and download it manually if Google blocks folder enumeration.")

if __name__ == "__main__":
    main()
