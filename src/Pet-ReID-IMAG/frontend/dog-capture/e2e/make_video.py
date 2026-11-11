"""Build fake-camera footage (Y4M) from real dog photos with hand-held jitter.

    python e2e/make_video.py --front front.png --side side.png --side-face-x 520 --out e2e/media
writes front.y4m (frontal dog, 6 s) and sides.y4m (a 3/4 view, then its mirror: left then right
of the same dog, face centred so the track continues).
"""
import argparse
from pathlib import Path
import subprocess

import cv2
import numpy as np

W, H, FPS = 1280, 720, 30


def place(image, center_x=None, scale=1.0, dx=0.0, dy=0.0):
    h, w = image.shape[:2]
    s = min(W / w, H / h) * 0.95 * scale
    cx = w / 2 if center_x is None else center_x
    m = np.float32([[s, 0, W / 2 - s * cx + dx], [0, s, H / 2 - s * h / 2 + dy]])
    return cv2.warpAffine(image, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 40, 40))


def write(frames, path):
    mp4 = path.with_suffix('.avi')
    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*'MJPG'), FPS, (W, H))
    for f in frames:
        writer.write(f)
    writer.release()
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(mp4), '-pix_fmt', 'yuv420p', str(path)], check=True)
    mp4.unlink()


def jitter(i):
    return 1.0 + 0.01 * np.sin(i / 9), 3 * np.sin(i / 5), 2 * np.cos(i / 7)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--front', required=True)
    parser.add_argument('--side', required=True)
    parser.add_argument('--side-face-x', type=float, required=True, help='face centre x in the side photo')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    front = cv2.imread(args.front)
    write([place(front, None, *jitter(i)) for i in range(6 * FPS)], out / 'front.y4m')
    side = cv2.imread(args.side)
    mirrored = side[:, ::-1].copy()
    x = args.side_face_x
    frames = [place(side, x, *jitter(i)) for i in range(4 * FPS)]
    frames += [place(mirrored, side.shape[1] - 1 - x, *jitter(i)) for i in range(4 * FPS, 8 * FPS)]
    write(frames, out / 'sides.y4m')
    print('wrote', sorted(p.name for p in out.iterdir()))


if __name__ == '__main__':
    main()
