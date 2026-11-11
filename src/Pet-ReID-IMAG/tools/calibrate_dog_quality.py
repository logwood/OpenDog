"""Measure quality metrics on real aligned dog faces vs. synthetic degradations.

Input is a YOLO-pose dataset in the dog-face6 layout (``train_dog_face_pose.py
prepare`` + ``pseudolabel``): faces whose eyes and nose are labelled are
aligned with the engine's PetFace template, then measured as-is and after
Gaussian defocus and linear motion blur.  The printout shows how well each
metric separates sharp from degraded faces and the score the current
constants give, so the defaults in ``dog_capture/engine/quality.py`` (or a
config) can be re-derived on your own camera data.

    python tools/calibrate_dog_quality.py --data datasets/dog-face6 --split val --min-iod 40
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dog_capture.engine import CaptureConfig, Detection  # noqa: E402
from dog_capture.engine.alignment import load_template  # noqa: E402
from dog_capture.engine.pose import face_geometry  # noqa: E402
from dog_capture.engine.quality import QualityAssessor  # noqa: E402
from dog_capture.engine.schema import DOGFACE6  # noqa: E402


def motion_kernel(length, angle):
    kernel = np.zeros((length, length), np.float32)
    kernel[length // 2, :] = 1
    rot = cv2.getRotationMatrix2D((length / 2 - .5, length / 2 - .5), angle, 1)
    kernel = cv2.warpAffine(kernel, rot, (length, length))
    return kernel / kernel.sum()


DEGRADATIONS = {
    'original': lambda img, iod: img,
    'defocus_0.03iod': lambda img, iod: cv2.GaussianBlur(img, (0, 0), max(.5, .03 * iod)),
    'defocus_0.06iod': lambda img, iod: cv2.GaussianBlur(img, (0, 0), .06 * iod),
    'defocus_0.12iod': lambda img, iod: cv2.GaussianBlur(img, (0, 0), .12 * iod),
    'motion_0.15iod': lambda img, iod: cv2.filter2D(img, -1, motion_kernel(max(3, int(.15 * iod)) | 1, 30)),
    'motion_0.3iod': lambda img, iod: cv2.filter2D(img, -1, motion_kernel(max(3, int(.3 * iod)) | 1, 30)),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', required=True)
    parser.add_argument('--split', default='val')
    parser.add_argument('--min-iod', type=float, default=40)
    parser.add_argument('--limit', type=int, default=400)
    parser.add_argument('--output', help='optional JSON with all measurements')
    args = parser.parse_args()
    config = CaptureConfig()
    assessor = QualityAssessor(config.quality, load_template('petface_dog'))
    front = config.pose.views[0]
    rows = {name: [] for name in DEGRADATIONS}
    used = 0
    for label in sorted((Path(args.data) / 'labels' / args.split).glob('*.txt')):
        if used >= args.limit:
            break
        image = cv2.imread(str(Path(args.data) / 'images' / args.split / (label.stem + '.jpg')))
        if image is None:
            continue
        h, w = image.shape[:2]
        for line in label.read_text().splitlines():
            v = [float(x) for x in line.split()]
            kps = np.array(v[5:]).reshape(6, 3)
            if not (kps[0, 2] and kps[1, 2] and kps[2, 2]):
                continue
            points = kps[:, :2] * [w, h]
            cx, cy, bw, bh = v[1] * w, v[2] * h, v[3] * w, v[4] * h
            det = Detection([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], .9, points, DOGFACE6,
                            np.where(kps[:, 2] > 0, 1.0, 0.0))
            geometry = face_geometry(det, config.pose)
            if not geometry.valid or geometry.iod < args.min_iod or abs(geometry.yaw) > .3:
                continue
            for name, degrade in DEGRADATIONS.items():
                report, _ = assessor.assess(degrade(image, geometry.iod), det, geometry, front)
                m = report.metrics
                if 'laplacian' not in m:
                    continue
                rows[name].append({'laplacian': m['laplacian'], 'blur_effect': m['blur_effect'],
                                   'eye_laplacian': min(m['eye_laplacian']), 'sharpness': report.factors['sharpness'],
                                   'eye_detail': report.factors['eye_detail'], 'score': report.score,
                                   'blurred': 'blurred' in report.reasons, 'iod': geometry.iod})
            used += 1
    print(f'{used} frontal-ish faces with labelled eyes and nose, iod >= {args.min_iod}px')
    keys = ('laplacian', 'blur_effect', 'eye_laplacian', 'sharpness', 'score')
    print(f'{"condition":18}' + ''.join(f'{k:>22}' for k in keys) + f'{"rejected":>10}')
    summary = {}
    for name, items in rows.items():
        if not items:
            continue
        stats = {k: np.percentile([r[k] for r in items], [10, 50, 90]).round(3).tolist() for k in keys}
        stats['rejected'] = float(np.mean([r['blurred'] for r in items]))
        summary[name] = stats
        print(f'{name:18}' + ''.join(f'{"/".join(f"{x:.3g}" for x in stats[k]):>22}' for k in keys)
              + f'{stats["rejected"]:>10.2f}')
    base = rows['original']
    for name, items in rows.items():
        if name == 'original' or not items:
            continue
        wins = np.mean([a['score'] > b['score'] for a, b in zip(base, items)])
        print(f'original scores higher than {name}: {wins:.3f}')
    if args.output:
        Path(args.output).write_text(json.dumps({'summary': summary, 'rows': rows}, indent=1))


if __name__ == '__main__':
    main()
