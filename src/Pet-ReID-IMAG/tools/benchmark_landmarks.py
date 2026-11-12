"""Head to head on dogface6 val: today's crop refinement vs the crop landmark model.

    python tools/benchmark_landmarks.py --data <dogface6> --detector dog_face_pose.onnx --landmarks dog_face_landmarks.onnx

Each sampled validation face is rolled by a random angle (the whole image, so the face sits
in its scene like a dog lying down), then given a tracker-like box: the rolled face's
axis-aligned box with +-10 % shift and +-15 % scale.  From that box:

  old x1        the YOLO-pose detector once, on the canonical crop at 0 deg (FaceRefiner.examine)
  old search    FaceRefiner still mode: four quarter turns + a re-crop at the residual (5-6 runs)
  new x1        the landmark model once, on the canonical 128 px crop at 0 deg
  new x2        ... then once more on the crop turned upright (its roll head beyond its
                keypoint range, else its eye line): FaceRefiner with the landmarker, still mode
  hybrid        new x2's upright crop, then the YOLO-pose detector once on it (the precise
                pass the engine runs on shot candidates)

Error = mean keypoint distance / ground-truth eye distance; good < 0.25 (as in
tools/benchmark_refine.py).  Rolls are reported in bands: |roll| < 45, 45-135, > 135 deg.
"""
from __future__ import annotations

import argparse
import glob
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dog_capture.engine.detectors import CropLandmarker, YoloPoseDetector  # noqa: E402
from dog_capture.engine.refine import FaceRefiner  # noqa: E402
from dog_capture.engine.schema import DOGFACE6, Detection  # noqa: E402


def rotate_image(image, angle):
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    c, s = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * s + w * c), int(h * c + w * s)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(image, m, (nw, nh), borderValue=(114, 114, 114)), m


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--detector', required=True)
    p.add_argument('--landmarks', required=True)
    p.add_argument('--faces', type=int, default=400)
    args = p.parse_args()
    detector = YoloPoseDetector(args.detector, score_threshold=0.15, threads=4)
    refiner = FaceRefiner(detector)
    landmarker = CropLandmarker(args.landmarks, score_threshold=0.05, threads=4)
    tracker = FaceRefiner(detector, landmarker=landmarker)
    rng = np.random.default_rng(3)
    files = sorted(glob.glob(f'{args.data}/images/val/*.jpg'))
    rng.shuffle(files)
    names = ('old x1', 'old search', 'new x1', 'new x2', 'hybrid')
    rows, times = [], {n: [] for n in names}
    for f in files:
        if len(rows) >= args.faces:
            break
        lab = f.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
        try:
            lines = [list(map(float, l.split())) for l in open(lab) if l.strip()]
        except FileNotFoundError:
            continue
        if not lines:
            continue
        img = cv2.imread(f)
        h, w = img.shape[:2]
        L = max(lines, key=lambda l: l[3] * l[4])
        gk = np.array(L[5:]).reshape(6, 3)
        gk[:, 0] *= w
        gk[:, 1] *= h
        if not (gk[0, 2] > 0 and gk[1, 2] > 0 and gk[2, 2] > 0):
            continue
        iod = float(np.linalg.norm(gk[0, :2] - gk[1, :2]))
        if iod < 8:
            continue
        roll = float(rng.uniform(-180, 180))
        rimg, m = rotate_image(img, roll)
        vis = gk[:, 2] > 0
        pts = (m[:, :2] @ gk[:, :2].T).T + m[:, 2]
        cx, cy, bw, bh = L[1] * w, L[2] * h, L[3] * w, L[4] * h
        corners = np.array([[cx - bw / 2, cy - bh / 2], [cx + bw / 2, cy - bh / 2], [cx - bw / 2, cy + bh / 2], [cx + bw / 2, cy + bh / 2]])
        rc = (m[:, :2] @ corners.T).T + m[:, 2]
        x1, y1, x2, y2 = rc[:, 0].min(), rc[:, 1].min(), rc[:, 0].max(), rc[:, 1].max()
        long_side = max(x2 - x1, y2 - y1)
        jx, jy = rng.uniform(-.1, .1, 2) * long_side
        js = math.exp(rng.uniform(math.log(.85), math.log(1.15)))
        ccx, ccy, half = (x1 + x2) / 2 + jx, (y1 + y2) / 2 + jy, long_side * js / 2
        box = np.array([ccx - half, ccy - half, ccx + half, ccy + half])
        err = lambda q: 9.0 if q is None else float(np.mean(np.linalg.norm(np.asarray(q)[vis] - pts[vis], axis=1)) / iod)
        row = {'roll': roll}
        t = time.perf_counter()
        d = refiner.examine(rimg, box, 0.0)
        times['old x1'].append(time.perf_counter() - t)
        row['old x1'] = err(None if d is None else d.keypoints)
        t = time.perf_counter()
        source = Detection(box.astype(np.float32), 0.5, np.tile(box[:2], (6, 1)), DOGFACE6)
        r = [x for x in refiner.refine(rimg, [source], 0.0, still=True) if x.extra.get('refined')]
        times['old search'].append(time.perf_counter() - t)
        row['old search'] = err(r[0].keypoints if r else None)
        t = time.perf_counter()
        d = tracker.examine(rimg, box, 0.0)
        times['new x1'].append(time.perf_counter() - t)
        row['new x1'] = err(None if d is None else d.keypoints)
        row['presence'] = 0.0 if d is None else d.score
        t = time.perf_counter()
        r = [x for x in tracker.refine(rimg, [source], 0.0, still=True) if x.extra.get('refined')]
        times['new x2'].append(time.perf_counter() - t)
        row['new x2'] = err(r[0].keypoints if r else None)
        t = time.perf_counter()
        p = None
        if r:   # the precise pass on the tracker's upright crop
            p = refiner.examine(rimg, r[0].extra['region'], r[0].extra['upright_angle'])
        times['hybrid'].append(time.perf_counter() - t + times['new x2'][-1])
        row['hybrid'] = err(None if p is None else p.keypoints)
        rows.append(row)
    bands = (('|roll|<45', lambda r: abs(r) < 45), ('45-135', lambda r: 45 <= abs(r) <= 135), ('>135', lambda r: abs(r) > 135),
             ('all', lambda r: True))
    print(f'{len(rows)} faces; good = keypoint error < 0.25 eye distance')
    for name in names:
        cells = []
        for label, keep in bands:
            e = np.array([r[name] for r in rows if keep(r['roll'])])
            cells.append(f'{label}: {np.mean(e < 0.25):.0%} (n{len(e)})')
        print(f'{name:11s} ' + ' | '.join(cells) + f' | median err {np.median([r[name] for r in rows]):.3f}'
              f' | {1000 * np.median(times[name]):.1f} ms native')
    print('presence on faces: median', round(float(np.median([r['presence'] for r in rows])), 3),
          'share < 0.5:', round(float(np.mean([r['presence'] < 0.5 for r in rows])), 3))


if __name__ == '__main__':
    main()
