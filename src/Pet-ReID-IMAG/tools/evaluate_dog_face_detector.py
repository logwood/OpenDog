"""Evaluate any engine face detector on a dog-face6 labelled split.

    python tools/evaluate_dog_face_detector.py --data datasets/dog-face6 --model models/pretrained/dog_capture_v1/dog_face_pose.onnx
    python tools/evaluate_dog_face_detector.py --data datasets/dog-face6 --model models/pretrained/dog_face_landmarks_v1 --kind dogflw

Reports face-box recall/precision at IoU >= 0.5 (one-to-one matching), nose
error against the *ground-truth* nose normalised by the face-box size, eye
error against the labelled eyes normalised by their distance (in a
pseudo-labelled set these are teacher eyes -- the report says so), keypoint
visibility on matched faces and per-image latency.  Different detectors
define their box and nose differently (e.g. DogFLW nose-pad centroid vs nose
tip); compare like with like.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dog_capture.engine import CaptureConfig  # noqa: E402
from dog_capture.engine.schema import box_iou_matrix  # noqa: E402
from dog_capture.engine.tracking import assign  # noqa: E402
from dog_capture.engine.zoo import _guess_kind, build_detector  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', required=True)
    parser.add_argument('--split', default='val')
    parser.add_argument('--model', required=True)
    parser.add_argument('--kind', choices=['yolo_pose', 'anyface', 'scrfd', 'dogflw'])
    parser.add_argument('--input-size', type=int, default=640)
    parser.add_argument('--score', type=float, default=0.35)
    parser.add_argument('--providers', default='auto')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--every', type=int, default=1, help='use every n-th image')
    parser.add_argument('--output')
    args = parser.parse_args()
    model = Path(args.model)
    kind = args.kind or ('dogflw' if model.is_dir() else _guess_kind(model))
    config = CaptureConfig().merged({'detector': {'kind': kind, 'model': str(model), 'input_size': args.input_size,
                                                  'score_threshold': args.score},
                                     'runtime': {'providers': args.providers}})
    detector = build_detector(config, kind, model)
    if hasattr(detector, 'warmup'):
        detector.warmup()
    labels = sorted((Path(args.data) / 'labels' / args.split).glob('*.txt'))[::args.every]
    if args.limit:
        labels = labels[:args.limit]
    gt_total = pred_total = matched = nose_faces = nose_hits = 0
    nose_err, eye_err, eye_vis, times = [], [], [], []
    for label in labels:
        image = cv2.imread(str(Path(args.data) / 'images' / args.split / (label.stem + '.jpg')))
        if image is None:
            continue
        h, w = image.shape[:2]
        rows = [[float(v) for v in line.split()] for line in label.read_text().splitlines() if line.strip()]
        started = time.perf_counter()
        found = detector.detect(image)
        times.append((time.perf_counter() - started) * 1000)
        gt_total += len(rows)
        pred_total += len(found)
        gt_boxes = np.array([[(r[1] - r[3] / 2) * w, (r[2] - r[4] / 2) * h, (r[1] + r[3] / 2) * w,
                              (r[2] + r[4] / 2) * h] for r in rows]).reshape(-1, 4)
        # Box-convention-free hit: some predicted nose within 0.2 face sizes of the true nose.
        noses = [d.keypoints[d.schema.names.index('nose')] for d in found]
        for r, box in zip(rows, gt_boxes):
            if r[5 + 3 * 2 + 2] > 0:
                nose_faces += 1
                size = float(np.sqrt((box[2] - box[0]) * (box[3] - box[1])))
                target = np.array([r[5 + 6] * w, r[5 + 7] * h])
                nose_hits += any(np.linalg.norm(n - target) < 0.2 * size for n in noses)
        if not rows or not found:
            continue
        iou = box_iou_matrix(gt_boxes, np.stack([d.box for d in found]))
        pairs, _, _ = assign(1 - iou, 0.5)
        for gi, pi in pairs:
            matched += 1
            row, det = rows[gi], found[pi]
            kps = np.array(row[5:5 + 18]).reshape(6, 3) * [w, h, 1]
            names = det.schema.names
            size = float(np.sqrt((gt_boxes[gi, 2] - gt_boxes[gi, 0]) * (gt_boxes[gi, 3] - gt_boxes[gi, 1])))
            if kps[2, 2] > 0:
                nose_err.append(float(np.linalg.norm(det.keypoints[names.index('nose')] - kps[2, :2])) / size)
            if kps[0, 2] > 0 and kps[1, 2] > 0:
                iod = float(np.linalg.norm(kps[0, :2] - kps[1, :2]))
                pred = np.stack([det.keypoints[names.index('left_eye')], det.keypoints[names.index('right_eye')]])
                # eye order conventions differ between detectors: take the better assignment
                direct = np.linalg.norm(pred - kps[:2, :2], axis=1).mean()
                crossed = np.linalg.norm(pred[::-1] - kps[:2, :2], axis=1).mean()
                eye_err.append(float(min(direct, crossed)) / max(iod, 1.0))
                if det.kp_scores is not None:
                    eye_vis.append(float(min(det.point_score('left_eye'), det.point_score('right_eye'))))

    def q(values):
        return {'mean': float(np.mean(values)), 'p50': float(np.median(values)),
                'p90': float(np.percentile(values, 90)), 'n': len(values)} if values else None

    report = {'model': str(model), 'kind': kind, 'split': args.split, 'images': len(times),
              'face_recall_iou50': matched / max(gt_total, 1), 'face_precision_iou50': matched / max(pred_total, 1),
              'gt_faces': gt_total, 'predicted_faces': pred_total,
              'nose_hit_recall': nose_hits / max(nose_faces, 1),
              'nose_error_over_face_size': q(nose_err),
              'eye_error_over_eye_distance': q(eye_err),
              'eye_reference': 'labelled eyes of this split (teacher pseudo-labels if pseudolabel was used)',
              'min_eye_visibility_on_labelled_eyes': q(eye_vis),
              'latency_ms': q(times), 'providers': getattr(detector, 'runtime', {}).get('used')}
    print(json.dumps(report, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
