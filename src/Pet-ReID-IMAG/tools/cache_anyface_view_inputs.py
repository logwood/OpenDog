"""Integration exercise using the repository's existing AnyFace, not a new model.

Run with the project's torch environment. Input must be a known single-dog
test video. This helper is deliberately separate from the lean selector and
reports the upstream cost separately. No claim that cached detections are free.
"""
import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dog_capture.cli import ROOT
from dog_capture.models import box_iou, file_hash
from dog_capture.observations import FaceObservation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video')
    parser.add_argument('--output', required=True)
    parser.add_argument('--known-single-dog', required=True, action='store_true')
    parser.add_argument('--fps', type=float, default=15)
    parser.add_argument('--max-seconds', type=float, default=0)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error('--fps must be positive')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Avoid importing pet_id.__init__ and its unrelated full training stack.
    spec = importlib.util.spec_from_file_location('_dog_views_anyface', ROOT / 'src/Pet-ReID-IMAG/pet_id/localization.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    model = module.AnyFaceDetector(ROOT / 'models/pretrained/anyface/yolov5l6_best.pt', device=args.device)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise ValueError('Cannot open input video')
    fps = cap.get(cv2.CAP_PROP_FPS)
    index, next_time, track = 0, 0., 0
    previous_box, last_seen = None, -float('inf')
    durations = []
    singles = 0
    started = time.perf_counter()
    with (output / 'observations.jsonl').open('w', encoding='utf8') as stream:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = index / fps
            if args.max_seconds and timestamp >= args.max_seconds:
                break
            if timestamp + 1e-6 >= next_time:
                begin = time.perf_counter()
                detections = model.detect(frame)
                durations.append((time.perf_counter() - begin) * 1000)
                obs = None
                if len(detections) == 1:
                    d = detections[0]
                    if timestamp - last_seen > .75 or (previous_box is not None and box_iou(previous_box, d.bbox_xyxy) < .15):
                        track += 1
                    obs = FaceObservation.from_anyface(d, frame_index=index, time_s=timestamp, track_id=f'anyface-{track}')
                    previous_box, last_seen = d.bbox_xyxy, timestamp
                    singles += 1
                elif len(detections) > 1:
                    track += 1
                    previous_box, last_seen = None, -float('inf')
                stream.write(json.dumps({'frame_index': index, 'observation': asdict(obs) if obs else None}, allow_nan=False) + '\n')
                next_time = timestamp + 1 / args.fps
            index += 1
    cap.release()
    report = {'video': str(Path(args.video).resolve()), 'input_sha256': file_hash(args.video),
              'sample_fps_requested': args.fps, 'frames_decoded': index, 'detections_computed': len(durations),
              'single_face_observations': singles, 'upstream_wall_seconds': time.perf_counter() - started,
              'upstream_compute_ms_p50': float(np.median(durations)), 'upstream_compute_ms_p95': float(np.percentile(durations, 95)),
              'model': 'existing AnyFace YOLOv5l6; image_size=800', 'weights_sha256': file_hash(model.weights),
              'notes': ['Known single-dog media only; animal class is supplied by test-video provenance, not inferred from AnyFace class_id.',
                        'Track IDs are provisional IoU continuity for this test helper, not identity recognition. Long losses create new tracks.',
                        'This cost is upstream and must be included if AnyFace is not already running.']}
    (output / 'upstream_cost.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
