"""Reference outputs of the face refiner on real frames, for the TypeScript parity test.

    python tools/export_refine_parity.py --model dog_face_pose.onnx --video clip.mp4 --frames 266,865 --out DIR
    (a dynamic-shape export: add --input-size 320 --crop-size 224 to check refine.crop_size)

Writes DIR/refine_<frame>.bgr (raw BGR bytes) and DIR/refine_ref.json with, per frame, the
refined detections of still-image refinement (orientation search + residual re-crop), their
eye-contrast evidence and the open-loop verification.  The SDK test
(packages/sdk/test/real-models.test.ts, DOG_CAPTURE_REAL=DIR) must reproduce them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dog_capture.engine.config import RefineConfig  # noqa: E402
from dog_capture.engine.detectors import YoloPoseDetector  # noqa: E402
from dog_capture.engine.refine import FaceRefiner, eye_contrast  # noqa: E402
from dog_capture.engine.schema import _plain  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--video', required=True)
    parser.add_argument('--frames', required=True, help='comma list of frame indices')
    parser.add_argument('--out', required=True)
    parser.add_argument('--input-size', type=int, default=0, help='full-frame size of a dynamic-shape model (0 = its sidecar)')
    parser.add_argument('--crop-size', type=int, default=0, help='refine.crop_size (dynamic-shape model only)')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wanted = {int(x) for x in args.frames.split(',')}
    detector = YoloPoseDetector(args.model, score_threshold=0.35, input_size=args.input_size)
    refiner = FaceRefiner(detector, RefineConfig(crop_size=args.crop_size))
    if args.crop_size and refiner.size != args.crop_size:
        parser.error(f'{args.model} cannot run {args.crop_size} px crops (fixed-shape model)')
    capture = cv2.VideoCapture(args.video)
    frames, index = [], 0
    while wanted:
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            wanted.discard(index)
            found = [d for d in detector.detect(frame) if d.validate() is None]
            refined = [d for d in refiner.refine(frame, found, 0.0, still=True) if d.extra.get('refined')]
            faces = []
            for det in refined:
                item = det.to_dict()
                item['upright_angle'] = round(float(det.extra['upright_angle']), 3)
                item['eye_contrast'] = _plain(eye_contrast(frame, det))
                item['verify'] = _plain(refiner.verify(frame, det))
                faces.append(item)
            name = f'refine_{index}.bgr'
            (out / name).write_bytes(frame.tobytes())
            frames.append({'frame_index': index, 'file': name, 'w': frame.shape[1], 'h': frame.shape[0],
                           'full': [d.to_dict() for d in found], 'faces': faces})
        index += 1
    (out / 'refine_ref.json').write_text(json.dumps({'model_input': detector.size, 'crop_size': args.crop_size, 'frames': frames}, indent=1))
    print(f'{len(frames)} frames -> {out / "refine_ref.json"}')


if __name__ == '__main__':
    main()
