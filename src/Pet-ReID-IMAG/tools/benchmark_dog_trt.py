"""Serial, synchronized model benchmarks and saved-output parity checks.

Run CPU first; pass its directory to --reference for subsequent backends.
No other inference/build process should run while collecting timings.
"""
import argparse
import gc
import json
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dog_capture.cli import ROOT
from dog_capture.landmarks import confidence_model
from dog_capture.models import LiteModel, file_hash, letterbox


def samples():
    faces, frames, origins = [], [], []
    for name in ('user', 'wolfhound_panting', 'labrador_barking', 'lurcher_lick'):
        path = ROOT / 'assets/dog_capture/benchmark_inputs' / (name + '.png')
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        faces.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        origins.append({'path': str(path.relative_to(ROOT)), 'sha256': file_hash(path)})
    faces += [np.zeros((384, 384, 3), np.uint8), np.full((384, 384, 3), 128, np.uint8)]
    manifest = json.loads((ROOT / 'assets/dog_capture/sources.json').read_text(encoding='utf8'))
    for entry in manifest.values():
        path = ROOT / entry['path']
        cap = cv2.VideoCapture(str(path))
        try:
            for index in (0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * .2), int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * .6)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, image = cap.read()
                if not ok:
                    raise RuntimeError(f'Cannot decode {path}, frame {index}')
                frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
        origins.append({'path': entry['path'], 'sha256': file_hash(path)})
    return {'localizer': [letterbox(im, 224)[0][None].astype(np.float32) / 255 for im in faces],
            'landmarks': [cv2.resize(im, (384, 384))[None].astype(np.float32) / 255 for im in faces],
            'detector': [letterbox(im, 320)[0][None] for im in frames]}, origins


def make_model(name, backend, engines):
    base = ROOT / 'models/pretrained/dog_face_landmarks_v1'
    if backend.startswith('trt'):
        from dog_capture.trt import TrtModel, TrtDetector
        graph = backend == 'trt'
        return TrtDetector(engines, use_cuda_graph=graph) if name == 'detector' else TrtModel(engines / (name + '.engine'), use_cuda_graph=graph)
    cls = LiteModel
    if backend == 'gpu' and name != 'detector':
        from dog_capture.gpu import GpuModel
        cls = GpuModel
    if name == 'landmarks':
        return cls(content=confidence_model(base / 'dog_face_landmarks_full.tflite', compact=True))
    return cls(base / ('dog_face_localizer.tflite' if name == 'localizer' else 'efficientdet_lite0_nms.tflite'))


def dog_detections(outputs):
    boxes, classes, scores, count = outputs
    return [(box, float(score)) for box, cls, score in zip(boxes[0, :int(count[0])], classes[0], scores[0])
            if int(cls) == 17 and score >= .35]


def parity(name, expected, actual):
    if name != 'detector':
        return {'max_abs_delta': [float(np.max(np.abs(a - b))) for a, b in zip(expected, actual)]}
    ref, test = dog_detections(expected), dog_detections(actual)
    matches = []
    for box, score in ref:
        candidates = []
        for other, other_score in test:
            overlap = np.prod(np.maximum(np.minimum(box[2:], other[2:]) - np.maximum(box[:2], other[:2]), 0))
            union = np.prod(box[2:] - box[:2]) + np.prod(other[2:] - other[:2]) - overlap
            candidates.append((float(overlap / max(union, 1e-9)), abs(score - other_score)))
        matches.append(max(candidates, default=(0., None)))
    return {'reference_dogs': len(ref), 'actual_dogs': len(test), 'matched_iou_and_score_delta': matches}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backend', choices=['cpu', 'gpu', 'trt', 'trt-direct'], required=True)
    p.add_argument('--engines', type=Path, default=ROOT / 'models/pretrained/dog_capture_trt_v3')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference', type=Path)
    p.add_argument('--repeat', type=int, default=8)
    args = p.parse_args()
    if args.repeat < 1:
        p.error('--repeat must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    inputs, origins = samples()
    reference = np.load(args.reference / 'outputs.npz') if args.reference else None
    saved, records = {}, {}
    for name, images in inputs.items():
        model = make_model(name, args.backend, args.engines)
        try:
            for image in images:
                model(image)
            times, comparisons = [], []
            for repeat in range(args.repeat):
                for i, image in enumerate(images):
                    start = time.perf_counter()
                    result = model(image)
                    times.append((time.perf_counter() - start) * 1000)
                    if not all(np.isfinite(x).all() for x in result):
                        raise AssertionError(f'Nonfinite {name} output')
                    if repeat == 0:
                        for j, value in enumerate(result):
                            saved[f'{name}_{i}_{j}'] = value.copy()
                        if reference is not None:
                            expected = [reference[f'{name}_{i}_{j}'] for j in range(len(result))]
                            comparisons.append({'sample': i, **parity(name, expected, result)})
            records[name] = {'n': len(times), 'p50_ms': float(np.median(times)),
                             'p95_ms': float(np.percentile(times, 95)), 'max_ms': max(times),
                             'parity': comparisons, 'timings_ms': times}
            print(name, records[name]['p50_ms'], records[name]['p95_ms'], flush=True)
        finally:
            if hasattr(model, 'close'):
                model.close()
            del model
            gc.collect()
    if reference is not None:
        reference.close()
    np.savez_compressed(args.output / 'outputs.npz', **saved)
    report = {'backend': args.backend, 'models': records, 'samples': origins,
              'python': sys.version, 'platform': platform.platform(),
              'notes': ['Batch=1; synchronous host-visible output. Includes input copy and output transfer; excludes resize and startup.',
                        'GPU baseline uses LiteRT WebGPU face models and the original CPU detector.',
                        'Detector conversion removes activation quantization; compare dog presence/boxes and full-video gates, not bitwise equality.']}
    (args.output / 'microbenchmark.json').write_text(json.dumps(report, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
