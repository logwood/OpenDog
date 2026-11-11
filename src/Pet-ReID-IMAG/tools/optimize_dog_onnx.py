"""FP16 / INT8 variants of a capture detector, checked against FP32.

    python tools/optimize_dog_onnx.py --model models/pretrained/dog_capture_v1/dog_face_pose.onnx \
        --fp16 --int8 --calib path/to/dog/images

FP16 suits GPUs and NPUs (CUDA/TensorRT/DirectML/CoreML); INT8 static QDQ
quantisation (ONNX Runtime, per-channel weights, calibration on your images)
suits CPUs and mobile accelerators (XNNPACK, NNAPI, QNN).  Each variant is run
through the engine's own decoder on held-out images and compared with FP32:
detection recall/precision at IoU 0.5, keypoint error in eye distances and
latency.  Quantisation is never assumed lossless -- read the report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dog_capture.engine import CaptureConfig  # noqa: E402
from dog_capture.engine.io import IMAGE_SUFFIXES, imread  # noqa: E402
from dog_capture.engine.schema import box_iou_matrix  # noqa: E402
from dog_capture.engine.zoo import _guess_kind, build_body_gate, build_detector  # noqa: E402


def make(kind, path, providers='cpu'):
    config = CaptureConfig().merged({'detector': {'kind': kind if kind != 'yolox' else 'auto', 'model': str(path)},
                                     'body_gate': {'model': str(path), 'enabled': 'on'},
                                     'runtime': {'providers': providers}})
    if kind == 'yolox':
        return build_body_gate(config, 'external')
    return build_detector(config, kind, Path(path))


def outputs(model, image):
    found = model.detect(image)
    if isinstance(found, tuple):  # body gate: (boxes, scores)
        return [(b, None) for b in found[0]]
    return [(d.box, d.keypoints) for d in found]


def compare(reference, candidate, images):
    matched = total_ref = total_new = 0
    errors, times = [], []
    for image in images:
        ref = outputs(reference, image)
        started = time.perf_counter()
        new = outputs(candidate, image)
        times.append((time.perf_counter() - started) * 1000)
        total_ref += len(ref)
        total_new += len(new)
        if not ref or not new:
            continue
        iou = box_iou_matrix(np.stack([b for b, _ in ref]), np.stack([b for b, _ in new]))
        for i, j in enumerate(iou.argmax(1)):
            if iou[i, j] >= 0.5:
                matched += 1
                kr, kn = ref[i][1], new[j][1]
                if kr is not None and kn is not None:
                    iod = max(float(np.linalg.norm(kr[0] - kr[1])), 1.0)
                    errors.append(float(np.linalg.norm(kr - kn, axis=1).mean()) / iod)
    return {'recall_vs_fp32': matched / max(total_ref, 1), 'precision_vs_fp32': matched / max(total_new, 1),
            'keypoint_error_iod_mean': float(np.mean(errors)) if errors else None,
            'keypoint_error_iod_p95': float(np.percentile(errors, 95)) if errors else None,
            'latency_ms_p50': float(np.median(times)), 'detections_fp32': total_ref, 'detections_variant': total_new}


class Reader:
    """ONNX Runtime calibration reader using the detector's own preprocessing."""

    def __init__(self, detector, images, input_name):
        self.items = iter([{input_name: detector.preprocess(img)[0].astype(np.float32)} for img in images])

    def get_next(self):
        return next(self.items, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True)
    parser.add_argument('--kind', choices=['yolo_pose', 'anyface', 'scrfd', 'yolox'])
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--int8', action='store_true')
    parser.add_argument('--calib', required=True, help='folder of representative images (frames from your cameras)')
    parser.add_argument('--max-images', type=int, default=200)
    parser.add_argument('--calibration', choices=['percentile', 'minmax', 'entropy'], default='percentile',
                        help='activation range estimator; percentile clips outliers that MinMax keeps')
    args = parser.parse_args()
    model = Path(args.model)
    kind = args.kind or ('yolox' if 'yolox' in model.name else _guess_kind(model))
    files = sorted(p for p in Path(args.calib).rglob('*') if p.suffix.lower() in IMAGE_SUFFIXES)[:args.max_images]
    if len(files) < 10:
        raise SystemExit('Need at least 10 calibration images')
    images = [imread(p) for p in files]
    calib, held_out = images[::2], images[1::2]  # never evaluate on calibration images
    reference = make(kind, model)
    report = {'model': str(model), 'kind': kind, 'calibration_images': len(calib), 'evaluation_images': len(held_out),
              'fp32_latency_ms_p50': compare(reference, reference, held_out[:20])['latency_ms_p50']}
    sidecar = model.with_suffix('.json')
    if args.fp16:
        import onnx
        from onnxruntime.transformers.float16 import convert_float_to_float16
        target = model.with_name(model.stem + '_fp16.onnx')
        onnx.save(convert_float_to_float16(onnx.load(str(model)), keep_io_types=True), str(target))
        if sidecar.is_file():
            shutil.copy2(sidecar, target.with_suffix('.json'))
        report['fp16'] = {'file': str(target), **compare(reference, make(kind, target), held_out)}
    if args.int8:
        from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
        from onnxruntime.quantization.shape_inference import quant_pre_process
        import onnx
        from onnx import version_converter
        prepared = model.with_name(model.stem + '_prep.onnx')
        target = model.with_name(model.stem + '_int8.onnx')
        graph = onnx.load(str(model))
        opset = next(o.version for o in graph.opset_import if o.domain in ('', 'ai.onnx'))
        if opset < 13:  # per-channel QDQ needs opset >= 13 (e.g. YOLOX release graphs are opset 11)
            graph = version_converter.convert_version(graph, 13)
        onnx.save(graph, str(prepared))
        quant_pre_process(str(prepared), str(prepared))
        input_name = reference.input.name
        quantize_static(str(prepared), str(target), Reader(reference, calib, input_name),
                        quant_format=QuantFormat.QDQ, per_channel=True, activation_type=QuantType.QUInt8,
                        weight_type=QuantType.QInt8,
                        calibrate_method={'minmax': CalibrationMethod.MinMax, 'entropy': CalibrationMethod.Entropy,
                                          'percentile': CalibrationMethod.Percentile}[args.calibration],
                        # Quantise convolutions only: detection heads concatenate box coordinates
                        # (0..input size) and scores (0..1) into one tensor, and a shared INT8 scale
                        # for that tensor erases the scores (measured: 0 detections).
                        op_types_to_quantize=['Conv'],
                        extra_options={'CalibPercentile': 99.99})
        prepared.unlink(missing_ok=True)
        if sidecar.is_file():
            shutil.copy2(sidecar, target.with_suffix('.json'))
        report['int8'] = {'file': str(target), 'calibration': args.calibration,
                          **compare(reference, make(kind, target), held_out)}
    for key in ('fp16', 'int8'):
        if key in report:
            report[key]['bytes'] = Path(report[key]['file']).stat().st_size
    report['fp32_bytes'] = model.stat().st_size
    out = model.with_name(model.stem + '_optimize_report.json')
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
