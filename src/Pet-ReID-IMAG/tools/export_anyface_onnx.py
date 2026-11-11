"""Export AnyFace / YOLOv5-face weights to the ONNX graph the v1 engine decodes.

    python tools/export_anyface_onnx.py --weights ../../models/pretrained/anyface/yolov5l6_best.pt --size 640

The graph uses the vendored YOLOv5-face ``export_cat`` head: one output
[1, N, 16] = cx, cy, w, h, objectness, 5 x (x, y), class score in input
pixels; decoding and NMS run in NumPy (``AnyFaceDetector``).  The export is
verified against PyTorch with ONNX Runtime and a sidecar JSON records the
source hash, size and licence.  ``--cfg models/yolov5n.yaml`` builds a
randomly initialised network instead (pipeline test only, detects nothing).

Only use checkpoints you trust: YOLOv5-face checkpoints are pickled models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
YOLO_ROOT = ROOT / 'third_party' / 'AnyFace' / 'yolov5-face'


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--weights', help='trusted YOLOv5-face/AnyFace .pt checkpoint')
    source.add_argument('--cfg', help='model yaml for a random-weight pipeline test')
    parser.add_argument('--size', type=int, default=640, help='square input size (multiple of the max stride)')
    parser.add_argument('--output', help='default: models/pretrained/dog_capture_v1/anyface_<name>_<size>.onnx')
    parser.add_argument('--opset', type=int, default=17)
    args = parser.parse_args()

    import numpy as np
    import onnxruntime as ort
    import torch
    sys.path.insert(0, str(YOLO_ROOT))
    from models.experimental import attempt_load
    from models.yolo import Model

    if args.weights:
        model = attempt_load(args.weights, map_location='cpu')
        name = Path(args.weights).stem.replace('_best', '')
    else:
        torch.manual_seed(0)
        model = Model(str(YOLO_ROOT / args.cfg) if not Path(args.cfg).is_file() else args.cfg, nc=1).float().fuse().eval()
        name = Path(args.cfg).stem + '_random'
    detect = model.model[-1]
    stride = int(max(model.stride))
    if args.size % stride:
        parser.error(f'--size must be a multiple of {stride} for this model')
    detect.export_cat = True
    if 'anchor_grid' in detect._buffers:
        del detect._buffers['anchor_grid']
    detect.anchor_grid = [torch.zeros(1)] * detect.nl  # rebuilt per level by _make_grid_new
    detect.grid = [torch.zeros(1)] * detect.nl
    model.eval()

    output = Path(args.output) if args.output else REPO / 'models' / 'pretrained' / 'dog_capture_v1' / f'anyface_{name}_{args.size}.onnx'
    output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, args.size, args.size)
    with torch.no_grad():
        model(dummy)
        export_kwargs = dict(opset_version=args.opset, input_names=['input'], output_names=['output'])
        try:
            torch.onnx.export(model, dummy, str(output), dynamo=False, **export_kwargs)
        except TypeError:  # older torch without the dynamo switch
            torch.onnx.export(model, dummy, str(output), **export_kwargs)
        sample = torch.rand(1, 3, args.size, args.size)
        reference = model(sample).numpy()
    session = ort.InferenceSession(str(output), providers=['CPUExecutionProvider'])
    produced = session.run(None, {'input': sample.numpy()})[0]
    if produced.shape != reference.shape or produced.shape[-1] != 16:
        raise SystemExit(f'Unexpected export output {produced.shape}; expected [1, N, 16]')
    error = float(np.abs(produced - reference).max())
    scale = float(np.abs(reference).max())
    if error > 1e-3 * max(scale, 1):
        raise SystemExit(f'ONNX Runtime output differs from PyTorch: max |diff| {error}')
    meta = {'kind': 'anyface', 'schema': 'anyface5', 'input_size': args.size, 'output_shape': list(produced.shape),
            'source': str(args.weights or args.cfg), 'source_sha256': sha256(args.weights) if args.weights else None,
            'random_weights': not args.weights, 'torch': torch.__version__, 'opset': args.opset,
            'max_abs_diff_vs_torch': error,
            'license': 'AnyFace code MIT / YOLOv5-face GPL-3.0; weights trained on research datasets (WIDER FACE, '
                       'AnimalWeb, ...). Check before commercial use.'}
    output.with_suffix('.json').write_text(json.dumps(meta, indent=2), encoding='utf8')
    print(json.dumps({'onnx': str(output), 'sha256': sha256(output), **meta}, indent=2))


if __name__ == '__main__':
    main()
