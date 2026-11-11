"""Build engines once, outside real-time capture. Separate conversion environment."""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dog_capture.cli import ROOT
from dog_capture.trt_export import build_engines, export_onnx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', default=str(ROOT / 'models/pretrained/dog_face_landmarks_v1'))
    parser.add_argument('--output', default=str(ROOT / 'models/pretrained/dog_capture_trt_v3'))
    parser.add_argument('--export-python', help='Python with tensorflow + tf2onnx; never needed at inference time')
    parser.add_argument('--download-models', action='store_true')
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument('--export-only', action='store_true')
    stages.add_argument('--build-only', action='store_true')
    stages.add_argument('--check', action='store_true', help='Load all engines and run synchronized startup checks')
    parser.add_argument('--precision', choices=['fp32', 'fp16'], default='fp16')
    parser.add_argument('--workspace-mb', type=int, default=512)
    parser.add_argument('--optimization-level', type=int, default=3)
    parser.add_argument('--engine-names', nargs='+', choices=['localizer', 'landmarks', 'detector'],
                        default=['localizer', 'landmarks', 'detector'])
    args = parser.parse_args()
    if args.check:
        from dog_capture.trt import TrtModel
        for name in ('localizer', 'landmarks', 'detector'):
            with TrtModel(Path(args.output) / (name + '.engine')) as model:
                values = model(np.zeros(model.input['shape'], model.input['dtype']))
                if not all(np.isfinite(value).all() for value in values):
                    raise RuntimeError(f'Nonfinite startup output: {name}')
                print(name, [v.shape for v in values], 'CUDA Graph:', model.cuda_graph_enabled)
        return
    if args.workspace_mb < 1 or not 0 <= args.optimization_level <= 5:
        parser.error('Invalid builder workspace/optimization level')
    if not args.build_only:
        if not args.export_python:
            parser.error('--export-python is required for ONNX export')
        if args.download_models:
            from dog_capture.models import ensure_models
            ensure_models(args.models, download=True)
        export_onnx(args.models, args.output, args.export_python)
    if not args.export_only:
        build_engines(args.output, precision=args.precision, workspace_mb=args.workspace_mb,
                      optimization_level=args.optimization_level, engine_names=args.engine_names)


if __name__ == '__main__':
    main()
