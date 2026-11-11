"""Matched A/B output-transfer benchmark; never counts cached upstream as free inference."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dog_capture.cli import ROOT
from dog_capture.landmarks import confidence_model, uncertainty
from dog_capture.models import LiteModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='gpu')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    sample_names = ('user', 'wolfhound_panting', 'labrador_barking', 'lurcher_lick')
    sample_root = ROOT / 'assets/dog_capture/benchmark_inputs'
    paths = ([sample_root / f'{name}.png' for name in sample_names] if sample_root.is_dir() else
             [ROOT / 'artifacts/runs/dog_capture/release_validation' / name / 'best_front_face.png' for name in sample_names])
    images = []
    for path in paths:
        im = cv2.imread(str(path))
        if im is None:
            raise FileNotFoundError(path)
        images.append(cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (384, 384))[None].astype(np.float32) / 255)
    images += [np.zeros_like(images[0]), np.full_like(images[0], .5)]
    if args.device == 'gpu':
        from dog_capture.gpu import GpuModel
        cls = GpuModel
    else:
        cls = LiteModel
    base = ROOT / 'models/pretrained/dog_face_landmarks_v1/dog_face_landmarks_full.tflite'
    models = {key: cls(content=confidence_model(base, compact=(key == 'compact')))
              for key in ('full', 'compact')}
    measurements, parity = defaultdict(list), []
    # Warm both models on all images, then alternate order to reduce clock bias.
    for model in models.values():
        for image in images:
            model(image)
    for repeat in range(5):
        for i, image in enumerate(images):
            results = {}
            for key in (('full', 'compact') if repeat % 2 == 0 else ('compact', 'full')):
                start = time.perf_counter()
                raw = models[key](image)
                conf = uncertainty(raw[1])
                measurements[key].append((time.perf_counter() - start) * 1000)
                results[key] = raw[0], conf
                assert np.isfinite(raw[0]).all() and all(np.isfinite(v) for v in conf.values())
            delta = float(np.max(np.abs(results['full'][0] - results['compact'][0])))
            entropy_delta = max(abs(results['full'][1][k] - results['compact'][1][k]) for k in results['full'][1])
            assert delta < 1e-5 and entropy_delta < .04, (i, delta, entropy_delta)
            if repeat == 0:
                parity.append({'sample': i, 'coordinates_max_abs_delta': delta, 'entropy_max_abs_delta': entropy_delta})
    report = {'device': args.device, 'samples': [str(p) for p in paths] + ['zero', 'flat gray'],
              'parity': parity, 'transfer_bytes': {'full': 6783344, 'compact': 552},
              'microbenchmark': {key: {'n': len(v), 'p50_ms': float(np.median(v)), 'p95_ms': float(np.percentile(v, 95))}
                                 for key, v in measurements.items()},
              'notes': ['Includes forward, output readback and uncertainty calculation; excludes resize and model startup.',
                        'Compact graph clamps probabilities at 1e-6 inside log for fp16 safety. Entropy bias is small, measured here.',
                        'Backbone and pretrained weights unchanged. This is not a 12,000x inference speedup.']}
    (output / 'microbenchmark.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    # Release compiled models before whole-video runs so they do not compete for memory.
    models.clear()
    rows = []
    for run_name, full in [('full_A1', True), ('compact_B1', False), ('compact_B2', False), ('full_A2', True)]:
        run_dir = output / run_name
        command = [sys.executable, str(ROOT / 'src/Pet-ReID-IMAG/tools/select_dog_views.py'),
                   str(ROOT / 'data/dog_capture/user/2611101816341481.mp4'), '--device', args.device,
                   '--no-preview', '--output', str(run_dir)]
        if full:
            command.append('--full-heatmaps')
        with (output / f'{run_name}.log').open('w', encoding='utf8') as log:
            subprocess.run(command, stdout=log, stderr=log, check=True)
        result = json.loads((run_dir / 'result.json').read_text(encoding='utf8'))
        rows.append({'run': run_name, 'compact': not full, 'performance': result['performance'],
                     'selected': {k: v['frame_index'] for k, v in result['selected'].items()}})
        print(json.dumps(rows[-1]), flush=True)
    report['whole_video_ABBA'] = rows
    (output / 'benchmark.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
