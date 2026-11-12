"""Check capture decisions against an earlier engine/refiner on every video frame.

Both implementations use the current, unchanged model adapters with separate ORT
sessions. The reference uses the supplied config; the current implementation adds
--optimized-config. This tests numerical and decision parity, not throughput.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import cv2
import numpy as np

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from dog_capture.engine.config import CaptureConfig
from dog_capture.engine.io import RunWriter, VideoSource
from dog_capture.engine.zoo import build_engine


def digest(value):
    return hashlib.sha256(value).hexdigest()


def load_reference(root):
    modules, hashes = {}, {}
    for name in ('refine', 'engine'):
        path = root / 'src/Pet-ReID-IMAG/dog_capture/engine' / f'{name}.py'
        hashes[name] = digest(path.read_bytes())
        key = f'dog_capture.engine._comparison_reference_{name}'
        spec = importlib.util.spec_from_file_location(key, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
        modules[name] = module
    modules['engine'].FaceRefiner = modules['refine'].FaceRefiner
    return modules['engine'].CaptureEngine, hashes


def trace_models(engine, trace, counts):
    for label, model in (('face', engine.detector), ('body', engine.body_gate),
                         ('landmarks', engine.refiner.landmarker)):
        if model is None:
            continue
        def run(tensor, original=model._run, label=label):
            result = original(tensor)
            key = (label, tensor.shape, tensor.dtype.str, digest(tensor.tobytes()))
            trace[key] = [(v.shape, v.dtype.str, digest(v.tobytes())) for v in result]
            counts[label] += 1
            return result
        model._run = run


def normalize_diagnostics(reference, current, refine):
    """Mask only an uncomputed mirror value, after proving an existing gate failed."""
    skipped = 0
    if len(reference['faces']) != len(current['faces']):
        return skipped
    for old, new in zip(reference['faces'], current['faces']):
        check = new['detection'].get('verify', {})
        if 'mirror_skipped' not in check:
            continue
        original = old['detection'].get('verify', {})
        # Both front and side use at least one of these thresholds. At least
        # one original rejection must exist; full per-frame decision equality
        # below additionally catches using a wrong front/side threshold.
        known_failure = (original.get('score', 1) < refine.min_shot_score
                         or min(original.get('eyes', (1, 1))) < refine.min_front_eye_conf
                         or original.get('agreement') is None
                         or original['agreement'] > refine.max_agreement)
        if not known_failure:
            raise AssertionError('Mirror was skipped without an existing rejection')
        original['mirror'] = check['mirror'] = None
        check.pop('mirror_skipped')
        skipped += 1
    return skipped


def numeric_faces(result):
    values = []
    for face in result.faces:
        det, geometry = face.detection, face.geometry
        values.extend(np.asarray(det.box).tolist())
        values.extend(np.asarray(det.keypoints).ravel().tolist())
        if det.kp_scores is not None:
            values.extend(np.asarray(det.kp_scores).tolist())
        values.extend([det.score, geometry.yaw, geometry.roll_deg, face.quality.score])
    return digest(np.asarray(values, np.float64).tobytes())


def selected(bundle):
    if bundle is None:
        return {}
    return {view: [{**shot.summary(), 'aligned_sha256': digest(shot.aligned.tobytes()),
                    'frame_sha256': None if shot.frame is None else digest(shot.frame.tobytes())}
                   for shot in shots] for view, shots in bundle.slots.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--optimized-config', type=Path, required=True)
    parser.add_argument('--source', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output directory')
    args.output.mkdir(parents=True)
    cv2.setNumThreads(1)
    config = CaptureConfig.load(args.config).merged({'runtime': {'providers': 'cpu'}})
    optimized = config.merged(json.loads(args.optimized_config.read_text(encoding='utf8')))
    reference_class, hashes = load_reference(args.reference_root)
    report = {'method': __doc__, 'reference_source_sha256': hashes,
              'config': config.to_dict(), 'optimized_config': optimized.to_dict(), 'videos': []}
    for path in args.source:
        old_models = build_engine(config)
        original = reference_class(config, old_models.detector, old_models.body_gate, old_models.refiner.landmarker)
        current = build_engine(optimized)
        engines = {'reference': original, 'optimized': current}
        for engine in engines.values():
            for model in (engine.detector, engine.body_gate, engine.refiner.landmarker):
                if model is not None:
                    if model.runtime['used'] != ['CPUExecutionProvider'] or model.runtime['fallback_errors']:
                        raise RuntimeError('This equivalence check requires explicit CPU sessions without fallback')
                    model.warmup(15)
        traces = {name: {} for name in engines}
        counts = {name: Counter() for name in engines}
        for name, engine in engines.items():
            trace_models(engine, traces[name], counts[name])
        mismatches, n, skipped = [], 0, 0
        source = VideoSource(path)
        declared = int(source.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        try:
            for frame, timestamp, index in source:
                results = {}
                for name in (('reference', 'optimized') if index % 2 == 0 else ('optimized', 'reference')):
                    traces[name].clear()
                    results[name] = engines[name].process(frame, timestamp, index)
                old, new = (results[name].to_dict() for name in ('reference', 'optimized'))
                old.pop('timings_ms'); new.pop('timings_ms')
                skipped += normalize_diagnostics(old, new, config.refine)
                bad_outputs = [key[0] for key, val in traces['optimized'].items()
                               if traces['reference'].get(key) != val]
                if old != new or numeric_faces(results['reference']) != numeric_faces(results['optimized']) or bad_outputs:
                    mismatches.append({'frame': index, 'frame_result_differs': old != new,
                                       'raw_output_models_differ': bad_outputs})
                n += 1
                if n % 400 == 0:
                    print(path.name, n, 'frames; mismatches:', len(mismatches), flush=True)
        finally:
            source.close()
        if declared - n not in (0, 1):
            raise AssertionError(f'Incomplete decode: {n}/{declared}')
        bundles = {name: engine.finalize() for name, engine in engines.items()}
        selection = {name: selected(bundle) for name, bundle in bundles.items()}
        row = {'video': str(path), 'sha256': digest(path.read_bytes()), 'frames': n,
               'model_calls': {name: dict(value) for name, value in counts.items()},
               'mirror_diagnostics_skipped': skipped, 'mismatches': mismatches,
               'selection_identical': selection['reference'] == selection['optimized'], 'selection': selection}
        for name, bundle in bundles.items():
            directory = args.output / path.stem / name
            directory.mkdir(parents=True)
            writer = RunWriter(directory, log_frames=False, image_format='png')
            writer.save_bundle(bundle)
            writer.close()
        report['videos'].append(row)
        (args.output / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf8')
        print(json.dumps({k: v for k, v in row.items() if k != 'selection'}), flush=True)
        del engines, original, current, old_models, engine, model
        gc.collect()  # release traced session cycles before the next video
    if any(r['mismatches'] or not r['selection_identical'] for r in report['videos']):
        raise SystemExit('Capture equivalence failed; inspect result.json')
    print('All frames and selected pixels match:', args.output / 'result.json', flush=True)


if __name__ == '__main__':
    main()
