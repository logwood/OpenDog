"""Check capture decisions against an earlier engine/refiner on every video frame.

The default keeps current adapters for compatibility with old engine revisions.
--isolated-reference loads the complete old package, including image sampling
and model adapters. Conditional empty outputs are accepted only when the original
scores prove no detection passes the same threshold. Scale-gated models must keep
every score and every eligible anchor bit-exact; only rejected heads may differ.
This checks numerical and decision parity, not throughput.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import importlib
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
from dog_capture.engine.process_policy import cpu_qos
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


def load_isolated_reference(root):
    """Load the complete old package, including adapters and image sampling."""
    directory = root.resolve() / 'src/Pet-ReID-IMAG/dog_capture'
    namespace = '_capture_reference_' + digest(str(directory).encode())[:12]
    spec = importlib.util.spec_from_file_location(namespace, directory / '__init__.py',
                                                submodule_search_locations=[str(directory)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[namespace] = module
    spec.loader.exec_module(module)
    factory = importlib.import_module(namespace + '.engine.zoo').build_engine
    config_class = importlib.import_module(namespace + '.engine.config').CaptureConfig
    hashes = {path.name: digest(path.read_bytes()) for path in (directory / 'engine').glob('*.py')}
    return lambda config: factory(config_class().merged(config.to_dict())), hashes


def trace_models(engine, trace, counts):
    for label, model in (('face', engine.detector), ('body', engine.body_gate),
                         ('landmarks', engine.refiner.landmarker)):
        if model is None:
            continue
        def run(tensor, original=model._run, label=label, model=model):
            result = original(tensor)
            key = (label, tensor.shape, tensor.dtype.str, digest(tensor.tobytes()))
            outputs = [(v.shape, v.dtype.str, digest(v.tobytes())) for v in result]
            width = 4 + getattr(model, 'num_classes', 0) + getattr(model, 'num_kpts', 0) * getattr(model, 'kpt_dims', 0)
            raw_pose = (label == 'face' and len(result) == 1 and result[0].ndim == 3
                        and result[0].shape[:2] == (1, width) and width > 4)
            empty = raw_pose and not np.any(result[0][:, 4:4 + model.num_classes] >= model.score_threshold)
            skipped = bool(raw_pose and getattr(model, 'conditional_score_input', None)
                           and result[0].shape[-1] == 0)
            signature = eligible_pose_signature(result[0], model.num_classes, model.score_threshold) if raw_pose else None
            trace[key] = {'outputs': outputs, 'empty': bool(empty), 'skipped_heads': skipped,
                          'eligible_outputs': signature, 'masked_scale_heads': bool(getattr(model, 'masked_scale_heads', False))}
            counts[label] += 1
            return result
        model._run = run


def eligible_pose_signature(output, classes, threshold):
    """Hash all scores, anchor positions and every value the decoder may keep."""
    scores = output[:, 4:4 + classes]
    mask = scores.max(axis=1)[0] >= threshold
    kept = output[:, :, mask]
    return (float(threshold), scores.shape, scores.dtype.str, digest(scores.tobytes()),
            digest(mask.tobytes()), kept.shape, kept.dtype.str, digest(kept.tobytes()))


def compare_outputs(reference, current, *, counts=None):
    """Only explicitly declared rejected-head outputs may differ."""
    bad, skipped = [], 0
    for key, value in current.items():
        old = reference.get(key)
        if old is not None and old['outputs'] == value['outputs']:
            continue
        if old is not None and old['empty'] and value['skipped_heads']:
            skipped += 1
        elif (old is not None and value.get('masked_scale_heads')
              and old.get('eligible_outputs') is not None
              and old['eligible_outputs'] == value.get('eligible_outputs')):
            if counts is not None:
                counts['inactive_scale_calls'] += 1
        else:
            bad.append(key[0])
    return bad, skipped


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
        if 'mirror_skipped' in original:
            # A newer reference already has the short circuit. Compare both
            # diagnostics as-is, including the reason; do not remove only one.
            continue
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
    parser.add_argument('--reference-cpu-qos', choices=('inherit', 'high'), default='inherit')
    parser.add_argument('--optimized-cpu-qos', choices=('inherit', 'high'), default='inherit')
    parser.add_argument('--isolated-reference', action='store_true',
                        help='Use the full old package, required when adapters or image sampling changed')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output directory')
    args.output.mkdir(parents=True)
    cv2.setNumThreads(1)
    config = CaptureConfig.load(args.config).merged({'runtime': {'providers': 'cpu'}})
    optimized = config.merged(json.loads(args.optimized_config.read_text(encoding='utf8')))
    reference_factory, hashes = (load_isolated_reference(args.reference_root) if args.isolated_reference
                                  else load_reference(args.reference_root))
    report = {'method': __doc__, 'reference_source_sha256': hashes,
              'config': config.to_dict(), 'optimized_config': optimized.to_dict(), 'videos': [], 'cpu_policies': {},
              'isolated_reference': args.isolated_reference}
    policies = {'reference': args.reference_cpu_qos, 'optimized': args.optimized_cpu_qos}
    for path in args.source:
        old_models = None
        if args.isolated_reference:
            original = reference_factory(config)
        else:
            old_models = build_engine(config)
            original = reference_factory(config, old_models.detector, old_models.body_gate, old_models.refiner.landmarker)
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
        mismatches, n, skipped, skipped_heads = [], 0, 0, 0
        suppressed = Counter()
        source = VideoSource(path)
        declared = int(source.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        try:
            for frame, timestamp, index in source:
                results = {}
                for name in (('reference', 'optimized') if index % 2 == 0 else ('optimized', 'reference')):
                    traces[name].clear()
                    with cpu_qos(policies[name]) as policy:
                        results[name] = engines[name].process(frame, timestamp, index)
                        report['cpu_policies'][name] = policy
                old, new = (results[name].to_dict() for name in ('reference', 'optimized'))
                old.pop('timings_ms'); new.pop('timings_ms')
                skipped += normalize_diagnostics(old, new, config.refine)
                bad_outputs, empty_heads = compare_outputs(traces['reference'], traces['optimized'], counts=suppressed)
                skipped_heads += empty_heads
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
               'conditional_empty_heads_skipped': skipped_heads,
               'inactive_scale_raw_output_changes': suppressed['inactive_scale_calls'],
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
