"""Compare quality/clarity ranking on identical full video inputs, saving both selections.

Only selection.ranking changes. All gates, models, sampling and lock thresholds
are inherited. This measures selection changes, NOT identity accuracy or live
throughput; use benchmark_dog_capture_v1.py separately for real-time trials.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import html
import json
from pathlib import Path
import sys

import cv2

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from dog_capture.engine.config import CaptureConfig
from dog_capture.engine.io import RunWriter, VideoSource
from dog_capture.engine.process_policy import cpu_qos
from dog_capture.engine.selection import clarity_score, LEVEL_ORDER
from dog_capture.engine.zoo import build_engine


def count_models(engine):
    counts = Counter()
    models = [('face', engine.detector), ('body', engine.body_gate),
              ('landmarks', engine.refiner.landmarker if engine.refiner else None)]
    for label, model in models:
        if model is None:
            continue
        model.warmup(15)
        def run(*args, label=label, original=model._run, **kwargs):
            counts[label] += 1
            return original(*args, **kwargs)
        model._run = run
    return counts


def audit_checks(engine):
    counts = Counter()
    original = engine._trusted
    def check(*args, **kwargs):
        reason, status = original(*args, **kwargs)
        counts[reason or 'passed'] += 1
        return reason, status
    engine._trusted = check
    return counts


def compare_video(config, path, output):
    engines = {ranking: build_engine(config.merged({'selection': {'ranking': ranking}}))
               for ranking in ('quality', 'clarity')}
    calls = {name: count_models(engine) for name, engine in engines.items()}
    checks = {name: audit_checks(engine) for name, engine in engines.items()}
    source = VideoSource(path)
    declared, count = int(source.capture.get(cv2.CAP_PROP_FRAME_COUNT)), 0
    log_dir = output / path.stem
    log_dir.mkdir(parents=True)
    try:
        with (log_dir / 'frames.jsonl').open('w', encoding='utf8') as log:
            for frame, timestamp, index in source:
                row = {'frame_index': index, 'time_s': timestamp}
                for name in (('quality', 'clarity') if index % 2 == 0 else ('clarity', 'quality')):
                    result = engines[name].process(frame, timestamp, index)
                    row[name] = {'guidance': result.guidance['code'],
                                 'accepted_tracks': [face.track_id for face in result.faces if face.accepted],
                                 'events': result.events}
                log.write(json.dumps(row) + '\n')
                count += 1
                if count % 400 == 0:
                    print(path.name, count, 'frames', flush=True)
    finally:
        source.close()
    if declared - count not in (0, 1):
        raise AssertionError(f'Incomplete input: {count}/{declared}')
    selections, assessments = {}, {}
    for name, engine in engines.items():
        folder = log_dir / name
        folder.mkdir()
        bundle = engine.finalize()
        writer = RunWriter(folder, log_frames=False, image_format='png')
        selections[name] = writer.save_bundle(bundle)
        writer.close()
        assessments[name] = []
        if bundle:
            for view, shots in bundle.slots.items():
                for shot in shots:
                    q = shot.quality
                    spec = next(v for v in config.pose.views if v.name == view)
                    allowed = (q.passed and not q.reasons
                               and LEVEL_ORDER[q.level] >= LEVEL_ORDER[config.selection.min_level]
                               and spec.lower - config.pose.exit_margin <= shot.geometry.yaw <= spec.upper + config.pose.exit_margin
                               and abs(shot.geometry.roll_deg) <= spec.max_roll_deg)
                    if not allowed:
                        raise AssertionError(f'{name} selected an ineligible shot: {view}/{shot.frame_index}')
                    assessments[name].append({'view': view, 'frame_index': shot.frame_index,
                        'quality_score': q.score, 'clarity_score': clarity_score(q.factors),
                        'factors': q.factors, 'metrics': q.metrics, 'yaw': shot.geometry.yaw,
                        'hard_gates_pass': allowed,
                        'aligned_sha256': hashlib.sha256(shot.aligned.tobytes()).hexdigest()})
    row = {'source': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
           'frames': count, 'output': str(log_dir.resolve()),
           'model_calls': {k: dict(v) for k, v in calls.items()},
           'trusted_checks': {k: dict(v) for k, v in checks.items()},
           'selection': selections, 'assessment': assessments}
    print(path.name, {name: [(s['view'], s['frame_index'], round(s['clarity_score'], 4))
                             for s in scores] for name, scores in assessments.items()}, flush=True)
    del engines, engine, bundle
    gc.collect()
    return row


def review(report, output):
    parts = ['<!doctype html><meta charset="utf-8"><title>Quality / clarity selection</title>',
             '<style>body{font:16px system-ui;margin:24px;background:#f2f4f7;color:#18202d}'
             'table{border-collapse:collapse;background:white}td,th{padding:14px;border:1px solid #ccc;vertical-align:top}'
             'img{width:224px;height:224px}pre{white-space:pre-wrap;max-width:420px}</style>',
             '<h1>Quality / clarity selection</h1><p>Identical gates and input frames. '
             'These scores do not establish identity-match accuracy.</p>']
    for row in report['videos']:
        parts.append('<h2>' + html.escape(Path(row['source']).name) + '</h2><table><tr><th>View</th><th>Quality</th><th>Clarity</th></tr>')
        views = sorted(set().union(*(v.keys() for v in row['selection'].values())))
        for view in views:
            parts.append('<tr><th>' + html.escape(view) + '</th>')
            for name in ('quality', 'clarity'):
                selected = row['selection'][name].get(view, [])
                if not selected:
                    parts.append('<td>No qualified shot</td>')
                    continue
                shot = selected[0]
                src = (Path(row['output']) / name / shot['aligned']).relative_to(output.resolve()).as_posix()
                assessment = next(s for s in row['assessment'][name] if s['view'] == view)
                details = {k: assessment[k] for k in ('frame_index', 'quality_score', 'clarity_score', 'yaw', 'factors')}
                parts.append(f'<td><img src="{html.escape(src, quote=True)}"><pre>{html.escape(json.dumps(details, indent=2))}</pre></td>')
            parts.append('</tr>')
        parts.append('</table>' if views else '</table><p>No qualified shots in either strategy.</p>')
    (output / 'review.html').write_text('\n'.join(parts), encoding='utf8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True, action='append')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-qos', choices=('inherit', 'high'), default='high')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output directory')
    if len({p.stem for p in args.source}) != len(args.source):
        parser.error('Input videos need distinct names')
    config = CaptureConfig.load(args.config)
    cv2.setNumThreads(1)
    args.output.mkdir(parents=True)
    report = {'method': __doc__, 'config': config.to_dict(), 'videos': []}
    with cpu_qos(args.cpu_qos) as policy:
        report['cpu_policy'] = policy
        for path in args.source:
            report['videos'].append(compare_video(config, path, args.output))
            (args.output / 'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    review(report, args.output)


if __name__ == '__main__':
    main()
