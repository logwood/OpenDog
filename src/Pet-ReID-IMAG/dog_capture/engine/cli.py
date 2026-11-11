"""``python -m dog_capture``: one entry point for every input and backend.

    python -m dog_capture run VIDEO|IMAGE|FOLDER|CAMERA_INDEX|rtsp://...
    python -m dog_capture models [--download yolox_tiny]
    python -m dog_capture doctor
    python -m dog_capture config --write my_config.yaml
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import platform
from pathlib import Path
import sys
import time

from . import __version__
from .config import CaptureConfig, ViewSpec
from .zoo import REPO_ROOT

VIEW_PRESETS = {'front': ViewSpec('front', 0.0, -0.13, 0.13), 'left': ViewSpec('left', -0.45, -0.75, -0.25),
                'right': ViewSpec('right', 0.45, 0.25, 0.75)}


def _overrides(args):
    o = {}
    def put(section, key, value):
        if value is not None:
            o.setdefault(section, {})[key] = value
    put('detector', 'kind', args.detector)
    put('detector', 'model', args.model)
    put('detector', 'input_size', args.input_size)
    put('detector', 'score_threshold', args.score)
    put('body_gate', 'model', args.body_model)
    if args.no_body_gate:
        put('body_gate', 'enabled', 'off')
    put('runtime', 'providers', args.providers)
    put('runtime', 'threads', args.threads)
    put('runtime', 'rotate', args.rotate)
    if args.mirror:
        put('runtime', 'mirror', True)
    put('selection', 'mode', args.mode)
    put('selection', 'top_k', args.top_k)
    if args.views:
        names = [v.strip() for v in args.views.split(',') if v.strip()]
        unknown = [v for v in names if v not in VIEW_PRESETS]
        if unknown:
            raise SystemExit(f'Unknown view(s) {unknown}; presets: {sorted(VIEW_PRESETS)} '
                             '(define custom bands in a config file)')
        o.setdefault('pose', {})['views'] = [VIEW_PRESETS[v].__dict__ for v in names]
    return o


def load_config(args):
    config = CaptureConfig.load(args.config) if args.config else CaptureConfig()
    return config.merged(_overrides(args))


def cmd_run(args):
    from .engine import CaptureEngine
    from .io import (DetectionLog, LatestFrameRunner, RunWriter, choose_output_dir, draw_overlay, open_source,
                     percentiles, upright, write_review)
    from .zoo import build_body_gate, build_engine
    config = load_config(args)
    log = DetectionLog(args.detections) if args.detections else None
    source = open_source(args.source, args.fps)
    if source.__class__.__name__ == 'ImageSource' and not config.runtime.still_images:
        config = config.merged({'runtime': {'still_images': True}})
    if log is not None:  # upstream detections: no face model is loaded
        config = config.merged({'detector': {'kind': 'external'}})
        engine = CaptureEngine(config, None, build_body_gate(config, 'external'))
    else:
        engine = build_engine(config)
    for model in (engine.detector, engine.body_gate):
        if hasattr(model, 'warmup'):
            model.warmup()
    output = choose_output_dir(args.output, args.source, REPO_ROOT)
    writer = RunWriter(output, log_frames=not args.no_frame_log,
                       preview_fps=source.fps if args.preview else None, image_format=config.output.image_format)
    guidance_counts = Counter()
    timings = {'total': [], 'detector': [], 'capture_to_result': []}
    last_code = None

    def on_result(result, frame):
        nonlocal last_code
        writer.frame(result, frame)
        guidance_counts[result.guidance['code']] += 1
        for key in timings:
            if key in result.timings_ms:
                timings[key].append(result.timings_ms[key])
        if not args.quiet and result.guidance['code'] != last_code:
            last_code = result.guidance['code']
            print(f'[{result.time_s:8.2f}s] {result.guidance["message"]}', flush=True)
        for event in result.events:
            if event['event'] in ('view_captured', 'view_locked') and not args.quiet:
                print(f'  -> {event["event"]}: {event["view"]} (track {event["track_id"]}, '
                      f'frame {event.get("frame_index", "")})', flush=True)

    realtime = args.realtime or source.live
    runner = LatestFrameRunner(engine, on_result, log.get if log else None) if realtime else None
    started = time.perf_counter()
    read = processed = 0
    show = None
    if args.show:
        import cv2
        show = cv2
    try:
        for frame, timestamp, index in source:
            if args.max_seconds and timestamp > args.max_seconds:
                break
            read += 1
            if runner is not None:
                if not source.live:  # replay a file at its own speed
                    delay = timestamp - (time.perf_counter() - started)
                    if delay > 0:
                        time.sleep(delay)
                runner.submit(frame, timestamp, index)
                latest = runner.poll()
            else:
                latest = engine.process(frame, timestamp, index,
                                        detections=log.get(index) if log else None)
                if latest is not None:
                    processed += 1
                    on_result(latest, frame)
            if show is not None:
                fresh = latest is not None and abs(timestamp - latest.time_s) < 0.25
                view = draw_overlay(upright(frame, latest), latest) if fresh else draw_overlay(frame, None)
                show.imshow('dog_capture (Esc to stop)', view)
                if show.waitKey(1) & 0xFF == 27:
                    break
    except KeyboardInterrupt:
        print('Interrupted; saving what was captured so far.', file=sys.stderr)
    finally:
        if runner is not None:
            runner.close()
            processed = runner.processed
        source.close()
        writer.close()
        if log is not None:
            log.close()
        if show is not None:
            show.destroyAllWindows()
    elapsed = time.perf_counter() - started
    best = engine.finalize()
    views = writer.save_bundle(best)
    others = [b.summary() for b in engine.bundles() if b is not best]
    report = {
        'engine': f'dog_capture {__version__}', 'python': platform.python_version(),
        'input': source.describe(), 'output_dir': str(output),
        'views': list(engine.view_names), 'models': engine.describe(),
        'selection': {'track_id': best.track_id if best else None, 'views': views,
                      'missing_views': [v for v in engine.view_names if v not in views],
                      'coverage_complete': bool(best) and all(v in views for v in engine.view_names)},
        'comparison_inputs': {v: items[0]['aligned'] for v, items in views.items()},
        'other_tracks': others,
        'stats': {'frames_read': read, 'frames_processed': processed,
                  'frames_dropped': (runner.dropped if runner else 0) + sum(engine.dropped.values()),
                  'elapsed_s': round(elapsed, 3), 'processing_fps': processed / max(elapsed, 1e-6),
                  'realtime': realtime, **{f'{k}_ms': percentiles(v) for k, v in timings.items()}},
        'guidance_counts': dict(guidance_counts.most_common()),
        'events': list(engine.events)[-200:], 'frame_log': not args.no_frame_log,
        'notes': ['Aligned crops use the PetFace 5-point template (224x224); frame/face images keep input pixels '
                  '(rotated upright if a rotation was applied).',
                  'yaw is a dimensionless landmark cue; approx_deg is display-only.',
                  'Real-time latency is measured from frame submission; camera/driver latency is excluded.'],
    }
    (output / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    write_review(output, report)
    captured = ', '.join(views) or 'none'
    print(f'\nCaptured views: {captured}; missing: {", ".join(report["selection"]["missing_views"]) or "none"}')
    print(f'Results: {output / "review.html"}')
    return 0 if views else 2


def cmd_models(args):
    from . import zoo
    if args.download:
        names = [e['name'] for e in zoo.manifest()['models'] if e.get('url')] if args.download == 'all' else [args.download]
        for name in names:
            print(f'{name}: {zoo.download(name)}')
    print(f'Model directory: {zoo.model_dir()}')
    for row in zoo.status():
        mark = 'OK ' if row['present'] and row.get('hash_ok', True) else ('BAD' if row['present'] else '-- ')
        print(f'[{mark}] {row["role"]:4} {row["name"]:18} {row["kind"]:9} {row["file"]}')
        if not row['present']:
            print(f'       get: {row.get("url") or row["obtain"]}')
        print(f'       license: {row["license"]}')
    return 0


def cmd_doctor(args):
    from . import runtime, zoo
    print(f'dog_capture {__version__} on Python {platform.python_version()} ({platform.platform()})')
    for module in ('numpy', 'cv2', 'onnxruntime', 'yaml', 'scipy', 'ai_edge_litert'):
        try:
            mod = __import__(module)
            print(f'  {module:15} {getattr(mod, "__version__", "ok")}')
        except ImportError:
            print(f'  {module:15} missing{" (required)" if module in ("numpy", "cv2") else ""}')
    print(f'  ONNX Runtime providers: {runtime.available_providers()}')
    config = CaptureConfig()
    try:
        kind, path = zoo.find_face_model(config)
        print(f'  face detector (auto): {kind} -> {path}')
    except FileNotFoundError as exc:
        print(f'  face detector: NOT FOUND\n    {exc}')
        kind = None
    gate = zoo.build_body_gate(config, kind) if kind else None
    print(f'  body gate: {gate.model_path if gate else "none (faces are not species-checked)"}')
    return 0


def cmd_config(args):
    value = CaptureConfig().to_dict()
    if args.write:
        path = Path(args.write)
        if path.suffix.lower() in ('.yaml', '.yml'):
            import yaml
            path.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding='utf8')
        else:
            path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf8')
        print(f'Wrote {path}')
    else:
        print(json.dumps(value, indent=2, ensure_ascii=False))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m dog_capture', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run', help='capture best shots from a video, photos, camera or stream')
    run.add_argument('source', help='video file, image file/folder, camera index or stream URL')
    run.add_argument('--config', help='JSON/YAML config (see `config --write`)')
    run.add_argument('--output', help='output directory (never overwritten)')
    run.add_argument('--detector', choices=['auto', 'yolo_pose', 'anyface', 'scrfd', 'dogflw'])
    run.add_argument('--model', help='face detector ONNX file (or DogFLW directory)')
    run.add_argument('--detections', help='upstream detections JSONL instead of a face model (integration mode)')
    run.add_argument('--input-size', type=int)
    run.add_argument('--score', type=float, help='face detector score threshold')
    run.add_argument('--body-model', help='YOLOX ONNX for the dog species gate')
    run.add_argument('--no-body-gate', action='store_true')
    run.add_argument('--providers', help='auto | cpu | cuda | tensorrt | coreml | directml | openvino, comma list')
    run.add_argument('--threads', type=int)
    run.add_argument('--views', help='comma list of front,left,right (default all three)')
    run.add_argument('--mode', choices=['best', 'fast', 'hybrid'])
    run.add_argument('--top-k', type=int)
    run.add_argument('--rotate', choices=['auto', '0', '90', '180', '270'])
    run.add_argument('--mirror', action='store_true', help='flip horizontally (selfie camera)')
    run.add_argument('--fps', type=float, help='only for files without usable timestamps')
    run.add_argument('--realtime', action='store_true', help='replay a file at its own speed, dropping frames like a camera')
    run.add_argument('--show', action='store_true', help='live window (needs opencv-python, not -headless)')
    run.add_argument('--preview', action='store_true', help='write an annotated preview.mp4')
    run.add_argument('--no-frame-log', action='store_true', help='skip frames.jsonl')
    run.add_argument('--max-seconds', type=float, default=0)
    run.add_argument('--quiet', action='store_true')
    run.set_defaults(func=cmd_run)
    models = sub.add_parser('models', help='list, verify and download models')
    models.add_argument('--download', help='model name from the list, or "all" for downloadable ones')
    models.set_defaults(func=cmd_models)
    doctor = sub.add_parser('doctor', help='check dependencies, providers and models')
    doctor.set_defaults(func=cmd_doctor)
    config = sub.add_parser('config', help='print or write the default configuration')
    config.add_argument('--write', help='file to write (.json or .yaml)')
    config.set_defaults(func=cmd_config)
    args = parser.parse_args(argv)
    return args.func(args)
