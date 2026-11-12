"""``python -m dog_capture``: one entry point for every input and backend.

    python -m dog_capture run VIDEO|IMAGE|FOLDER|CAMERA_INDEX|rtsp://...
    python -m dog_capture models [--download yolox_nano]
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
    put('refine', 'landmarks', getattr(args, 'landmarks', None))
    if args.no_body_gate:
        put('body_gate', 'enabled', 'off')
    put('runtime', 'providers', args.providers)
    put('runtime', 'threads', args.threads)
    put('runtime', 'cuda_graph', args.cuda_graph)
    put('runtime', 'thread_spinning', args.thread_spinning)
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
    for model in (engine.detector, engine.body_gate, getattr(engine.refiner, 'landmarker', None)):
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
    until_done = args.until_done if args.until_done is not None else source.live
    runner = LatestFrameRunner(engine, on_result, log.get if log else None) if realtime else None
    started = time.perf_counter()
    read = processed = 0
    show = None
    if args.show:
        import cv2
        try:  # opencv-python-headless has no GUI: warn and continue without a window
            cv2.namedWindow('dog_capture (Esc to stop)', cv2.WINDOW_NORMAL)
            show = cv2
        except cv2.error:
            print('warning: this OpenCV build has no GUI support (opencv-python-headless?); '
                  'install opencv-python for --show. Continuing without a window.', file=sys.stderr)
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
            if until_done and engine.complete():
                if not args.quiet:
                    print('All views captured and locked; stopping early.', flush=True)
                break
            if show is not None:
                fresh = latest is not None and abs(timestamp - latest.time_s) < 0.25
                same = fresh and latest.frame_index == index  # boxes only on the analysed frame
                view = (draw_overlay(upright(frame, latest), latest, boxes=same) if fresh
                        else draw_overlay(frame, None))
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
                      'coverage_complete': bool(best) and all(v in views for v in engine.view_names),
                      # credible faces (landmarks verified) that missed only the quality level
                      'near_misses': {v: engine.near_misses[v] for v in engine.view_names
                                      if v not in views and v in engine.near_misses}},
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


def cmd_bench(args):
    """Per-stage latency and size on this device: detector, gate, engine overhead."""
    import numpy as np
    from .io import open_source
    from ..runtime import peak_rss_mb
    from .zoo import build_engine
    config = load_config(args)
    engine = build_engine(config.merged({'runtime': {'rotate': '0'}}))
    frames = []
    if args.source:
        source = open_source(args.source)
        for frame, _, _ in source:
            frames.append(frame)
            if len(frames) >= args.frames:
                break
        source.close()
    if not frames:
        rng = np.random.default_rng(0)
        frames = [(rng.random((720, 1280, 3)) * 255).astype(np.uint8) for _ in range(8)]
    def timed(fn, frame, runs):
        values = []
        for _ in range(runs):
            started = time.perf_counter()
            fn(frame)
            values.append((time.perf_counter() - started) * 1000)
        return values
    report = {'device_providers': engine.detector.runtime['used'] if hasattr(engine.detector, 'runtime') else None,
              'frames': len(frames), 'frame_shape': list(frames[0].shape)}
    for name, model in (('detector', engine.detector), ('body_gate', engine.body_gate)):
        if model is None:
            continue
        model.warmup(3) if hasattr(model, 'warmup') else None
        values = [v for f in frames for v in timed(model.detect, f, max(1, args.runs // len(frames)))]
        path = getattr(model, 'model_path', None)
        report[name] = {'model': str(path) if path else None,
                        'megabytes': round(Path(path).stat().st_size / 1e6, 2) if path and Path(path).is_file() else None,
                        'input_size': getattr(model, 'size', None),
                        'ms_p50': round(float(np.median(values)), 2), 'ms_p95': round(float(np.percentile(values, 95)), 2)}
    totals, overhead = [], []
    for i, frame in enumerate(frames * max(1, args.runs // len(frames))):
        result = engine.process(frame, i / 30.0, i)
        totals.append(result.timings_ms['total'])
        overhead.append(result.timings_ms['total'] - sum(result.timings_ms.get(k, 0) for k in
                                                         ('detector', 'roi_detector', 'body_gate', 'head_gate')))
    report['pipeline_ms'] = {'p50': round(float(np.median(totals)), 2), 'p95': round(float(np.percentile(totals, 95)), 2),
                             'fps_at_p50': round(1000 / max(float(np.median(totals)), 1e-6), 1)}
    report['engine_overhead_ms'] = {'p50': round(float(np.median(overhead)), 2),
                                    'p95': round(float(np.percentile(overhead, 95)), 2)}
    report['peak_rss_mb'] = peak_rss_mb()
    print(json.dumps(report, indent=2))
    return 0


def cmd_models(args):
    from . import zoo
    if args.import_face:
        print(f'Imported reference: {zoo.import_face(args.import_face, args.metadata)}')
    elif args.metadata:
        raise ValueError('--metadata requires --import-face')
    if args.download:
        names = [e['name'] for e in zoo.manifest()['models'] if e.get('url')] if args.download == 'all' else [args.download]
        for name in names:
            print(f'{name}: {zoo.download(name)}')
    print(f'Model directory: {zoo.model_dir()}')
    rows = zoo.status()
    for row in rows:
        mark = 'OK ' if row['present'] and row.get('hash_ok', True) else ('BAD' if row['present'] else '-- ')
        print(f'[{mark}] {row["role"]:4} {row["name"]:18} {row["kind"]:9} {row["file"]}')
        if not row['present']:
            print(f'       get: {row.get("url") or row["obtain"]}')
        print(f'       license: {row["license"]}')
    return 1 if any(row.get('hash_ok') is False for row in rows) else 0


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
    return 0 if kind else 1


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
    run.add_argument('--landmarks', help="crop landmark ONNX for per-frame tracking, or 'off' (default: "
                     "dog_face_landmarks.onnx in the model directory if present)")
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
    run.add_argument('--until-done', dest='until_done', action='store_true', default=None,
                     help='stop once every view is captured and locked (default for cameras/streams)')
    run.add_argument('--no-until-done', dest='until_done', action='store_false')
    run.add_argument('--quiet', action='store_true')
    run.set_defaults(func=cmd_run)
    bench = sub.add_parser('bench', help='measure model size and per-stage latency on this device')
    bench.add_argument('source', nargs='?', help='video/images to use (default: random 720p frames)')
    bench.add_argument('--config')
    bench.add_argument('--frames', type=int, default=30)
    bench.add_argument('--runs', type=int, default=60)
    for flag in ('--detector', '--model', '--body-model', '--providers', '--views', '--mode', '--rotate'):
        bench.add_argument(flag, default=None)
    bench.add_argument('--threads', type=int)
    bench.add_argument('--input-size', type=int)
    bench.add_argument('--score', type=float)
    bench.add_argument('--top-k', type=int)
    bench.add_argument('--no-body-gate', action='store_true')
    bench.add_argument('--mirror', action='store_true')
    bench.set_defaults(func=cmd_bench)
    for command in (run, bench):
        command.add_argument('--cuda-graph', action=argparse.BooleanOptionalAction, default=None,
                             help='warm and use FP32 CUDA Graph with CUDA first, falling back to ordinary inference')
        command.add_argument('--thread-spinning', action=argparse.BooleanOptionalAction, default=None,
                             help='allow ONNX Runtime worker threads to spin while idle (default: enabled)')
    models = sub.add_parser('models', help='list, verify, import and download models')
    models.add_argument('--download', help='model name from the list, or "all" for downloadable ones')
    models.add_argument('--import-face', help='install the registered dog_face_pose.onnx reference')
    models.add_argument('--metadata', help='JSON sidecar for --import-face (default: same basename)')
    models.set_defaults(func=cmd_models)
    doctor = sub.add_parser('doctor', help='check dependencies, providers and models')
    doctor.set_defaults(func=cmd_doctor)
    config = sub.add_parser('config', help='print or write the default configuration')
    config.add_argument('--write', help='file to write (.json or .yaml)')
    config.set_defaults(func=cmd_config)
    parser.add_argument('--debug', action='store_true', help='show Python tracebacks')
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as exc:
        if args.debug:
            raise
        print(f'error: {exc}', file=sys.stderr)
        return 1
