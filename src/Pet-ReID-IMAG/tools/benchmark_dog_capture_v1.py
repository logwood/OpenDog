"""Measure the shipped v1 ONNX models and the unmodified capture pipeline.

Run one backend per process, and run processes sequentially. Video replay uses
the production capacity-one worker at the source's timestamps, without early
completion, preview rendering, frame-log writes, HTTP, or Re-ID inference.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src/Pet-ReID-IMAG'))
DLL_HANDLES = []


def quantiles(values):
    import numpy as np
    if not values:
        return {'count': 0}
    data = np.asarray(values, dtype=np.float64)
    return {'count': len(data), 'mean': float(data.mean()),
            **{f'p{q}': float(np.percentile(data, q)) for q in (50, 95, 99)},
            'max': float(data.max())}


def relative(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def gpu_snapshot():
    try:
        return subprocess.check_output([
            'nvidia-smi', '--query-gpu=name,driver_version,pstate,utilization.gpu,memory.used,power.draw,clocks.gr',
            '--format=csv'], text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def preload_cuda():
    """Process-local DLL loading; no package installs or system PATH changes."""
    import onnxruntime as ort
    spec = importlib.util.find_spec('torch')
    if os.name == 'nt' and spec is not None:
        directory = Path(spec.origin).parent / 'lib'
        if directory.is_dir():
            DLL_HANDLES.append(os.add_dll_directory(str(directory)))
            # cuDNN's NVRTC dependency also searches the process PATH.
            os.environ['PATH'] = str(directory) + os.pathsep + os.environ.get('PATH', '')
    if hasattr(ort, 'preload_dlls'):
        ort.preload_dlls()


def sample_video(path, count):
    """Sequential decode, evenly spaced samples including the last frame.

    Sampling only the first frames can benchmark an empty scene. Sequential
    decoding also avoids backend-specific seeking errors in OGV/WebM files.
    """
    import cv2
    import numpy as np
    from dog_capture.engine.io import VideoSource
    source = VideoSource(path)
    declared = int(source.capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if declared <= 0:
        source.close()
        raise ValueError(f'Video must declare a frame count: {path}')
    indices = set(np.linspace(0, declared - 1, min(count, declared), dtype=int).tolist())
    samples, decoded, last = [], 0, None
    try:
        for frame, timestamp, index in source:
            decoded += 1
            last = (frame, timestamp, index)
            if index in indices:
                samples.append((frame, timestamp, index))
        # Some containers round duration * FPS up by one (the archived OGV
        # declares 1,213 but decodes 1,212 frames). Keep the actual final frame.
        if last is not None and decoded == declared - 1 and len(samples) < len(indices):
            samples.append(last)
        info = {**source.describe(), 'path': relative(path), 'decoded_frames': decoded,
                'declared_minus_decoded': declared - decoded,
                'sha256': sha256(path), 'sample_indices': [s[2] for s in samples],
                'shape': list(samples[0][0].shape) if samples else None}
    finally:
        source.close()
    if len(samples) != len(indices) or declared - decoded not in (0, 1):
        raise ValueError(f'Incomplete decode: {path}: {decoded}/{declared}, samples={len(samples)}/{len(indices)}')
    return samples, info


def verify_provider(model, tensor, output, label):
    """A separate profiled session proves which provider executes graph nodes.

    Profiling is never enabled on the sessions being timed. CPU comparison is
    numerical evidence on this tensor, not a held-out model accuracy claim.
    """
    import numpy as np
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.enable_profiling = True
    options.profile_file_prefix = str(output / f'provider-{label}')
    options.log_severity_level = 3
    providers = []
    for provider in model.runtime['used']:
        settings = dict(model.runtime['provider_options'].get(provider, {}))
        settings.pop('enable_cuda_graph', None)  # profile individual kernels in an ordinary session
        providers.append((provider, settings))
    session = ort.InferenceSession(str(model.model_path), sess_options=options, providers=providers)
    got = session.run(model.outputs, {model.input.name: tensor})
    profile = session.end_profiling()
    events = json.loads(Path(profile).read_text(encoding='utf8'))
    nodes = Counter(event.get('args', {}).get('provider') for event in events
                    if event.get('cat') == 'Node' and event.get('args', {}).get('provider'))
    expected = model.runtime['used'][0]
    if nodes[expected] == 0:
        raise RuntimeError(f'{label}: no nodes actually executed on {expected}: {dict(nodes)}')
    result = {'node_executions_by_provider': dict(nodes), 'profile': Path(profile).name}
    if expected != 'CPUExecutionProvider':
        cpu_options = ort.SessionOptions()
        cpu_options.intra_op_num_threads = 4
        reference = ort.InferenceSession(str(model.model_path), sess_options=cpu_options,
                                        providers=['CPUExecutionProvider'])
        want = reference.run(model.outputs, {model.input.name: tensor})
        result['cpu_comparison'] = [{'shape': list(a.shape),
            'max_abs_difference': float(np.max(np.abs(a - b))),
            'mean_abs_difference': float(np.mean(np.abs(a - b))),
            'allclose_atol_0_01_rtol_0_001': bool(np.allclose(a, b, atol=.01, rtol=.001))}
            for a, b in zip(got, want)]
    return result


def model_benchmark(model, frames, warmup, runs):
    tensors = [model.preprocess(frame)[0].astype(model.dtype, copy=False) for frame in frames]
    result = {}
    # session.run synchronizes returned CPU outputs, including GPU transfers.
    for name, fn, inputs in (('inference_ms', model._run, tensors),
                             ('detect_ms', model.detect, frames)):
        for i in range(warmup):
            fn(inputs[i % len(inputs)])
        latencies = []
        for i in range(runs):
            started = time.perf_counter()
            fn(inputs[i % len(inputs)])
            latencies.append((time.perf_counter() - started) * 1000)
        result[name] = quantiles(latencies)
    return result


def decoded_comparison(engine, config, frames):
    """Compare actual boxes/keypoints, separately from low-confidence raw anchors.

    This runs outside every performance timer. It is a numerical backend
    check on sampled frames, not a pose-accuracy or species-accuracy metric.
    """
    import numpy as np
    from dog_capture.engine.schema import box_iou_matrix
    from dog_capture.engine.zoo import build_engine
    reference = build_engine(config.merged({'runtime': {'providers': 'cpu', 'cuda_graph': False}}))
    reports = {}
    for name in ('detector', 'body_gate'):
        actual, cpu = getattr(engine, name), getattr(reference, name)
        mismatches, boxes, points, scores, kp_scores, count = [], [], [], [], [], 0
        for index, frame in enumerate(frames):
            got, want = actual.detect(frame), cpu.detect(frame)
            if name == 'detector':
                abox = np.asarray([d.box for d in got]).reshape(-1, 4)
                bbox = np.asarray([d.box for d in want]).reshape(-1, 4)
                ascore, bscore = [d.score for d in got], [d.score for d in want]
            else:
                (abox, ascore), (bbox, bscore) = got, want
            overlap = box_iou_matrix(abox, bbox)
            matched = 0
            while overlap.size and overlap.max() >= .5:
                a, b = np.unravel_index(np.argmax(overlap), overlap.shape)
                boxes.extend(np.abs(abox[a] - bbox[b]).tolist())
                scores.append(abs(float(ascore[a]) - float(bscore[b])))
                if name == 'detector':
                    points.extend(np.abs(got[a].keypoints - want[b].keypoints).reshape(-1).tolist())
                    if got[a].kp_scores is not None and want[b].kp_scores is not None:
                        kp_scores.extend(np.abs(got[a].kp_scores - want[b].kp_scores).tolist())
                overlap[a, :] = -1
                overlap[:, b] = -1
                matched += 1
            count += matched
            if matched != len(abox) or matched != len(bbox):
                mismatches.append({'sample': index, 'backend_detections': len(abox),
                                   'cpu_detections': len(bbox), 'matched': matched})
        reports[name] = {'frames': len(frames), 'matched_detections': count, 'mismatches': mismatches,
            'box_abs_difference_px': quantiles(boxes), 'keypoint_abs_difference_px': quantiles(points),
            'score_abs_difference': quantiles(scores), 'keypoint_score_abs_difference': quantiles(kp_scores)}
    return reports


def video_benchmark(config, detector, gate, path, realtime, output, repeat, expected_frames, *, landmarker=None):
    from dog_capture.engine.engine import CaptureEngine
    from dog_capture.engine.io import LatestFrameRunner, RunWriter, VideoSource
    engine = CaptureEngine(config, detector, gate, landmarker)  # fresh tracking/selection state
    source = VideoSource(path)
    info = source.describe()
    rows, times, guidance = [], defaultdict(list), Counter()
    calls, originals = Counter(), []
    def counted(name, original):
        def detect(*args, **kwargs):
            calls[name] += 1
            return original(*args, **kwargs)
        return detect
    for name, model in (('face', detector), ('body', gate), ('landmarks', landmarker)):
        if model is not None:
            originals.append((model, model.detect))
            model.detect = counted(name, model.detect)
    started = time.perf_counter()

    def on_result(result, frame):
        for key, value in result.timings_ms.items():
            times[key].append(value)
        # End-to-end from scheduled video PTS, also catches a late producer.
        if realtime:
            times['scheduled_to_result'].append((time.perf_counter() - started - result.time_s) * 1000)
        guidance[result.guidance['code']] += 1
        rows.append({'frame_index': result.frame_index, 'time_s': result.time_s,
                     'timings_ms': result.timings_ms, 'faces': len(result.faces),
                     'rotation_deg': result.rotation_deg, 'guidance': result.guidance['code']})

    runner = LatestFrameRunner(engine, on_result) if realtime else None
    read = processed = 0
    try:
        iterator = iter(source)
        while True:
            before_decode = time.perf_counter()
            try:
                frame, timestamp, index = next(iterator)
            except StopIteration:
                break
            times['decode'].append((time.perf_counter() - before_decode) * 1000)
            read += 1
            if runner is not None:
                delay = timestamp - (time.perf_counter() - started)
                if delay > 0:
                    time.sleep(delay)
                times['producer_lateness'].append(max(0, time.perf_counter() - started - timestamp) * 1000)
                runner.submit(frame, timestamp, index)
            else:
                result = engine.process(frame, timestamp, index)
                if result is not None:
                    on_result(result, frame)
                    processed += 1
    finally:
        if runner:
            runner.close()
            processed = runner.processed
        source.close()
        for model, original in originals:
            model.detect = original
    elapsed = time.perf_counter() - started
    dropped = (runner.dropped if runner else 0) + sum(engine.dropped.values())
    if read != processed + dropped:
        raise AssertionError(f'Unaccounted frames: read={read} processed={processed} dropped={dropped}')
    if read != expected_frames:
        raise AssertionError(f'Incomplete full-video run: {read}/{expected_frames}')
    # Keep auditable timing traces and actual selected images outside the timer.
    output.mkdir(parents=True, exist_ok=False)
    (output / 'timings.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n', encoding='utf8')
    writer = RunWriter(output, log_frames=False, image_format=config.output.image_format)
    best = engine.finalize()
    selected = writer.save_bundle(best)
    writer.close()
    result = {'source': relative(path), 'models': engine.describe(),
              'landmarks': landmarker.describe() if landmarker is not None else None,
              'model_calls': dict(calls),
              'playback': 'realtime' if realtime else 'offline',
              'repeat': repeat, 'source_fps': info['fps'], 'frames_read': read,
              'frames_processed': processed, 'frames_dropped': dropped, 'drop_rate': dropped / max(read, 1),
              'elapsed_s': elapsed, 'processed_fps': processed / elapsed,
              'timings_ms': {key: quantiles(value) for key, value in times.items()},
              'frames_with_faces': sum(r['faces'] > 0 for r in rows),
              'over_source_budget_frames': sum(r['timings_ms']['total'] > 1000 / info['fps'] for r in rows),
              'guidance_counts': dict(guidance), 'selection': selected,
              'missing_views': [v for v in engine.view_names if v not in selected],
              'output': relative(output)}
    (output / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'assets/dog_capture/v1_reference/capture_quality.json')
    parser.add_argument('--providers', default='cpu', choices=('cpu', 'cuda', 'tensorrt'))
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--cuda-graph', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--thread-spinning', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--cv-threads', type=int, default=0, help='0 preserves the installed OpenCV default')
    parser.add_argument('--stage', choices=('micro', 'video', 'all'), default='all')
    parser.add_argument('--playback', choices=('offline', 'realtime', 'both'), default='both')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--samples', type=int, default=24)
    parser.add_argument('--runs', type=int, default=300)
    parser.add_argument('--warmup', type=int, default=30)
    parser.add_argument('--preload-cuda', action='store_true')
    parser.add_argument('--verify-provider', action='store_true')
    parser.add_argument('--verify-decoded', action='store_true', help='compare decoded sampled detections with CPU after timings')
    parser.add_argument('--verify-decoded-fp32', action='store_true',
                        help='require CUDA FP32 and compare decoded detections with CPU after timings')
    args = parser.parse_args()
    if min(args.samples, args.runs, args.repeats, args.warmup, args.threads) <= 0 or args.cv_threads < 0:
        parser.error('samples/runs/repeats/warmup/threads must be positive; cv-threads must be nonnegative')
    if args.output.exists():
        parser.error('Use a fresh output directory')
    if args.verify_decoded_fp32 and args.providers != 'cuda':
        parser.error('--verify-decoded-fp32 requires --providers cuda')
    args.output.mkdir(parents=True)
    if args.preload_cuda:
        preload_cuda()
    import cv2
    import numpy as np
    import onnxruntime as ort
    from dog_capture.engine.config import CaptureConfig
    from dog_capture.engine.runtime import ALIASES
    from dog_capture.engine.zoo import build_engine
    from dog_capture.runtime import peak_rss_mb
    if args.cv_threads:
        cv2.setNumThreads(args.cv_threads)
    runtime = {'providers': args.providers, 'threads': args.threads,
               'cache_dir': str(ROOT / '.tmp/dog-capture-v1-ort-cache')}
    for key in ('cuda_graph', 'thread_spinning'):
        value = getattr(args, key)
        if value is not None:
            runtime[key] = value
    config = CaptureConfig.load(args.config).merged({'runtime': runtime})
    report = {'created_utc': datetime.now(timezone.utc).isoformat(),
        'command': sys.argv, 'python': sys.version, 'executable': sys.executable,
        'platform': platform.platform(), 'processor': platform.processor(), 'logical_cpus': os.cpu_count(),
        'versions': {'numpy': np.__version__, 'opencv': cv2.__version__, 'onnxruntime': ort.__version__},
        'cv_threads': cv2.getNumThreads(), 'config': config.to_dict(), 'gpu_before': gpu_snapshot(),
        'method': {'warmup_calls_per_model': args.warmup, 'micro_timed_calls_per_stage': args.runs,
            'micro_sampling': 'uniform across each whole video; sequential decode; interleaved videos',
            'inference': 'prepared CPU tensor to returned CPU outputs, including GPU transfers/synchronization',
            'detect': 'BGR frame preprocessing + inference + decode/NMS',
            'total': 'production CaptureEngine.process; all gates/rotation/tracking/quality/selection enabled',
            'capture_to_result': 'submission to result, including worker queue; excludes camera/driver',
            'scheduled_to_result': 'scheduled video PTS to result, including producer lateness',
            'excluded': 'preview/UI rendering, network, Re-ID, final image saving; models warm before replay',
            'startup': 'new model sessions in an existing Python process; OS file cache not flushed'},
        'sources': [], 'micro': {}, 'videos': []}

    started = time.perf_counter()
    engine = build_engine(config)
    landmarker = getattr(engine.refiner, 'landmarker', None)
    report['session_load_ms'] = (time.perf_counter() - started) * 1000
    expected = ALIASES[args.providers]

    def save():
        report['models'] = engine.describe()
        report['landmarks'] = landmarker.describe() if landmarker is not None else None
        (args.output / 'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf8')
        for model in (engine.detector, engine.body_gate):
            if model is None or model.runtime['used'][0] != expected or model.runtime['fallback_errors']:
                raise RuntimeError(f'Requested provider must run both models without fallback: {expected}')
            graph = model.runtime['cuda_graph']
            if config.runtime.cuda_graph and not graph['active']:
                raise RuntimeError(f'{model.kind}: CUDA Graph fallback: {graph["fallback_reason"]}')
        if landmarker is not None:
            if landmarker.runtime['used'][0] != expected or landmarker.runtime['fallback_errors']:
                raise RuntimeError(f'Landmarker did not use the requested provider: {expected}')
            if config.runtime.cuda_graph and not landmarker.runtime['cuda_graph']['active']:
                raise RuntimeError('Landmarker CUDA Graph fallback')

    save()
    sample_sets = []
    for path in args.source:
        samples, info = sample_video(path, args.samples)
        sample_sets.append(samples)
        report['sources'].append(info)
    # Alternate source videos so a device state change does not favour one.
    frames = [group[i % len(group)][0] for i in range(max(map(len, sample_sets))) for group in sample_sets]
    for name, model in (('face', engine.detector), ('body', engine.body_gate)):
        first = time.perf_counter()
        model.detect(frames[0])
        record = {'first_detect_ms': (time.perf_counter() - first) * 1000}
        if args.verify_provider:
            tensor = model.preprocess(frames[len(frames) // 2])[0].astype(model.dtype, copy=False)
            record['provider_verification'] = verify_provider(model, tensor, args.output, name)
        if args.stage in ('micro', 'all'):
            record.update(model_benchmark(model, frames, args.warmup, args.runs))
        else:
            for i in range(args.warmup):
                model.detect(frames[i % len(frames)])
        report['micro'][name] = record
        save()
        print(f'{name}: {record}', flush=True)
    if args.stage in ('video', 'all'):
        if landmarker is not None:
            landmarker.warmup(args.warmup)
        for repeat in range(1, args.repeats + 1):
            for path, source_info in zip(args.source, report['sources']):
                for playback in ('offline', 'realtime') if args.playback == 'both' else (args.playback,):
                    name = f'{path.stem}-{playback}-{repeat}'
                    print(f'Starting {name}', flush=True)
                    row = video_benchmark(config, engine.detector, engine.body_gate, path,
                                          playback == 'realtime', args.output / name, repeat,
                                          source_info['decoded_frames'], landmarker=landmarker)
                    report['videos'].append(row)
                    save()
                    print(f'{name}: {row["processed_fps"]:.2f} fps; '
                          f'p95={row["timings_ms"]["total"]["p95"]:.2f} ms; '
                          f'dropped={row["frames_dropped"]}/{row["frames_read"]}', flush=True)
    report['peak_rss_mb'] = peak_rss_mb()
    report['gpu_after'] = gpu_snapshot()
    if args.verify_decoded or args.verify_decoded_fp32:
        report['decoded_cpu_comparison'] = decoded_comparison(engine, config, frames)
    if args.verify_decoded_fp32:
        for model in (engine.detector, engine.body_gate):
            options = model.session.get_provider_options()['CUDAExecutionProvider']
            if options.get('use_tf32') != '0':
                raise RuntimeError('Strict FP32 numerical check requires use_tf32=0')
        report['decoded_cpu_comparison_tf32_disabled'] = report['decoded_cpu_comparison']
        report['strict_fp32_check_note'] = 'CUDA had use_tf32=0 throughout timing and decoded comparison.'
    save()
    print(args.output / 'summary.json', flush=True)


if __name__ == '__main__':
    main()
