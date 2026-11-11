"""Run all performance comparisons serially, with identical selection policy."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dog_capture.cli import ROOT


def run(command, log):
    print('Running', log.stem, flush=True)
    with log.open('w', encoding='utf8') as stream:
        subprocess.run([sys.executable, *map(str, command)], stdout=stream, stderr=stream, check=True)


def summarize(output):
    rows = []
    for path in sorted(output.rglob('result.json')):
        if 'validation' not in path.parts and 'comparison' not in path.parts:
            continue
        report = json.loads(path.read_text(encoding='utf8'))
        frames = [json.loads(line) for line in (path.parent / 'frames.jsonl').read_text(encoding='utf8').splitlines()]
        groups = {}
        for name, records in [('all', frames), ('neural', [r for r in frames if r.get('inference_mode', '').startswith('neural')]),
                              ('accepted', [r for r in frames if r['quality']['accepted']])]:
            if records:
                values = [r['compute_ms'] for r in records]
                groups[name] = {'n': len(values), 'p50_ms': float(np.median(values)),
                                'p95_ms': float(np.percentile(values, 95)), 'max_ms': max(values)}
        rows.append({'run': path.parent.relative_to(output).as_posix(),
                     'selected': {key: value['frame_index'] for key, value in report['selected'].items()},
                     'missing_views': report.get('missing_views'), 'performance': report.get('performance'),
                     'real_time': {key: report[key] for key in ('processing_fps', 'capture_to_result_ms_p50',
                                  'capture_to_result_ms_p95', 'capture_to_result_ms_max', 'source_frames', 'frames_processed',
                                  'source_frames_skipped', 'frames_replaced_before_processing', 'queue_wait_ms_p95') if key in report},
                     'groups': groups})
    result = {'runs': rows, 'notes': ['Models and policies are unchanged between backends; detector float conversion is disclosed in the export manifest.',
                                   'Offline FPS includes decode, processing and JSONL writing; previews disabled.',
                                   'Real-time latency starts at decoded-frame submission, excludes sensor/decode/display and startup.',
                                   'Neural and accepted-frame statistics prevent background rejection from hiding expensive frames.']}
    (output / 'comparison.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--reuse-existing', action='store_true')
    p.add_argument('--summarize-only', action='store_true')
    args = p.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        summarize(output)
        return
    tools = ROOT / 'src/Pet-ReID-IMAG/tools'
    for backend in ('cpu', 'gpu', 'trt-direct', 'trt'):
        dest = output / 'micro' / backend
        if not (args.reuse_existing and (dest / 'microbenchmark.json').exists()):
            cmd = [tools / 'benchmark_dog_trt.py', '--backend', backend, '--output', dest]
            if backend != 'cpu':
                cmd += ['--reference', output / 'micro/cpu']
            run(cmd, output / f'micro-{backend}.log')
    run([tools / 'validate_dog_capture.py', '--mode', 'views', '--device', 'trt', '--include-fixtures', '--no-preview',
         '--output', output / 'validation', *(['--reuse-existing'] if args.reuse_existing else [])], output / 'validation.log')
    videos = {'user': ROOT / 'data/dog_capture/user/2611101816341481.mp4',
              'wolfhound': ROOT / 'data/dog_capture/public/wolfhound_panting.ogv'}
    # Alternate backend order on the second input; inference jobs never overlap.
    for name, video in videos.items():
        backends = ('cpu', 'gpu', 'trt', 'trt-direct') if name == 'user' else ('trt-direct', 'trt', 'gpu', 'cpu')
        for backend in backends:
            dest = output / 'comparison' / f'{name}-{backend}'
            if args.reuse_existing and (dest / 'result.json').exists():
                continue
            device = 'trt' if backend.startswith('trt') else backend
            cmd = [tools / 'select_dog_views.py', video, '--device', device, '--no-preview', '--output', dest]
            if backend == 'trt-direct':
                cmd.append('--no-cuda-graph')
            run(cmd, output / f'{name}-{backend}.log')
    for backend in ('cpu', 'gpu', 'trt'):
        dest = output / 'comparison' / f'live-user-{backend}'
        if args.reuse_existing and (dest / 'result.json').exists():
            continue
        run([tools / 'live_dog_views.py', '--video', videos['user'], '--device', backend, '--output', dest],
            output / f'live-user-{backend}.log')
    summarize(output)


if __name__ == '__main__':
    main()
