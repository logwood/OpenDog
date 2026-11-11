"""Repeat complete, paced videos under the dedicated NVIDIA application profile."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dog_capture.cli import ROOT
from configure_dog_trt_windows import configure


def summarize(directory):
    report = json.loads((directory / 'result.json').read_text(encoding='utf8'))
    frames = [json.loads(line) for line in (directory / 'frames.jsonl').read_text(encoding='utf8').splitlines()]
    groups = {}
    for name, records in [('all', frames), ('neural', [f for f in frames if f['inference_mode'].startswith('neural')]),
                          ('accepted', [f for f in frames if f['quality']['accepted']])]:
        if records:
            values = [f['capture_to_result_ms'] for f in records]
            groups[name] = dict(n=len(values), p50_ms=float(np.median(values)),
                                p95_ms=float(np.percentile(values, 95)), max_ms=max(values))
    keys = ('source_frames', 'frames_processed', 'source_frames_skipped', 'frames_replaced_before_processing',
            'processing_fps', 'capture_to_result_ms_p50', 'capture_to_result_ms_p95', 'capture_to_result_ms_max',
            'queue_wait_ms_p95', 'queue_capacity', 'peak_rss_mb', 'model_calls', 'runtime_assets')
    return dict(run=directory.name, **{k: report[k] for k in keys}, groups=groups,
                selected={k: v['frame_index'] for k, v in report['selected'].items()},
                input_sha256=report['input_sha256'], missing_views=report['missing_views'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--controls', action='store_true', help='Also run unprofiled python.exe for each user-video repeat')
    parser.add_argument('--queue-capacity', type=int, choices=range(1, 5), default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('Use an empty output directory')
    tools = ROOT / 'src/Pet-ReID-IMAG/tools'
    runtime = ROOT / '.tmp/dog-trt-env/Scripts'
    dedicated, generic = runtime / 'dog_capture_trt.exe', runtime / 'python.exe'
    profile = configure(dedicated, 'status')
    if not profile['profile_exists'] or profile['preferred_pstate'] != 1:
        raise RuntimeError('Enable the dedicated application performance preference before this validation')
    plan = []
    for repeat in range(1, args.repeats + 1):
        modes = ['dedicated', 'generic-control'] if repeat % 2 else ['generic-control', 'dedicated']
        for mode in modes:
            if mode == 'generic-control' and not args.controls:
                continue
            plan.append((f'live-user-{mode}-{repeat}', mode, ROOT / 'data/dog_capture/user/2611101816341481.mp4'))
    for repeat in range(1, args.repeats + 1):
        plan.append((f'live-wolfhound-dedicated-{repeat}', 'dedicated', ROOT / 'data/dog_capture/public/wolfhound_panting.ogv'))
    evidence = dict(started_utc=datetime.now(timezone.utc).isoformat(), profile_before=profile, runs=[],
                    notes=['Complete videos paced at original frame rate, no preview window; jobs run sequentially.',
                           'Dedicated and generic controls use identical models, thresholds, DLL bootstrap and environment repair.',
                           'Only dog_capture_trt.exe matches the saved NVIDIA preference; the control is ordinary python.exe.',
                           'Latency starts at decoded-frame submission and excludes sensor, decoding, display and model startup.',
                           'This desktop measurement includes other desktop applications; no system-wide settings are changed.'])
    def save():
        (output / 'summary.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf8')
    save()
    for name, mode, video in plan:
        print('Running', name, flush=True)
        executable = dedicated if mode == 'dedicated' else generic
        command = [str(executable), str(tools / 'bootstrap_dog_trt.py'), str(tools / 'live_dog_views.py'),
                   '--video', str(video), '--device', 'trt', '--queue-capacity', str(args.queue_capacity), '--output', str(output / name)]
        started = time.perf_counter()
        with (output / f'{name}.log').open('w', encoding='utf8') as log:
            subprocess.run(command, stdout=log, stderr=log, check=True, cwd=ROOT)
        row = summarize(output / name)
        row.update(executable=str(executable), wall_seconds_with_startup=time.perf_counter() - started)
        evidence['runs'].append(row)
        save()
        print(json.dumps({k: row[k] for k in ('run', 'frames_processed', 'capture_to_result_ms_p95', 'selected')}), flush=True)
    evidence['profile_after'] = configure(dedicated, 'status')
    if evidence['profile_after'] != profile:
        raise RuntimeError('Driver profile changed during validation')
    evidence['finished_utc'] = datetime.now(timezone.utc).isoformat()
    save()


if __name__ == '__main__':
    main()
