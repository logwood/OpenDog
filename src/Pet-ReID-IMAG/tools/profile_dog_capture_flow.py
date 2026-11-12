"""Profile complete capture data flow without changing decisions or deployed models.

Produces exclusive component totals, hierarchical paths, individual frame spans,
Chrome trace JSON and static timing plots. UI, network and ReID are not exercised.
Use uninstrumented real-time benchmarks for final performance acceptance.
"""
from __future__ import annotations

import argparse
import gc
import gzip
import json
from pathlib import Path
import sys
import time

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))


def plot_profile(profile, output, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    all_rows = profile['all']['components']
    slow = {r['name']: r for r in profile['slowest_5_percent']['components']}
    names = [r['name'] for r in all_rows]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ax, values, heading in ((axes[0], [r['frame_ms']['mean'] for r in all_rows], 'All frames'),
                               (axes[1], [slow.get(k, {}).get('frame_ms', {}).get('mean', 0) for k in names],
                                'Slowest 5% of frames')):
        ax.barh(names, values, color='#2678ac')
        ax.invert_yaxis()
        ax.set_xlabel('Exclusive mean milliseconds per processed frame')
        ax.set_title(heading)
        ax.spines[['top', 'right']].set_visible(False)
        for i, value in enumerate(values):
            ax.text(value, i, f' {value:.3f}', va='center', fontsize=9)
        ax.set_xlim(0, max(values) * 1.2)
    fig.suptitle(title + ' | CPU wall clock, instrumented')
    fig.savefig(output / 'components.png', dpi=150)
    fig.savefig(output / 'components.svg')
    plt.close(fig)
    paths = {tuple(row['path']): row for row in profile['all']['paths']}
    fig, ax = plt.subplots(figsize=(17, 7), constrained_layout=True)
    def draw(path, x):
        row = paths[path]
        depth = len(path) - 1
        width = row['inclusive_ms_per_frame']
        color = '#e8a34b' if path[-1].startswith('network.') else '#8bbdd7' if path[-1].startswith('image.') else '#d9e1e6'
        rectangle = Rectangle((x, depth), width, .9, facecolor=color, edgecolor='none')
        ax.add_patch(rectangle)
        capacity = int(width / profile['processing_ms']['mean'] * 280)
        if capacity >= 6:
            name = path[-1].replace('capture.', '').replace('model.', '').replace('network.', 'infer.')
            if len(name) > capacity:
                name = name[:capacity - 1] + '…'
            text = ax.text(x + width / 2, depth + .45, f'{name}\n{width:.3f}',
                           ha='center', va='center', fontsize=7, clip_on=True)
            text.set_clip_path(rectangle)
        children = sorted((p for p in paths if len(p) == len(path) + 1 and p[:-1] == path),
                          key=lambda p: paths[p]['id'])
        for child in children:
            draw(child, x)
            x += paths[child]['inclusive_ms_per_frame']
    draw(('capture',), 0)
    ax.set_xlim(0, profile['processing_ms']['mean'])
    ax.set_ylim(max(map(len, paths)) + .5, -.5)
    ax.set_yticks([])
    ax.set_xlabel('Inclusive mean ms per frame; children partition each parent, remaining width = self time')
    ax.set_title(title + ' | Complete call tree (aggregated, not chronological)')
    fig.savefig(output / 'flow.png', dpi=180)
    fig.savefig(output / 'flow.svg')
    plt.close(fig)


def replay(config, path, output, realtime, warmup, qos='inherit'):
    from dog_capture.engine.process_policy import cpu_qos
    with cpu_qos(qos, realtime=realtime) as policy:
        return _replay(config, path, output, realtime, warmup, policy)


def _replay(config, path, output, realtime, warmup, cpu_policy):
    import cv2
    from dog_capture.engine.io import LatestFrameRunner, RunWriter, VideoSource
    from dog_capture.engine.profiling import CaptureProfiler, quantiles
    from dog_capture.engine.zoo import build_engine
    output.mkdir(parents=True)
    engine = build_engine(config)
    for model in (engine.detector, engine.body_gate, getattr(engine.refiner, 'landmarker', None)):
        if model is not None:
            model.warmup(warmup)
    source = VideoSource(path)
    declared = int(source.capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames, decode, lateness, queue, end_to_end, producer_events = [], [], [], [], [], []
    with CaptureProfiler(engine) as profiler:
        origin_ns = time.perf_counter_ns()
        started = time.perf_counter()
        def on_result(result, frame):
            frames.append(result.frame_index)
            if realtime:
                lateness_value = (time.perf_counter() - started - result.time_s) * 1000
                end_to_end.append(lateness_value)
                queue.append(result.timings_ms['queue_wait'])
        runner = LatestFrameRunner(engine, on_result) if realtime else None
        count = 0
        try:
            iterator = iter(source)
            while True:
                before_ns = time.perf_counter_ns()
                before = time.perf_counter()
                try:
                    frame, ts, index = next(iterator)
                except StopIteration:
                    break
                decode.append((time.perf_counter() - before) * 1000)
                producer_events.append({'name': 'source.decode', 'cat': 'producer', 'ph': 'X', 'pid': 1, 'tid': 2,
                                        'ts': (before_ns - origin_ns) / 1000,
                                        'dur': (time.perf_counter_ns() - before_ns) / 1000,
                                        'args': {'frame_index': index}})
                if runner:
                    delay = ts - (time.perf_counter() - started)
                    if delay > 0:
                        time.sleep(delay)
                    lateness.append(max(0, (time.perf_counter() - started - ts) * 1000))
                    runner.submit(frame, ts, index)
                else:
                    result = engine.process(frame, ts, index)
                    on_result(result, frame)
                count += 1
                if count % 400 == 0:
                    print(path.name, 'realtime' if realtime else 'offline', count, flush=True)
        finally:
            if runner:
                runner.close()
            source.close()
    elapsed = time.perf_counter() - started
    if declared - count not in (0, 1):
        raise AssertionError(f'Incomplete decode: {count}/{declared}')
    if count != len(frames) + (runner.dropped if runner else 0) + sum(engine.dropped.values()):
        raise AssertionError('Unaccounted frames')
    report = profiler.summary()
    report.update(source=str(path), playback='realtime' if realtime else 'offline', cpu_policy=cpu_policy,
                  frames_read=count, frames_dropped=runner.dropped if runner else 0,
                  fps=len(frames) / elapsed, decode_ms=quantiles(decode),
                  producer_lateness_ms=quantiles(lateness), queue_wait_ms=quantiles(queue),
                  scheduled_to_result_ms=quantiles(end_to_end), models=engine.describe(),
                  landmarks=engine.refiner.landmarker.describe() if getattr(engine.refiner, 'landmarker', None) else None)
    writer = RunWriter(output, log_frames=False)
    report['selection'] = writer.save_bundle(engine.finalize())
    writer.close()
    (output / 'profile.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    with gzip.open(output / 'frames.json.gz', 'wt', encoding='utf8') as stream:
        json.dump({'paths': profiler.paths, 'frames': profiler.frames}, stream)
    with gzip.open(output / 'trace.json.gz', 'wt', encoding='utf8') as stream:
        trace = profiler.chrome_trace(origin_ns)
        trace['traceEvents'].extend(producer_events)
        json.dump(trace, stream)
    plot_profile(report, output, path.stem + ' / ' + report['playback'])
    print(json.dumps({'source': path.name, 'mode': report['playback'], 'fps': report['fps'],
                      'dropped': report['frames_dropped'], 'processing': report['processing_ms'],
                      'components': [(r['name'], round(r['share'] * 100, 2), round(r['frame_ms']['mean'], 3))
                                     for r in report['all']['components']],
                      'face_calls': report['face_call_reasons']}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, action='append', required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--playback', choices=('offline', 'realtime', 'both'), default='both')
    parser.add_argument('--cv-threads', type=int, default=1)
    parser.add_argument('--cpu-qos', choices=('auto', 'inherit', 'high'), default='inherit')
    parser.add_argument('--warmup', type=int, default=15)
    args = parser.parse_args()
    if args.output.exists() or args.cv_threads < 1 or args.warmup < 1:
        parser.error('Use a fresh output directory and positive thread/warmup values')
    import cv2
    from dog_capture.engine.config import CaptureConfig
    cv2.setNumThreads(args.cv_threads)
    config = CaptureConfig.load(args.config)
    args.output.mkdir(parents=True)
    summary = {'config': config.to_dict(), 'cv_threads': cv2.getNumThreads(), 'videos': []}
    for path in args.source:
        for mode in (('offline', 'realtime') if args.playback == 'both' else (args.playback,)):
            row = replay(config, path, args.output / (path.stem + '-' + mode), mode == 'realtime', args.warmup, args.cpu_qos)
            summary['videos'].append(row)
            (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
            gc.collect()


if __name__ == '__main__':
    main()
