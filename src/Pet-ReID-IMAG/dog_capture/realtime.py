"""One worker, one waiting frame: slow inference never builds a video backlog."""
from __future__ import annotations

import argparse
from collections import Counter, deque
import json
from pathlib import Path
import queue
import sys
import threading
import time

import cv2
import numpy as np

from .cli import ROOT, draw_overlay, save_image
from .models import DogFaceModels, ensure_models, file_hash
from .pipeline import CapturePipeline
from .runtime import peak_rss_mb


class LatestFrameWorker:
    def __init__(self, pipeline, on_record=None):
        self.pipeline = pipeline
        self.pending = queue.Queue(maxsize=1)
        self.ready = queue.Queue(maxsize=1)
        self.stopping = threading.Event()
        self.error = None
        self.submitted = self.dropped = self.processed = 0
        self.records = deque(maxlen=4096)
        self.on_record = on_record
        self.mode_counts, self.state_counts = Counter(), Counter()
        self.thread = threading.Thread(target=self._run, name='dog-capture-inference', daemon=True)
        self.thread.start()

    def submit(self, frame, timestamp, frame_index, observation=None):
        if self.stopping.is_set():
            raise RuntimeError('Worker is closed')
        if self.error:
            raise RuntimeError('Inference worker failed') from self.error
        item = (frame.copy(), timestamp, frame_index, time.perf_counter(), observation)
        try:
            self.pending.get_nowait()
            self.dropped += 1
        except queue.Empty:
            pass
        self.pending.put_nowait(item)
        self.submitted += 1

    def _run(self):
        try:
            while not self.stopping.is_set() or not self.pending.empty():
                try:
                    frame, timestamp, index, captured, observation = self.pending.get(timeout=.05)
                except queue.Empty:
                    continue
                started = time.perf_counter()
                record = (self.pipeline.process(frame, timestamp, index, observation=observation)
                          if observation is not None else self.pipeline.process(frame, timestamp, index))
                record['queue_wait_ms'] = (started - captured) * 1000
                record['capture_to_result_ms'] = (time.perf_counter() - captured) * 1000
                self.processed += 1
                self.records.append(record)
                self.mode_counts[record['inference_mode']] += 1
                self.state_counts[record['tracking']['stable_state']] += 1
                if self.on_record:
                    self.on_record(record)
                try:
                    self.ready.get_nowait()
                except queue.Empty:
                    pass
                self.ready.put_nowait((frame, record, time.perf_counter()))
        except Exception as exc:
            self.error = exc
            self.stopping.set()

    def poll(self):
        if self.error:
            raise RuntimeError('Inference worker failed') from self.error
        try:
            return self.ready.get_nowait()
        except queue.Empty:
            return None

    def close(self):
        self.stopping.set()
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise TimeoutError('Inference worker did not stop')
        if self.error:
            raise RuntimeError('Inference worker failed') from self.error


def main(default_mode='challenge'):
    parser = argparse.ArgumentParser(description='Real-time dog capture, latest-frame queue, optional live window')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--video', help='Replay a file at its original frame rate')
    source.add_argument('--camera', type=int, help='Camera device index')
    parser.add_argument('--mode', choices=['challenge', 'views'], default=default_mode)
    parser.add_argument('--target-config')
    parser.add_argument('--output', required=True)
    parser.add_argument('--models', default=str(ROOT / 'models/pretrained/dog_face_landmarks_v1'))
    parser.add_argument('--download-models', action='store_true')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', choices=['auto', 'cpu', 'gpu', 'trt'], default='auto')
    parser.add_argument('--trt-engines')
    parser.add_argument('--no-cuda-graph', action='store_true')
    parser.add_argument('--max-seconds', type=float, default=0)
    parser.add_argument('--show', action='store_true', help='Needs opencv-python (GUI variant); Esc stops')
    parser.add_argument('--mirror', action='store_true')
    args = parser.parse_args()
    if args.threads < 1 or args.max_seconds < 0:
        parser.error('Invalid runtime parameters')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('Use a new output directory')
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        raise RuntimeError('Cannot open video source')
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if args.video and (not np.isfinite(fps) or fps <= 0):
        raise ValueError('Video must declare a valid frame rate')
    views_mode = args.mode == 'views'
    model_dir = Path(args.models) if args.device == 'trt' else ensure_models(args.models, args.download_models)
    models = DogFaceModels(model_dir, args.threads, device=args.device, compact_confidence=views_mode,
                           trt_directory=args.trt_engines, trt_cuda_graph=not args.no_cuda_graph)
    if views_mode:
        from .views import TargetViewPipeline
        from .view_io import load_targets
        pipeline = TargetViewPipeline(models, targets=load_targets(args.target_config))
    else:
        pipeline = CapturePipeline(models)
    log = (output / 'frames.jsonl').open('w', encoding='utf8')
    worker = LatestFrameWorker(pipeline, lambda record: log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n'))
    index, source_skips, last, displays = 0, 0, None, 0
    started = time.perf_counter()
    try:
        while True:
            target = index / fps if args.video else time.perf_counter() - started
            if args.max_seconds and target >= args.max_seconds:
                break
            if args.video:
                # Pace capture independently of the inference worker.
                delay = target - (time.perf_counter() - started)
                if delay > 0:
                    time.sleep(delay)
                elif delay < -1 / fps:
                    if not cap.grab():
                        break
                    index += 1
                    source_skips += 1
                    continue
            ok, frame = cap.read()
            if not ok:
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)
            timestamp = index / fps if args.video else time.perf_counter() - started
            worker.submit(frame, timestamp, index)
            received = worker.poll()
            if received:
                last = received
            if args.show:
                # Video stays current. Old coordinates are never painted on a
                # different frame; status expires 250 ms after source capture.
                if last and timestamp - last[1]['time_s'] < .25:
                    shown = draw_overlay(frame, {**last[1], 'box': None, 'landmarks': None})
                else:
                    shown = cv2.resize(frame, (round(frame.shape[1] * min(1, 960 / max(frame.shape[:2]))),
                                              round(frame.shape[0] * min(1, 960 / max(frame.shape[:2])))))
                    cv2.putText(shown, 'ACQUIRING FACE', (12, 30), 0, .7, (0, 230, 255), 2)
                cv2.imshow('Dog Capture - Esc to stop', shown)
                displays += 1
                if cv2.pollKey() & 255 == 27:
                    break
            index += 1
    finally:
        cap.release()
        try:
            worker.close()
        finally:
            log.close()
            if sys.exc_info()[0] is not None:
                models.close()
        if args.show:
            cv2.destroyAllWindows()
    elapsed = time.perf_counter() - started
    if not worker.records:
        raise ValueError('No video frames processed')
    if views_mode:
        try:
            pipeline.finalize()
        finally:
            models.close()
        elapsed = time.perf_counter() - started
    else:
        models.close()
    selected = {}
    snapshot = pipeline.snapshot()
    for state, item in snapshot['frames'].items():
        record, frame = item['record'], item['image']
        save_image(output / f'best_{state}_frame.png', frame)
        x1, y1, x2, y2 = record['box']
        save_image(output / f'best_{state}_face.png', np.rot90(frame[y1:y2, x1:x2], record['rotation_quarters']))
        selected[state] = {'frame_index': record['frame_index'], 'time_s': record['time_s'],
                           'selection_basis': record.get('selection_basis', 'confirmed_pose_dwell'),
                           'box_xyxy': record['box'], 'rotation_quarters': record['rotation_quarters'],
                           'quality': record['quality'], 'pose': record['pose'],
                           'frame_file': f'best_{state}_frame.png', 'face_file': f'best_{state}_face.png'}
    latency = [r['capture_to_result_ms'] for r in worker.records]
    wait = [r['queue_wait_ms'] for r in worker.records]
    report = {'input': args.video if args.video else f'camera:{args.camera}',
              'input_sha256': file_hash(args.video) if args.video else None,
              'source_fps': fps, 'elapsed_seconds': elapsed, 'source_frames': index,
              'frames_submitted': worker.submitted, 'frames_processed': worker.processed,
              'frames_replaced_before_processing': worker.dropped, 'source_frames_skipped': source_skips,
              'processing_fps': worker.processed / elapsed,
              'peak_rss_mb': peak_rss_mb(),
              'device': pipeline.models.device,
              'capture_to_result_ms_p50': float(np.median(latency)),
              'capture_to_result_ms_p95': float(np.percentile(latency, 95)),
              'capture_to_result_ms_max': max(latency),
              'queue_wait_ms_p95': float(np.percentile(wait, 95)),
              'mode': args.mode, 'queue_capacity': 1, 'sequence_complete': snapshot['complete'] if not views_mode else False, 'selected': selected,
              'recognition_input': {'image_path': str(output / selected['front']['frame_file']),
                                    'face_path': str(output / selected['front']['face_file']),
                                    'sequence_complete': snapshot['complete'] if not views_mode else False} if 'front' in selected else None,
              'latency_window_frames': len(worker.records),
              'inference_mode_counts': dict(worker.mode_counts),
              'stable_state_counts': dict(worker.state_counts),
              'events': list(pipeline.events if views_mode else pipeline.tracker.events), 'display_updates': displays,
              'notes': ['Capture-to-result starts when a decoded frame is submitted; camera sensor/driver latency is not included.',
                        'Skipped frames are explicit. This benchmark does not claim every input frame received inference.',
                        'Latency percentiles use the most recent 4096 results; per-frame JSONL contains the complete run.']}
    report['model_calls'] = dict(models.calls)
    report['runtime_assets'] = models.runtime_assets
    if views_mode:
        report.update(coverage_complete=snapshot['complete'], view_configuration=pipeline.configuration(),
                      missing_views=[t.name for t in pipeline.targets if t.name not in selected],
                      candidate_verifications=pipeline.verifications,
                      candidate_verification_rejects=pipeline.verification_rejects)
        for name, item in snapshot['frames'].items():
            selected[name]['verification'] = item['record']['verification']
    (output / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
