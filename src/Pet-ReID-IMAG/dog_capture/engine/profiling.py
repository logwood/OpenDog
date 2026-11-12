"""Opt-in, nested wall-clock tracing for an isolated capture diagnostic process.

No hooks are installed by the normal engine. Use one profiler at a time in a
dedicated process: module-level image helpers are temporarily wrapped, while
only calls inside the selected engine's process() are recorded. Parent self
time excludes children, so the component breakdown sums to the frame time.
ORT run time includes its runtime, synchronization and any device transfers.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import wraps
import threading
import time

import numpy as np


def quantiles(values):
    values = list(values)
    if not values:
        return {'mean': 0.0, 'p50': 0.0, 'p95': 0.0, 'p99': 0.0, 'max': 0.0}
    return dict(zip(('mean', 'p50', 'p95', 'p99', 'max'),
                    map(float, (np.mean(values), *np.percentile(values, [50, 95, 99]), max(values)))))


def component(path):
    """Exclusive buckets, never inclusive parent/child totals added together."""
    name = path[-1]
    if name.startswith('network.'):
        return name
    if name in ('image.resize', 'image.warp', 'image.pack', 'image.color', 'image.rotate'):
        return name
    if name.startswith('quality.') or name in ('image.gaussian', 'image.laplacian', 'image.blur'):
        return 'quality'
    if name.startswith('geometry.') or name.startswith('pose.'):
        return 'pose_geometry'
    if name.startswith('tracking.'):
        return 'tracking'
    if name.startswith('model.'):
        return 'model_adapters'
    if name.startswith('selection.'):
        return 'selection'
    if name.startswith('refine.'):
        return 'roi_control'
    return 'capture_control'


class CaptureProfiler:
    def __init__(self, engine, *, clock=time.perf_counter_ns):
        self.engine, self.clock = engine, clock
        self.local = threading.local()
        self.frames, self.paths, self._path_ids, self._patches = [], [], {}, []

    def _patch(self, obj, attribute, label, *, root=False):
        if not hasattr(obj, attribute):
            return
        original = getattr(obj, attribute)
        owned = attribute in vars(obj)

        @wraps(original)
        def measured(*args, **kwargs):
            stack = getattr(self.local, 'stack', None)
            if not root and not stack:
                return original(*args, **kwargs)
            if root:
                if stack:
                    raise RuntimeError('Recursive capture profiling is unsupported')
                stack = self.local.stack = []
                self.local.spans = []
                self.local.start = self.clock()
            path = (*stack[-1][0], label) if stack else (label,)
            if path not in self._path_ids:
                self._path_ids[path] = len(self.paths)
                self.paths.append(path)
            entry = [path, self.clock(), 0]
            stack.append(entry)
            try:
                return original(*args, **kwargs)
            finally:
                elapsed = self.clock() - entry[1]
                stack.pop()
                if stack:
                    stack[-1][2] += elapsed
                self.local.spans.append([self._path_ids[path], (entry[1] - self.local.start) / 1e6,
                                         elapsed / 1e6, (elapsed - entry[2]) / 1e6])
                if root:
                    index = args[2] if len(args) > 2 else kwargs.get('frame_index')
                    timestamp = args[1] if len(args) > 1 else kwargs.get('timestamp')
                    self.frames.append({'frame_index': index, 'time_s': timestamp,
                                        'trace_origin_ns': self.local.start,
                                        'total_ms': elapsed / 1e6, 'spans': self.local.spans})
                    del self.local.spans

        self._patches.append((obj, attribute, original, owned))
        setattr(obj, attribute, measured)

    def __enter__(self):
        import cv2
        from . import alignment, detectors, engine as engine_module, pose, quality, refine, selection
        try:
            self._patch(self.engine, 'process', 'capture', root=True)
            for method in ('_roi_zoom', '_refresh_bodies', '_check_dog', '_track_is_dog', '_probe',
                           '_process_still', '_analyse', '_sharpen', '_unreliable', '_trusted', '_near_miss', '_result'):
                self._patch(self.engine, method, 'capture.' + method.lstrip('_'))
            for method in ('next', 'report'):
                self._patch(self.engine.orientation, method, 'capture.orientation_' + method)
            self._patch(self.engine.tracker, 'update', 'tracking.update')
            for method in ('assess', 'align'):
                self._patch(self.engine.assessor, method, 'quality.' + method)
            refiner = self.engine.refiner
            if refiner is not None:
                for method in ('refine', 'crop', 'examine', '_look', '_search', '_settle', 'precise', 'verify', '_mirror_points'):
                    self._patch(refiner, method, 'refine.' + method.lstrip('_'))
            for name, model in (('face', self.engine.detector), ('body', self.engine.body_gate),
                                ('landmarks', getattr(refiner, 'landmarker', None))):
                if model is None:
                    continue
                for method in ('detect', 'preprocess', '_finish'):
                    self._patch(model, method, 'model.' + name + '.' + method.lstrip('_'))
                self._patch(model, '_run', 'network.' + name)
            for module, function, label in (
                (detectors, 'letterbox', 'model.letterbox'), (detectors, '_rgb_tensor', 'image.pack'),
                (detectors, 'resize_area', 'image.resize'), (refine, 'resize_area', 'image.resize'),
                (alignment, 'resize_area', 'image.resize'),
                (detectors, 'nms', 'model.nms'), (engine_module, 'as_bgr', 'image.input'),
                (engine_module, 'rotate_frame', 'image.rotate'), (engine_module, 'face_geometry', 'pose.geometry'),
                (engine_module, 'eye_contrast', 'quality.eye_contrast'),
                (refine, 'warp_crop', 'refine.warp_crop'), (refine, 'crop_matrix', 'geometry.crop_matrix'),
                (refine, 'apply', 'geometry.map_points'), (refine, 'eye_line_deg', 'geometry.eye_line'),
                (alignment, 'umeyama_similarity', 'geometry.similarity'),
                (quality, 'align_face', 'quality.align_face'), (engine_module, 'align_face', 'quality.align_face'),
                (quality, 'blur_effect', 'quality.reblur'), (quality, '_laplacian_energy', 'quality.laplacian'),
                (np.linalg, 'inv', 'geometry.inverse'), (np.linalg, 'svd', 'geometry.svd'),
                (cv2, 'resize', 'image.resize'), (cv2, 'warpAffine', 'image.warp'),
                (cv2, 'cvtColor', 'image.color'), (cv2, 'rotate', 'image.rotate'),
                (cv2, 'GaussianBlur', 'image.gaussian'), (cv2, 'Laplacian', 'image.laplacian'),
                (cv2, 'blur', 'image.blur'), (pose.ViewClassifier, 'update', 'pose.temporal'),
            ):
                self._patch(module, function, label)
            for method in ('observe', 'wants', 'offer'):
                self._patch(selection.TrackShots, method, 'selection.' + method)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        for obj, attribute, original, owned in reversed(self._patches):
            if owned:
                setattr(obj, attribute, original)
            else:
                delattr(obj, attribute)
        self._patches.clear()

    def summary(self):
        n = len(self.frames)
        if not n:
            raise ValueError('No frames were profiled')
        totals = [f['total_ms'] for f in self.frames]
        cutoff = float(np.percentile(totals, 95))
        slow = [f for f in self.frames if f['total_ms'] >= cutoff]

        def summarize(frames):
            inclusive, exclusive, calls = defaultdict(list), Counter(), Counter()
            components, per_frame = Counter(), defaultdict(list)
            for frame in frames:
                sums = Counter()
                self_sum = 0.0
                for key, offset, duration, own in frame['spans']:
                    path = self.paths[key]
                    inclusive[key].append(duration)
                    exclusive[key] += own
                    calls[key] += 1
                    bucket = component(path)
                    components[bucket] += own
                    sums[bucket] += own
                    self_sum += own
                if abs(self_sum - frame['total_ms']) > 1e-6:
                    raise AssertionError('Nested timing does not reconcile')
                for bucket in {component(path) for path in self.paths}:
                    per_frame[bucket].append(sums[bucket])
            full = sum(frame['total_ms'] for frame in frames)
            return {
                'frames': len(frames), 'total_ms': full,
                'components': sorted([{'name': name, 'total_ms': value, 'share': value / full,
                                      'frame_ms': quantiles(per_frame[name])}
                                     for name, value in components.items()], key=lambda r: -r['total_ms']),
                'paths': sorted([{'id': key, 'path': list(self.paths[key]), 'calls': calls[key],
                                  'inclusive_ms': sum(values), 'exclusive_ms': exclusive[key],
                                  'inclusive_ms_per_frame': sum(values) / len(frames),
                                  'exclusive_ms_per_frame': exclusive[key] / len(frames),
                                  'call_ms': quantiles(values)} for key, values in inclusive.items()],
                                key=lambda r: -r['exclusive_ms']),
            }

        reasons = defaultdict(Counter)
        for frame in self.frames:
            for key, _, duration, _ in frame['spans']:
                path = self.paths[key]
                if path[-1] != 'network.face':
                    continue
                reason = ('mirror' if 'refine.mirror_points' in path else
                          'verification' if 'refine.verify' in path else
                          'precise' if 'refine.precise' in path else
                          'roi_zoom' if 'capture.roi_zoom' in path else
                          'refinement' if 'refine.refine' in path else 'full_frame')
                reasons[reason]['calls'] += 1
                reasons[reason]['total_ms'] += duration
        return {'frames': n, 'processing_ms': quantiles(totals), 'slow_frame_threshold_ms': cutoff,
                'all': summarize(self.frames), 'slowest_5_percent': summarize(slow),
                'face_call_reasons': dict(reasons),
                'notes': ['Wall-clock profiling includes instrumentation overhead and OS scheduling.',
                          'Only exclusive components add up. Inclusive path totals overlap their descendants.',
                          'ORT calls include runtime, synchronization and device transfer, not just arithmetic.',
                          'Producer decode can overlap worker processing; do not add its mean to frame latency.']}

    def chrome_trace(self, origin_ns=None):
        origin_ns = origin_ns if origin_ns is not None else self.frames[0]['trace_origin_ns']
        events = []
        for frame in self.frames:
            for key, offset, duration, own in frame['spans']:
                path = self.paths[key]
                events.append({'name': path[-1], 'cat': component(path), 'ph': 'X', 'pid': 1,
                               'tid': 1, 'ts': (frame['trace_origin_ns'] - origin_ns) / 1000 + offset * 1000,
                               'dur': duration * 1000, 'args': {'frame_index': frame['frame_index'],
                                                              'self_ms': own, 'path': '/'.join(path)}})
        return {'traceEvents': events, 'displayTimeUnit': 'ms',
                'otherData': {'time_axis': 'measured monotonic wall time, microseconds from origin',
                              'origin_ns': origin_ns}}
