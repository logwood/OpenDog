"""Causal per-frame API shared by file validation and real-time consumers."""
import time
import math

from .geometry import Pose, TurnTracker
from .quality import BestFrames, Quality, assess_quality


def snapshot_rank(snapshot):
    if not snapshot:
        return (-1, -1, -1)
    frames = snapshot['frames']
    return (int(snapshot['complete']), int('front' in frames),
            frames.get('front', {}).get('record', {}).get('quality', {}).get('score', -1))


class CapturePipeline:
    """Call from one worker only, with BGR pixels and monotonic capture times.

    The returned record is JSON serializable. Best-frame images remain available
    in snapshot(), in original resolution. No disk or network I/O in process().
    """
    def __init__(self, models, hold=.25, side_threshold=.25, min_face_pixels=100,
                 min_laplacian=12, turn_order=('left', 'right')):
        self.models = models
        self.tracker = TurnTracker(hold_seconds=hold, side_threshold=side_threshold,
                                   turn_order=turn_order)
        self.selector = BestFrames()
        self.previous_segment = 0
        self.archived = None
        self.completed = set()
        self.min_face_pixels = min_face_pixels
        self.min_laplacian = min_laplacian

    def process(self, frame, timestamp, frame_index):
        previous = self.tracker.previous_time
        if not math.isfinite(timestamp) or (previous is not None and timestamp <= previous):
            raise ValueError('Capture times must be finite and strictly increasing')
        started = time.perf_counter()
        detected = self.models.infer(frame, timestamp)
        if detected['valid']:
            pose = detected['pose']
            quality = assess_quality(frame, detected['points'], detected['box'], pose,
                                     self.min_face_pixels, self.min_laplacian)
        else:
            pose = Pose(reason=detected['reason'])
            quality = Quality(False, [detected['reason']])
        tracking = self.tracker.update(timestamp, pose, quality.accepted,
                                       detected.get('discontinuity', False))
        if tracking['segment'] != self.previous_segment:
            snapshot = {'segment': self.previous_segment, 'frames': self.selector.frames.copy(),
                        'complete': self.previous_segment in self.completed}
            if snapshot_rank(snapshot) > snapshot_rank(self.archived):
                self.archived = snapshot
            self.previous_segment = tracking['segment']
            self.completed.clear()
        if tracking['sequence_complete']:
            self.completed.add(tracking['segment'])
        record = {'frame_index': int(frame_index), 'time_s': float(timestamp),
                  'pose': pose.to_dict(), 'quality': quality.to_dict(), 'tracking': tracking,
                  'timings_ms': detected['timings_ms'], 'dog_score': detected.get('dog_score'),
                  'inference_mode': detected.get('inference_mode', 'no_face'),
                  'rotation_quarters': detected.get('rotation_quarters', 0),
                  'uncertainty': detected.get('uncertainty'),
                  'box': list(detected['box']) if detected['valid'] else None,
                  'landmarks': detected['points'].tolist() if detected['valid'] else None}
        self.selector.consider(frame, record)
        record['compute_ms'] = (time.perf_counter() - started) * 1000
        return record

    def snapshot(self):
        current = {'segment': self.tracker.segment, 'frames': self.selector.frames.copy(),
                   'complete': self.tracker.segment in self.completed}
        return self.archived if snapshot_rank(self.archived) > snapshot_rank(current) else current
