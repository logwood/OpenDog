"""Targeted, unordered view capture: three original photographs, no challenge.

The hot path can consume existing model outputs with zero neural inference.
A standalone DogFaceModels provider is optional for raw-video experiments.
"""
from collections import deque
from dataclasses import asdict, dataclass
import math
import time

import cv2
import numpy as np

from .geometry import EYE_A, EYE_B, Pose
from .observations import FaceObservation
from .quality import Quality, assess_quality


@dataclass(frozen=True)
class ViewTarget:
    name: str
    center: float
    lower: float
    upper: float
    max_roll: float = 20

    def contains(self, pose):
        return self.lower <= pose.yaw_cue <= self.upper and abs(pose.roll_deg) <= self.max_roll


DEFAULT_TARGETS = (ViewTarget('front', 0, -.13, .13),
                   ViewTarget('left', -.45, -.75, -.25),
                   ViewTarget('right', .45, .25, .75))


def target_quality(frame, detected, target, min_face_pixels=100, min_laplacian=12):
    """Face detail, visible eye/nose detail, and fit to a comparable pose band."""
    points, box, pose = detected['points'], detected['box'], detected['pose']
    five = len(points) == 5
    q = assess_quality(frame, points, box, pose, min_face_pixels, min_laplacian,
                       core_points=points if five else None).to_dict()
    if not q['accepted'] or target is None:
        return q
    # A forward photograph requires particularly unambiguous eyes/nose. These
    # are concentration heuristics, not a calibrated probability of visibility.
    confidence = detected.get('uncertainty') or {}
    if target.name == 'front' and confidence and (confidence.get('eye_entropy', 99) >= 3.3
                                                 or confidence.get('nose_entropy', 99) >= 3.5):
        q['accepted'] = False
        q['reasons'].append('front_landmarks_uncertain')
        return q
    centers = [*points[:2], points[2]] if five else [points[EYE_A].mean(0), points[EYE_B].mean(0), points[24:32].mean(0)]
    sharpness, dark = [], []
    for index, center in enumerate(centers):
        radius = max(4, pose.eye_distance * (.16 if index < 2 else .27))
        x1, y1 = np.maximum([0, 0], np.floor(center - radius)).astype(int)
        x2, y2 = np.minimum([frame.shape[1], frame.shape[0]], np.ceil(center + radius + 1)).astype(int)
        patch = frame[y1:y2, x1:x2]
        if not patch.size:
            q['accepted'], q['reasons'] = False, ['invalid_detail_region']
            return q
        gray = cv2.cvtColor(cv2.resize(patch, (40, 40), interpolation=cv2.INTER_AREA
                                       if max(patch.shape[:2]) > 40 else cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY)
        smooth = cv2.GaussianBlur(gray, (3, 3), .55)
        sharpness.append(float(cv2.Laplacian(smooth, cv2.CV_32F).var()))
        dark.append(float(np.percentile(gray, 20)))
    front = target.name == 'front'
    # The five-point path has no per-keypoint uncertainty output. Require actual
    # dark eye detail as well as geometry, temporal agreement and detector score.
    if five and (max(dark[:2]) if front else min(dark[:2])) > 135:
        q['accepted'], q['reasons'] = False, ['upstream_eye_visibility_unresolved']
        return q
    eye_focus = min(sharpness[:2]) if front else max(sharpness[:2])
    detail = .5 * math.log1p(eye_focus) + .5 * math.log1p(sharpness[2])
    extent = max(target.center - target.lower, target.upper - target.center)
    fit = max(0., 1 - abs(pose.yaw_cue - target.center) / extent)
    # Hard bands govern comparability. Within the band, clarity is primary and
    # target closeness breaks otherwise similar choices; no angle is fabricated.
    selection_score = (.65 * q['score'] + .35 * detail) * (.85 + .15 * fit) * (1 - .08 * abs(pose.roll_deg) / target.max_roll)
    q.update(detail_laplacian=sharpness, eye_dark_p20=dark[:2], detail_score=detail,
             pose_fit=fit, selection_score=selection_score)
    return q


def _rank(snapshot):
    if not snapshot:
        return (-1, -1, -1.)
    frames = snapshot['frames']
    return (len(frames), int('front' in frames), sum(v['record']['quality']['selection_score'] for v in frames.values()))


class TargetViewPipeline:
    """Single-worker, bounded-memory API; no file/network work in process().

    models=None requires a same-frame FaceObservation on every process call.
    An absent observation rejects the frame; it never starts another model.
    Track IDs/long losses define independent bundles; identities never merge.
    """
    def __init__(self, models=None, *, targets=DEFAULT_TARGETS, hold_seconds=.12,
                 max_observation_gap=.20, lost_seconds=.75, min_face_pixels=100,
                 min_laplacian=12, verification_period=.15):
        self.models = models
        self.targets = tuple(targets)
        if not self.targets or len({t.name for t in self.targets}) != len(self.targets):
            raise ValueError('Targets need unique nonempty names')
        for target in self.targets:
            if (not target.name or not target.name.isidentifier()
                    or not all(math.isfinite(x) for x in (target.lower, target.center, target.upper, target.max_roll))
                    or not target.lower < target.center < target.upper or target.max_roll <= 0):
                raise ValueError('Invalid target band')
        ordered = sorted(self.targets, key=lambda t: t.lower)
        if any(a.upper >= b.lower for a, b in zip(ordered, ordered[1:])):
            raise ValueError('Target bands must not overlap')
        if not 0 < hold_seconds <= max_observation_gap <= lost_seconds or verification_period < 0:
            raise ValueError('Invalid temporal thresholds')
        self.hold, self.max_gap, self.lost = hold_seconds, max_observation_gap, lost_seconds
        self.verification_period = verification_period
        self.min_face_pixels, self.min_laplacian = min_face_pixels, min_laplacian
        self.previous_time = self.last_valid = None
        self.last_track_id = None
        self.previous_shape = None
        self.previous_rotation = None
        self.segment = 0
        self.frames = {}
        self.archived = None
        self.events = deque(maxlen=128)
        self.verifications = self.verification_rejects = 0
        self._clear_visit()

    def _clear_visit(self):
        self.visit = None
        self.since = None
        self.count = 0
        self.anchors = deque(maxlen=32)
        self.pending = None
        self.confirmed = False
        self.last_attempt = None
        self.last_verify_time = -float('inf')

    def _target(self, pose):
        if not pose.valid:
            return None
        return next((t for t in self.targets if t.contains(pose)), None)

    def _record(self, frame, detected, timestamp, frame_index):
        valid = detected['valid']
        pose = detected['pose'] if valid else Pose(reason=detected['reason'])
        target = self._target(pose)
        quality = target_quality(frame, detected, target, self.min_face_pixels, self.min_laplacian) if valid else Quality(False, [detected['reason']]).to_dict()
        if valid and target is None:
            quality['accepted'] = False
            quality['reasons'].append('outside_target_bands')
        return {'frame_index': int(frame_index), 'time_s': float(timestamp), 'pose': pose.to_dict(),
                'quality': quality, 'target_view': target.name if target else None,
                'box': list(detected['box']) if valid else None,
                'landmarks': detected['points'].tolist() if valid else None,
                'landmark_schema': detected.get('landmark_schema', 'dogflw46'),
                'inference_mode': detected.get('inference_mode', 'no_face'),
                'uncertainty': detected.get('uncertainty'),
                'rotation_quarters': detected.get('rotation_quarters', 0),
                'dog_score': detected.get('dog_score'), 'timings_ms': dict(detected.get('timings_ms', {}))}

    def _promote(self, timestamp, force=False):
        if not self.confirmed or self.pending is None:
            return
        candidate = self.pending
        record, image = candidate['record'], candidate['image']
        name = record['target_view']
        key = record['frame_index']
        old = self.frames.get(name)
        if self.last_attempt == key or (old and record['quality']['selection_score'] <= old['record']['quality']['selection_score']):
            return
        if not force and timestamp - self.last_verify_time < self.verification_period:
            return
        self.last_attempt, self.last_verify_time = key, timestamp
        if record['inference_mode'] == 'optical_flow':
            self.verifications += 1
            checked = self.models.verify_candidate(image, record)
            verified = self._record(image, checked, record['time_s'], key)
            if not verified['quality']['accepted'] or verified['target_view'] != name:
                self.verification_rejects += 1
                # A failed peak must not block a lower-scoring, valid frame later.
                self.pending = None
                return
            verified['preverification_pose'] = record['pose']
            record = verified
            self.pending = {'image': image, 'record': record}
        if old and record['quality']['selection_score'] <= old['record']['quality']['selection_score']:
            return
        record = {**record, 'selection_basis': 'independent_view_confirmation',
                  'confirmed_at_s': float(timestamp), 'segment': self.segment,
                  'verification': 'same_frame_neural' if record['inference_mode'] in ('neural', 'candidate_neural') else 'same_frame_upstream'}
        self.frames[name] = {'image': image, 'record': record}
        self.events.append({'event': 'view_updated', 'view': name, 'frame_index': key,
                            'time_s': float(timestamp), 'segment': self.segment})

    def _reset(self, timestamp, reason):
        self._promote(timestamp, force=True)
        current = self._snapshot_current()
        if _rank(current) > _rank(self.archived):
            self.archived = current
        self.segment += 1
        self.frames = {}
        self.last_valid = None
        self._clear_visit()
        self.events.append({'event': 'reset', 'reason': reason, 'segment': self.segment, 'time_s': float(timestamp)})

    def process(self, frame, timestamp, frame_index, observation: FaceObservation | None = None):
        if not math.isfinite(timestamp) or (self.previous_time is not None and timestamp <= self.previous_time):
            raise ValueError('Capture times must be finite and strictly increasing')
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError('Expected uint8 BGR frame')
        started = time.perf_counter()
        if self.models is None:
            detected = observation.detect(frame, timestamp, frame_index) if observation is not None else {
                'valid': False, 'reason': 'missing_upstream_observation', 'inference_mode': 'upstream_missing', 'timings_ms': {}}
        else:
            if observation is not None:
                raise ValueError('Choose upstream observations OR standalone models for one pipeline')
            detected = self.models.infer(frame, timestamp)
        track = detected.get('track_id')
        changed = track is not None and self.last_track_id is not None and track != self.last_track_id
        lost = self.last_valid is not None and timestamp - self.last_valid > self.lost
        shape_changed = self.previous_shape is not None and frame.shape != self.previous_shape
        rotation_changed = (detected['valid'] and self.previous_rotation is not None
                            and detected.get('rotation_quarters', 0) != self.previous_rotation)
        multiple = detected.get('reason') == 'multiple_dogs' and bool(self.frames or self.visit)
        if changed or lost or shape_changed or rotation_changed or multiple or detected.get('discontinuity', False):
            self._reset(timestamp, 'track_changed' if changed else 'long_loss' if lost else 'discontinuity')
        if track is not None:
            self.last_track_id = track
        self.previous_shape = frame.shape
        if detected['valid']:
            self.previous_rotation = detected.get('rotation_quarters', 0)
        record = self._record(frame, detected, timestamp, frame_index)
        candidate_name = record['target_view'] if record['quality']['accepted'] else None
        # Pose confirmation and photograph quality are different decisions.
        # A blurred neighbour may corroborate geometry but can never become a
        # saved image. A narrow exit band avoids losing a clear peak to one
        # boundary-jitter frame; entering a view still requires the strict band.
        hard_reasons = set(record['quality']['reasons']) - {'blurred', 'outside_target_bands'}
        name = record['target_view'] if detected['valid'] and not hard_reasons else None
        if name is None and self.visit and detected['valid'] and not hard_reasons:
            target = next(t for t in self.targets if t.name == self.visit)
            pose = detected['pose']
            if target.lower - .04 <= pose.yaw_cue <= target.upper + .04 and abs(pose.roll_deg) <= target.max_roll:
                name = target.name
        confidence = detected.get('uncertainty') or {}
        if name == 'front' and confidence and (confidence.get('eye_entropy', 99) >= 3.3
                                               or confidence.get('nose_entropy', 99) >= 3.5):
            name = None
        gap = self.previous_time is not None and timestamp - self.previous_time > self.max_gap
        if name != self.visit or gap:
            self._promote(timestamp, force=True)
            self._clear_visit()
            self.visit = name
        if detected['valid']:
            self.last_valid = timestamp
        if name:
            if self.since is None:
                self.since = timestamp
            self.count += 1
            if record['inference_mode'] in ('neural', 'upstream'):
                self.anchors.append(timestamp)
            independent = len(self.anchors) >= 2 and self.anchors[-1] - self.anchors[0] >= .08
            self.confirmed = bool(self.count >= 3 and timestamp - self.since >= self.hold and independent)
            if candidate_name == name and (self.pending is None or record['quality']['selection_score'] > self.pending['record']['quality']['selection_score']):
                self.pending = {'image': frame.copy(), 'record': record.copy()}
            self._promote(timestamp)
        self.previous_time = timestamp
        missing = [t.name for t in self.targets if t.name not in self.frames]
        # Compatibility with the existing single-worker viewer; this is NOT an
        # ordered action challenge and sequence_complete stays false.
        record['tracking'] = {'state': name or 'unknown', 'stable_state': name if name and self.confirmed else 'unknown',
                              'target': missing[0] if missing else 'done', 'segment': self.segment,
                              'step': len(self.frames), 'step_count': len(self.targets),
                              'sequence_complete': False, 'coverage_complete': not missing}
        record['compute_ms'] = (time.perf_counter() - started) * 1000
        return record

    def finalize(self):
        if self.previous_time is not None:
            self._promote(self.previous_time, force=True)

    def _snapshot_current(self):
        return {'segment': self.segment, 'track_id': self.last_track_id, 'frames': self.frames.copy(),
                'complete': len(self.frames) == len(self.targets)}

    def snapshot(self):
        current = self._snapshot_current()
        return self.archived if _rank(self.archived) > _rank(current) else current

    def configuration(self):
        return {'targets': [asdict(t) for t in self.targets], 'hold_seconds': self.hold,
                'max_observation_gap': self.max_gap, 'lost_seconds': self.lost,
                'verification_period': self.verification_period,
                'confirmation_exit_margin': .04,
                'upstream_only': self.models is None,
                'note': 'Yaw is a dimensionless landmark cue, not degrees. Directions use upright face coordinates.'}
