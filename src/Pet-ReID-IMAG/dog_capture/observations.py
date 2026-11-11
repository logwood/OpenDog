"""Small, model-free boundary for existing AnyFace / DogFLW results.

No torch, SAM, Re-ID or LiteRT import is needed for this interface. Coordinates
and timestamps must belong to the exact BGR frame passed to the selector.
"""
from dataclasses import dataclass
import math

import numpy as np

from .geometry import Pose
from .landmarks import canonical_points, landmark_gate
from .models import clip_box


@dataclass(frozen=True)
class FaceObservation:
    frame_index: int
    time_s: float
    track_id: str
    bbox_xyxy: tuple
    landmarks_xy: tuple
    confidence: float
    schema: str = 'anyface5'
    dog_count: int = 1
    rotation_quarters: int | None = None
    uncertainty: dict | None = None

    @classmethod
    def from_anyface(cls, detection, *, frame_index, time_s, track_id, dog_count=1, rotation_quarters=None):
        """Accept pet_id.localization.FaceDetection without importing pet_id.

        Upstream must establish dog presence and track identity. AnyFace also
        detects other animals; its class_id alone is not a dog-presence gate.
        """
        return cls(frame_index, time_s, str(track_id) if track_id is not None else '', tuple(detection.bbox_xyxy),
                   tuple(tuple(p) for p in detection.landmarks_xy), float(detection.confidence),
                   dog_count=dog_count, rotation_quarters=rotation_quarters)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)

    def detect(self, frame, timestamp, frame_index):
        def reject(reason):
            return {'valid': False, 'reason': reason, 'timings_ms': {},
                    'inference_mode': 'upstream_rejected', 'track_id': self.track_id}
        if self.frame_index != frame_index or not math.isfinite(self.time_s) or abs(self.time_s - timestamp) > 1e-4:
            return reject('stale_upstream_observation')
        if not isinstance(self.track_id, str) or not self.track_id.strip():
            return reject('missing_track_id')
        if self.dog_count != 1:
            return reject('multiple_dogs' if self.dog_count > 1 else 'no_dog')
        if not math.isfinite(self.confidence) or not .60 <= self.confidence <= 1:
            return reject('upstream_confidence_low')
        if self.schema not in ('anyface5', 'dogflw46') or (self.rotation_quarters is not None and self.rotation_quarters not in range(4)):
            return reject('unsupported_landmark_schema')
        try:
            points = np.asarray(self.landmarks_xy, np.float32)
            raw_box = np.asarray(self.bbox_xyxy, np.float32)
        except (ValueError, TypeError):
            return reject('invalid_upstream_geometry')
        expected = 5 if self.schema == 'anyface5' else 46
        if (points.shape != (expected, 2) or not np.isfinite(points).all()
                or raw_box.shape != (4,) or not np.isfinite(raw_box).all()
                or raw_box[2] <= raw_box[0] or raw_box[3] <= raw_box[1]):
            return reject('invalid_upstream_geometry')
        box = clip_box(raw_box, frame.shape)
        if box[2] - box[0] < 24 or box[3] - box[1] < 24:
            return reject('invalid_upstream_box')
        h, w = frame.shape[:2]
        core = points if expected == 5 else points[16:32]
        if (core[:, 0].min() < 3 or core[:, 1].min() < 3 or core[:, 0].max() > w - 4
                or core[:, 1].max() > h - 4 or core[:, 0].min() < box[0]
                or core[:, 1].min() < box[1] or core[:, 0].max() > box[2]
                or core[:, 1].max() > box[3]):
            return reject('face_clipped')
        rotation = self.rotation_quarters
        if expected == 46:
            if rotation is None:
                return reject('missing_landmark_orientation')
            if not self.uncertainty or not all(k in self.uncertainty for k in ('eye_entropy', 'nose_entropy', 'core_entropy')):
                return reject('missing_landmark_uncertainty')
            valid, reason, pose = landmark_gate(points, box, rotation, self.uncertainty)
        else:
            candidates = []
            # A quarter-turn can be determined from existing geometry without
            # rotating and running the detector again. Nose must be below eyes;
            # the opposite orientation is therefore not accepted as upright.
            for k in range(4) if rotation is None else (rotation,):
                canonical, crop_box = canonical_points(points, box, k)
                candidate = estimate_five_point_pose(canonical, crop_box)
                candidates.append((not candidate.valid, abs(candidate.roll_deg), k, candidate))
            _, _, rotation, pose = min(candidates, key=lambda value: value[:3])
            valid, reason = pose.valid, pose.reason
        return {'valid': valid, 'reason': reason, 'pose': pose, 'points': points,
                'box': box, 'rotation_quarters': rotation,
                'uncertainty': self.uncertainty, 'dog_score': self.confidence,
                'inference_mode': 'upstream', 'timings_ms': {}, 'track_id': self.track_id,
                'landmark_schema': self.schema}


def estimate_five_point_pose(points, box):
    """Same dimensionless eye/nose cue; five-point nose is AnyFace nose_top.

    It is not calibrated in degrees and not numerically identical to DogFLW's
    nose centroid. Target bands remain configurable per upstream model.
    """
    a, b = sorted(points[:2], key=lambda p: p[0])
    span = float(np.linalg.norm(b - a))
    if span < 18 or span < .15 * (box[2] - box[0]):
        return Pose(eye_distance=span, reason='eyes_unresolved')
    horizontal = (b - a) / span
    vertical = np.array([-horizontal[1], horizontal[0]])
    offset = points[2] - (a + b) / 2
    yaw, height = float(offset @ horizontal / span), float(offset @ vertical / span)
    roll = math.degrees(math.atan2(horizontal[1], horizontal[0]))
    mouth = float(((points[3] + points[4]) / 2 - points[2]) @ vertical / span)
    reason = 'ok'
    if abs(roll) > 25:
        reason = 'head_tilt'
    elif not .25 <= height <= 1.4 or abs(yaw) > 1.0 or not -.10 <= mouth <= .85:
        reason = 'upstream_pose_unresolved'
    return Pose(yaw, roll, height, span, reason == 'ok', reason)
