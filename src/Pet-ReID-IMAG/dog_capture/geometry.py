"""DogFLW geometry and a causal, timestamp-based capture state machine.

Directions describe the nose in the displayed image, NOT anatomical left/right.
The yaw cue is dimensionless 2-D geometry, not an angle or a probability.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import deque
import math

import numpy as np

EYE_A = [16, 18, 20, 22]
EYE_B = [17, 19, 21, 23]
NOSE = list(range(24, 32))


@dataclass
class Pose:
    yaw_cue: float = 0.0
    roll_deg: float = 0.0
    nose_height: float = 0.0
    eye_distance: float = 0.0
    valid: bool = False
    reason: str = "no_face"

    def to_dict(self):
        return asdict(self)


def estimate_pose(points: np.ndarray, box, image_shape) -> Pose:
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (46, 2) or not np.isfinite(points).all():
        return Pose(reason="invalid_landmarks")
    a, b = points[EYE_A].mean(0), points[EYE_B].mean(0)
    if a[0] > b[0]:
        a, b = b, a
    span = float(np.linalg.norm(b - a))
    if span < 8 or span < 0.075 * (box[2] - box[0]):
        return Pose(eye_distance=span, reason="eyes_unresolved")
    horizontal = (b - a) / span
    vertical = np.array([-horizontal[1], horizontal[0]])
    offset = points[NOSE].mean(0) - (a + b) / 2
    yaw = float(offset @ horizontal / span)
    pitch_cue = float(offset @ vertical / span)
    roll = math.degrees(math.atan2(horizontal[1], horizontal[0]))
    h, w = image_shape[:2]
    core = points[16:32]
    reason = "ok"
    if (core[:, 0] < 3).any() or (core[:, 0] > w - 4).any() or (core[:, 1] < 3).any() or (core[:, 1] > h - 4).any():
        reason = "face_clipped"
    elif abs(roll) > 35:
        reason = "head_tilt"
    elif not 0.10 <= pitch_cue <= 1.75 or abs(yaw) > 1.5:
        reason = "extreme_pose"
    return Pose(yaw, roll, pitch_cue, span, reason == "ok", reason)


class TurnTracker:
    """Hysteresis + short dwell, with strict invalid-frame and long-gap resets."""

    def __init__(self, hold_seconds=0.25, smoothing_seconds=0.10,
                 side_threshold=0.25, front_threshold=0.13,
                 turn_order=("left", "right"), lost_seconds=0.75):
        if hold_seconds <= 0 or smoothing_seconds < 0 or not 0 < front_threshold < side_threshold:
            raise ValueError("Invalid capture thresholds")
        if tuple(sorted(turn_order)) != ("left", "right"):
            raise ValueError("turn_order must contain left and right once")
        self.hold = hold_seconds
        self.tau = smoothing_seconds
        self.side = side_threshold
        self.front = front_threshold
        self.lost_seconds = lost_seconds
        self.sequence = ["front", turn_order[0], "front", turn_order[1], "front"]
        self.index = 0
        self.segment = 0
        self.filtered = None
        self.previous_time = None
        self.last_valid = None
        self.pending = "unknown"
        self.since = None
        self.count = 0
        self.events = deque(maxlen=4096)

    def reset(self, timestamp, reason="track_discontinuity"):
        self.segment += 1
        self.index = 0
        self.filtered = None
        self.last_valid = None
        self.pending, self.since, self.count = "unknown", None, 0
        self.events.append({"time_s": timestamp, "event": "reset", "reason": reason, "segment": self.segment})

    def update(self, timestamp: float, pose: Pose, usable: bool, discontinuity=False):
        timestamp = float(timestamp)
        if not math.isfinite(timestamp) or (self.previous_time is not None and timestamp <= self.previous_time):
            raise ValueError("Frame timestamps must be finite and strictly increasing")
        dt = timestamp - self.previous_time if self.previous_time is not None else 0.0
        if discontinuity or (self.last_valid is not None and timestamp - self.last_valid > self.lost_seconds):
            self.reset(timestamp)
        self.previous_time = timestamp
        completed_step = None
        raw_state = "unknown"
        stable = "unknown"
        if not usable or not pose.valid:
            # No carry-over dwell across occlusion, blur, or absence.
            self.pending, self.since, self.count = "unknown", None, 0
            self.filtered = None
        else:
            self.last_valid = timestamp
            alpha = 1.0 if self.filtered is None or self.tau == 0 else 1 - math.exp(-dt / self.tau)
            self.filtered = pose.yaw_cue if self.filtered is None else self.filtered + alpha * (pose.yaw_cue - self.filtered)
            # Raw cue must agree; smoothing alone must not create a false dwell.
            if self.filtered <= -self.side and pose.yaw_cue <= -self.front:
                raw_state = "left"
            elif self.filtered >= self.side and pose.yaw_cue >= self.front:
                raw_state = "right"
            elif abs(self.filtered) <= self.front and abs(pose.yaw_cue) <= self.front:
                raw_state = "front"
            elif self.pending == "left" and self.filtered < -self.front and pose.yaw_cue < -self.front:
                raw_state = "left"
            elif self.pending == "right" and self.filtered > self.front and pose.yaw_cue > self.front:
                raw_state = "right"
            if raw_state == "unknown":
                self.pending, self.since, self.count = "unknown", None, 0
            elif raw_state != self.pending or dt > 0.30 + 1e-9:
                self.pending, self.since, self.count = raw_state, timestamp, 1
            else:
                self.count += 1
            if self.since is not None and timestamp - self.since >= self.hold and self.count >= 3:
                stable = raw_state
                if self.index < len(self.sequence) and stable == self.sequence[self.index]:
                    completed_step = self.sequence[self.index]
                    self.events.append({"time_s": timestamp, "event": "step", "pose": completed_step, "step": self.index, "segment": self.segment})
                    self.index += 1
        return {
            "state": raw_state, "stable_state": stable, "yaw_smoothed": self.filtered,
            "segment": self.segment, "step": self.index, "steps_total": len(self.sequence),
            "target": self.sequence[self.index] if self.index < len(self.sequence) else "done",
            "completed_step": completed_step, "sequence_complete": self.index == len(self.sequence),
        }
