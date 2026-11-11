"""Head pose from facial keypoints and a temporally stable view classifier.

Geometry
    The eyes are re-ordered so that the nose lies *below* the eye line.  This
    makes the pose independent of detector left/right conventions and of any
    in-plane rotation (camera held sideways), replacing the old four-way
    90-degree re-inference with arithmetic.

Yaw cue
    ``yaw = (nose - eye midpoint) . eye_axis / eye_span``, the same
    dimensionless cue as the v0.x selector, so existing view bands still apply.
    Under a pinhole model with eyes at +-e and the nose tip protruding by d,
    ``yaw = (d / 2e) * tan(theta)``; ``approx_deg`` inverts this with a fixed
    snout factor k = d / 2e.  Snout length differs by breed, so the degree
    value is for display only and is never used for decisions.

    Where available, the mouth midpoint / chin offset gives a second yaw cue;
    a sign disagreement between the cues marks the landmarks as unreliable,
    the 5-point consistency check used by face SDKs before pose gating.

View classifier
    One-Euro smoothing, an entry band, a wider exit band (hysteresis) and a
    minimum dwell in time *and* observations.  This is the debouncing used by
    face-enrolment UIs, and replaces the old pending/anchor state machine.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math

import numpy as np

from .tracking import OneEuroFilter


@dataclass
class FaceGeometry:
    points: dict = field(default_factory=dict)  # canonical name -> (2,) frame coordinates
    scores: dict = field(default_factory=dict)  # canonical name -> keypoint confidence
    iod: float = 0.0                 # inter-ocular distance (pixels)
    roll_deg: float = 0.0            # eye-line angle; 0 = upright
    yaw: float = 0.0                 # dimensionless cue, negative = nose to the image left
    pitch: float = 0.0               # nose drop below the eye line / iod
    lower_yaw: float | None = None   # mouth-midpoint or chin cue, if present
    approx_deg: float = 0.0          # display only
    swapped: bool = False            # detector eye order was reversed
    valid: bool = False
    reason: str = 'no_face'

    def to_dict(self):
        value = asdict(self)
        value['points'] = {k: [round(float(v[0]), 2), round(float(v[1]), 2)] for k, v in self.points.items()}
        value['scores'] = {k: round(float(v), 4) for k, v in self.scores.items()}
        for key in ('iod', 'roll_deg', 'yaw', 'pitch', 'approx_deg'):
            value[key] = round(float(value[key]), 4)
        if value['lower_yaw'] is not None:
            value['lower_yaw'] = round(float(value['lower_yaw']), 4)
        return value


def face_geometry(detection, config) -> FaceGeometry:
    points = {name: np.asarray(detection.keypoints[i], np.float64)
              for i, name in enumerate(detection.schema.names)}
    scores = ({name: float(detection.kp_scores[i]) for i, name in enumerate(detection.schema.names)}
              if detection.kp_scores is not None else {})
    a, b, nose = points['left_eye'], points['right_eye'], points['nose']
    axis = b - a
    span = float(np.linalg.norm(axis))
    if span < 1e-6:
        return FaceGeometry(points, scores, reason='eyes_coincide')
    normal = np.array([-axis[1], axis[0]]) / span
    swapped = float((nose - (a + b) / 2) @ normal) < 0
    if swapped:
        for left, right in detection.schema.pairs:
            if left in points and right in points:
                points[left], points[right] = points[right], points[left]
                if scores:
                    scores[left], scores[right] = scores[right], scores[left]
        a, b = points['left_eye'], points['right_eye']
        axis = b - a
        normal = np.array([-axis[1], axis[0]]) / span
    horizontal = axis / span
    mid = (a + b) / 2
    offset = nose - mid
    yaw = float(offset @ horizontal / span)
    pitch = float(offset @ normal / span)
    roll = math.degrees(math.atan2(horizontal[1], horizontal[0]))
    lower = None
    if 'left_mouth' in points and 'right_mouth' in points:
        lower = float(((points['left_mouth'] + points['right_mouth']) / 2 - mid) @ horizontal / span)
    elif 'chin' in points and (not scores or scores.get('chin', 1) >= 0.3):
        lower = float((points['chin'] - mid) @ horizontal / span)
    geometry = FaceGeometry(points, scores, span, roll, yaw, pitch, lower,
                            math.degrees(math.atan2(yaw, config.snout_factor)), swapped)
    box_w = max(detection.width, detection.height)
    if span < config.min_iod_box_ratio * box_w:
        geometry.reason = 'eyes_unresolved'
    elif span > config.max_iod_box_ratio * box_w:
        geometry.reason = 'implausible_eye_span'
    elif not config.pitch_range[0] <= pitch <= config.pitch_range[1]:
        geometry.reason = 'implausible_nose_position'
    elif abs(yaw) > config.max_abs_yaw:
        geometry.reason = 'extreme_pose'
    elif lower is not None and abs(yaw) > 0.18 and abs(lower) > 0.18 and np.sign(lower) != np.sign(yaw):
        geometry.reason = 'inconsistent_landmarks'
    else:
        geometry.valid, geometry.reason = True, 'ok'
    return geometry


@dataclass
class ViewState:
    raw: str | None          # band containing this frame's own yaw (entry band)
    view: str | None         # debounced current view
    stable: bool             # dwell satisfied
    smoothed_yaw: float | None
    dwell_s: float = 0.0
    observations: int = 0
    changed: bool = False    # view differs from the previous call

    def to_dict(self):
        value = asdict(self)
        if value['smoothed_yaw'] is not None:
            value['smoothed_yaw'] = round(value['smoothed_yaw'], 4)
        value['dwell_s'] = round(value['dwell_s'], 4)
        return value


class ViewClassifier:
    """Per-track debounced view state; one instance per track."""

    def __init__(self, config, still=False):
        self.config = config
        self.views = {v.name: v for v in config.views}
        self.still = still
        self.filter = OneEuroFilter(config.smooth_min_cutoff, config.smooth_beta)
        self.reset()

    def reset(self):
        self.filter.reset()
        self.view = None
        self.since = None
        self.count = 0
        self.last_time = None

    def band(self, yaw, roll, margin=0.0):
        for v in self.views.values():
            if v.lower - margin <= yaw <= v.upper + margin and abs(roll) <= v.max_roll_deg:
                return v.name
        return None

    def update(self, timestamp, geometry: FaceGeometry | None) -> ViewState:
        c = self.config
        if self.still:  # unrelated photos: classify each image on its own
            if geometry is None or not geometry.valid:
                return ViewState(None, None, False, None)
            name = self.band(geometry.yaw, geometry.roll_deg)
            return ViewState(name, name, name is not None, geometry.yaw, 0.0, 1, True)
        previous = self.view
        if self.last_time is not None and timestamp - self.last_time > c.max_gap_s:
            self.reset()
        if geometry is None or not geometry.valid:
            # Invalid geometry (occlusion, impossible landmarks) breaks the dwell.
            self.view, self.since, self.count = None, None, 0
            self.filter.reset()
            self.last_time = timestamp
            return ViewState(None, None, False, None, changed=previous is not None)
        smoothed = float(self.filter(timestamp, geometry.yaw))
        roll = geometry.roll_deg
        raw = self.band(geometry.yaw, roll)
        entry = raw if raw is not None and self.band(smoothed, roll) == raw else None
        if self.view is not None:
            spec = self.views[self.view]
            m = c.exit_margin
            keep = (spec.lower - m <= geometry.yaw <= spec.upper + m and spec.lower - m <= smoothed <= spec.upper + m
                    and abs(roll) <= spec.max_roll_deg)
            if keep and (entry is None or entry == self.view):
                self.count += 1
            elif entry is not None:
                self.view, self.since, self.count = entry, timestamp, 1
            else:
                self.view, self.since, self.count = None, None, 0
        elif entry is not None:
            self.view, self.since, self.count = entry, timestamp, 1
        self.last_time = timestamp
        dwell = 0.0 if self.since is None else timestamp - self.since
        stable = self.view is not None and self.count >= c.min_dwell_obs and dwell >= c.min_dwell_s
        return ViewState(raw, self.view, stable, smoothed, dwell, self.count, self.view != previous)
