"""Rank original frames using only face pixels at a consistent spatial scale."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import cv2
import numpy as np

from .geometry import Pose


@dataclass
class Quality:
    accepted: bool
    reasons: list[str]
    score: float = 0.0
    laplacian: float = 0.0
    gradient: float = 0.0
    contrast: float = 0.0
    face_pixels: int = 0
    clipped_fraction: float = 0.0

    def to_dict(self):
        return asdict(self)


def assess_quality(frame, points, box, pose: Pose, min_face_pixels=100, min_laplacian=12.0, core_points=None) -> Quality:
    reasons = []
    if not pose.valid:
        return Quality(False, [pose.reason])
    h, w = frame.shape[:2]
    core = np.asarray(points[16:] if core_points is None else core_points, dtype=np.float32)
    # Eyes / muzzle / jaw only: never rank using sharp grass, fences, or ears.
    x1, y1 = np.floor(core.min(0)).astype(int)
    x2, y2 = np.ceil(core.max(0)).astype(int) + 1
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
    size = int(min(x2 - x1, y2 - y1))
    if size < 12:
        return Quality(False, ["face_too_small"], face_pixels=size)
    if min(box[2] - box[0], box[3] - box[1]) < min_face_pixels or pose.eye_distance < 18:
        reasons.append("face_too_small")
    region = frame[y1:y2, x1:x2]
    mask = np.zeros(region.shape[:2], np.uint8)
    hull = cv2.convexHull(np.round(core - [x1, y1]).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    # Fixed face scale avoids rewarding a larger crop simply for its dimensions.
    scale = 160.0 / max(region.shape[:2])
    dims = (max(12, round(region.shape[1] * scale)), max(12, round(region.shape[0] * scale)))
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, dims, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    mask = cv2.resize(mask, dims, interpolation=cv2.INTER_NEAREST)
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8)) > 0
    if mask.sum() < 50:
        return Quality(False, ["invalid_face_region"])
    values = gray[mask]
    contrast = float(np.percentile(values, 95) - np.percentile(values, 5))
    clipped = float(np.mean(values >= 250))
    # Mild denoising keeps isolated sensor noise from masquerading as detail.
    smooth = cv2.GaussianBlur(gray, (3, 3), 0.55)
    lap = float(cv2.Laplacian(smooth, cv2.CV_32F)[mask].var())
    gx, gy = cv2.Sobel(smooth, cv2.CV_32F, 1, 0), cv2.Sobel(smooth, cv2.CV_32F, 0, 1)
    gradient = float(np.mean((gx * gx + gy * gy)[mask]))
    if contrast < 20 or float(np.percentile(values, 90)) < 25:
        reasons.append("too_dark_or_flat")
    if clipped > 0.45:
        reasons.append("overexposed")
    if lap < min_laplacian:
        reasons.append("blurred")
    focus = 0.7 * math.log1p(lap) + 0.3 * math.log1p(gradient / 16)
    resolution = min(1.0, size / 240)
    score = focus * (0.75 + 0.25 * resolution) * (1 - min(clipped, 0.8))
    return Quality(not reasons, reasons, score, lap, gradient, contrast, size, clipped)


class BestFrames:
    """Bounded memory; originals stay untouched. Different track segments never mix."""

    def __init__(self):
        self.segment = None
        self.frames = {}
        self.updates = 0
        self.pending = None
        self.pending_state = None
        self.front_checks = []

    def consider(self, frame, record):
        state = record["tracking"]["stable_state"]
        segment = record["tracking"]["segment"]
        if segment != self.segment:
            self.frames.clear()
            self.segment = segment
            self.pending = None
            self.pending_state = None
            self.front_checks = []
        raw = record['tracking'].get('state', state)
        if raw != self.pending_state or not record['quality']['accepted']:
            self.pending = None
            self.front_checks = []
        self.pending_state = raw
        if raw in ('front', 'left', 'right') and record['quality']['accepted']:
            yaw = record['pose']['yaw_cue']
            eligible = ((raw == 'front' and abs(yaw) <= .13 and abs(record['pose']['roll_deg']) <= 20)
                        or (raw == 'left' and yaw < -.13) or (raw == 'right' and yaw > .13))
            if eligible and (self.pending is None or record['quality']['score'] > self.pending['record']['quality']['score']):
                self.pending = {'image': frame.copy(), 'record': record}
        # A usable identity photograph does not require the entire action dwell.
        # Two distinct neural observations may confirm a fleeting frontal view
        # even when real-time processing skipped the intervening frames. This
        # does NOT advance the action state machine.
        confirmed_photo = False
        confidence = record.get('uncertainty') or {}
        if (raw == 'front' and record['quality']['accepted'] and self.pending is not None
                and record.get('inference_mode') == 'neural'
                and confidence.get('eye_entropy', 99) < 3.3
                and confidence.get('nose_entropy', 99) < 3.5):
            now = record['time_s']
            self.front_checks = [t for t in self.front_checks if .03 <= now - t <= .50]
            if self.front_checks:
                best = self.pending
                if 'front' not in self.frames or best['record']['quality']['score'] > self.frames['front']['record']['quality']['score']:
                    self.frames['front'] = {'image': best['image'].copy(),
                                            'record': {**best['record'], 'selection_basis': 'two_neural_front_observations'}}
                    self.updates += 1
                    confirmed_photo = True
            self.front_checks.append(now)
            self.front_checks = self.front_checks[-2:]
        if state not in ("front", "left", "right") or not record["quality"]["accepted"]:
            return confirmed_photo
        # Require agreement with current unsmoothed pose: don't save an old state.
        yaw = record["pose"]["yaw_cue"]
        if (state == "front" and (abs(yaw) > 0.13 or abs(record["pose"]["roll_deg"]) > 20)) or (state == "left" and yaw >= -0.13) or (state == "right" and yaw <= 0.13):
            return False
        item = self.pending or {'image': frame, 'record': record}
        score = item['record']['quality']['score']
        if state not in self.frames or score > self.frames[state]["record"]["quality"]["score"]:
            self.frames[state] = {"image": item['image'].copy(), "record": {**item['record'], 'selection_basis': 'confirmed_pose_dwell'}}
            self.updates += 1
            return True
        return False
