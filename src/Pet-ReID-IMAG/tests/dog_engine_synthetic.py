"""Deterministic synthetic dog-face scenes with exact keypoints for engine tests."""
import math

import cv2
import numpy as np

from dog_capture.engine.schema import ANYFACE5, Detection

_RNG = np.random.default_rng(7)
_FUR = cv2.GaussianBlur((_RNG.random((512, 512)) * 255).astype(np.uint8), (0, 0), 1.2)
_BACKGROUND = cv2.GaussianBlur((_RNG.random((720, 1280)) * 255).astype(np.uint8), (0, 0), 6)


def canonical_points(yaw=0.0):
    """Five AnyFace-order points in iod units around the eye midpoint."""
    return np.array([[-0.5, 0.0], [0.5, 0.0], [yaw, 0.5],
                     [-0.46 + 0.5 * yaw, 0.63], [0.46 + 0.5 * yaw, 0.63]], np.float64)


def render(center=(640, 360), iod=90.0, yaw=0.0, roll=0.0, blur=0.0, gain=1.0, size=(720, 1280),
           extra_faces=()):
    """Return (bgr image, [Detection]) for one or more synthetic faces."""
    h, w = size
    image = cv2.resize(_BACKGROUND, (w, h))
    image = cv2.cvtColor(image // 3 + 60, cv2.COLOR_GRAY2BGR)
    detections = []
    for spec in ((center, iod, yaw, roll),) + tuple(extra_faces):
        image, det = _draw_face(image, *spec)
        detections.append(det)
    if blur > 0:
        image = cv2.GaussianBlur(image, (0, 0), blur)
    if gain != 1.0:
        image = np.clip(image.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return image, detections


def _draw_face(image, center, iod, yaw, roll):
    s = 200  # canonical canvas: 1 iod = 100 px, eye midpoint at (s, s)
    unit = 100.0
    canvas = np.zeros((2 * s, 2 * s, 3), np.uint8)
    mask = np.zeros((2 * s, 2 * s), np.uint8)
    face_c = (round(s + yaw * 30), round(s + 25))
    cv2.ellipse(mask, face_c, (100, 92), 0, 0, 360, 255, -1)
    fur = cv2.resize(_FUR, (2 * s, 2 * s))
    canvas[...] = cv2.cvtColor((fur // 2 + 90), cv2.COLOR_GRAY2BGR)
    pts = canonical_points(yaw) * unit + s
    for i, (x, y) in enumerate(pts[:2]):
        r = 13 * (1 - 0.35 * abs(yaw) if (i == 0 and yaw > 0) or (i == 1 and yaw < 0) else 1)
        cv2.circle(canvas, (round(x), round(y)), max(4, round(r)), (25, 20, 15), -1)
        cv2.circle(canvas, (round(x - 3), round(y - 3)), 3, (240, 240, 240), -1)
    nx, ny = pts[2]
    cv2.ellipse(canvas, (round(nx), round(ny)), (19, 13), 0, 0, 360, (20, 20, 20), -1)
    cv2.line(canvas, tuple(np.round(pts[3]).astype(int)), (round(nx), round(ny + 20)), (40, 35, 30), 3)
    cv2.line(canvas, tuple(np.round(pts[4]).astype(int)), (round(nx), round(ny + 20)), (40, 35, 30), 3)
    scale = iod / unit
    angle = math.radians(roll)
    rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]) * scale
    matrix = np.hstack([rot, (np.asarray(center, np.float64) - rot @ [s, s])[:, None]])
    hh, ww = image.shape[:2]
    warped = cv2.warpAffine(canvas, matrix, (ww, hh), flags=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    warped_mask = cv2.warpAffine(mask, matrix, (ww, hh)) > 127
    image = image.copy()
    image[warped_mask] = warped[warped_mask]
    points = pts @ matrix[:, :2].T + matrix[:, 2]
    corners = np.array([[face_c[0] - 100, face_c[1] - 92 - 10], [face_c[0] + 100, face_c[1] - 92 - 10],
                        [face_c[0] - 100, face_c[1] + 92], [face_c[0] + 100, face_c[1] + 92]], np.float64)
    mapped = corners @ matrix[:, :2].T + matrix[:, 2]
    box = np.r_[mapped.min(0), mapped.max(0)]
    return image, Detection(box, 0.9, points, ANYFACE5, source='synthetic')


class ScriptedDetector:
    """Returns pre-computed detections; mimics a real detector interface."""

    def __init__(self, detections=None):
        self.queue = list(detections or [])
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return self.queue.pop(0) if self.queue else []

    def describe(self):
        return {'kind': 'scripted'}
