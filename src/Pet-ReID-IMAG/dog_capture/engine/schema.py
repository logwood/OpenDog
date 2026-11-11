"""Detector-to-engine contract: one face box, its score and named keypoints.

Every detector backend (AnyFace/YOLOv5-face, Ultralytics YOLO-pose, SCRFD,
the legacy DogFLW cascade or an upstream system) is reduced to this one
structure, the same way face SDKs reduce RetinaFace/SCRFD/MTCNN output to
"box + score + 5 points".  Downstream code only uses keypoint *names*, never
detector-specific indices.

Naming convention: ``left_*`` is the point on the image-left side of an
upright, frontal face (the dog's anatomical right), matching the 5-point
order used by RetinaFace, SCRFD and AnyFace.  The engine re-orders the eyes
geometrically before use, so a detector with the opposite convention still
works.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np


@dataclass(frozen=True)
class KeypointSchema:
    name: str
    names: tuple[str, ...]
    # Pairs swapped by a horizontal mirror / by the engine's eye re-ordering.
    pairs: tuple[tuple[str, str], ...] = ()

    def __len__(self):
        return len(self.names)

    def index(self, key):
        try:
            return self.names.index(key)
        except ValueError:
            return None


ANYFACE5 = KeypointSchema(
    'anyface5', ('left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth'),
    (('left_eye', 'right_eye'), ('left_mouth', 'right_mouth')))
# Face subset of the StanfordExtra / Ultralytics dog-pose 24 keypoints.
DOGFACE6 = KeypointSchema(
    'dogface6', ('left_eye', 'right_eye', 'nose', 'chin', 'left_ear_base', 'right_ear_base'),
    (('left_eye', 'right_eye'), ('left_ear_base', 'right_ear_base')))
# Minimal schema: eye centres and nose (e.g. reduced DogFLW 46-point output).
CORE3 = KeypointSchema('core3', ('left_eye', 'right_eye', 'nose'), (('left_eye', 'right_eye'),))

SCHEMAS = {s.name: s for s in (ANYFACE5, DOGFACE6, CORE3)}
REQUIRED = ('left_eye', 'right_eye', 'nose')


def get_schema(value) -> KeypointSchema:
    if isinstance(value, KeypointSchema):
        return value
    if value not in SCHEMAS:
        raise ValueError(f'Unknown keypoint schema {value!r}; known: {sorted(SCHEMAS)}')
    return SCHEMAS[value]


@dataclass
class Detection:
    """One face in *frame pixel* coordinates of the frame given to the engine."""
    box: np.ndarray                      # (4,) float32 x1, y1, x2, y2
    score: float                         # detector confidence in [0, 1]
    keypoints: np.ndarray                # (K, 2) float32
    schema: KeypointSchema
    kp_scores: np.ndarray | None = None  # (K,) in [0, 1]; None if the model has none
    body_box: np.ndarray | None = None   # associated dog body box (species gate)
    body_score: float | None = None
    source: str = ''
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.schema = get_schema(self.schema)
        self.box = np.asarray(self.box, np.float32).reshape(4)
        self.keypoints = np.asarray(self.keypoints, np.float32).reshape(-1, 2)
        if self.kp_scores is not None:
            self.kp_scores = np.clip(np.asarray(self.kp_scores, np.float32).reshape(-1), 0, 1)
        self.score = float(self.score)

    def validate(self):
        """Return None if usable, otherwise a short machine-readable reason."""
        if len(self.keypoints) != len(self.schema):
            return 'keypoint_count_mismatch'
        if self.kp_scores is not None and len(self.kp_scores) != len(self.schema):
            return 'keypoint_score_count_mismatch'
        if not (np.isfinite(self.box).all() and np.isfinite(self.keypoints).all()
                and math.isfinite(self.score)):
            return 'nonfinite_detection'
        if self.box[2] <= self.box[0] or self.box[3] <= self.box[1]:
            return 'empty_box'
        if not 0 <= self.score <= 1:
            return 'score_out_of_range'
        if any(self.schema.index(k) is None for k in REQUIRED):
            return 'schema_without_eyes_and_nose'
        return None

    def point(self, name):
        index = self.schema.index(name)
        return None if index is None else self.keypoints[index]

    def point_score(self, name):
        index = self.schema.index(name)
        if index is None or self.kp_scores is None:
            return None
        return float(self.kp_scores[index])

    @property
    def width(self):
        return float(self.box[2] - self.box[0])

    @property
    def height(self):
        return float(self.box[3] - self.box[1])

    def to_dict(self):
        value = {'box': [round(float(v), 2) for v in self.box], 'score': round(self.score, 4),
                 'schema': self.schema.name,
                 'keypoints': [[round(float(x), 2), round(float(y), 2)] for x, y in self.keypoints]}
        if self.kp_scores is not None:
            value['kp_scores'] = [round(float(v), 4) for v in self.kp_scores]
        if self.body_box is not None:
            value['body_box'] = [round(float(v), 2) for v in self.body_box]
            value['body_score'] = None if self.body_score is None else round(float(self.body_score), 4)
        if self.source:
            value['source'] = self.source
        return value

    @classmethod
    def from_dict(cls, value):
        """Accept the engine's own JSON and the legacy FaceObservation fields."""
        box = value.get('box', value.get('bbox_xyxy'))
        points = value.get('keypoints', value.get('landmarks_xy'))
        schema = value.get('schema', 'anyface5')
        score = value.get('score', value.get('confidence'))
        if box is None or points is None or score is None:
            raise ValueError('Detection needs box/bbox_xyxy, keypoints/landmarks_xy and score/confidence')
        return cls(np.asarray(box, np.float32), float(score), np.asarray(points, np.float32), schema,
                   kp_scores=value.get('kp_scores'),
                   body_box=None if value.get('body_box') is None else np.asarray(value['body_box'], np.float32),
                   body_score=value.get('body_score'), source=value.get('source', 'external'))


def box_iou_matrix(a, b):
    """Pairwise IoU of (N, 4) and (M, 4) xyxy boxes."""
    a = np.asarray(a, np.float32).reshape(-1, 4)
    b = np.asarray(b, np.float32).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-6)


def nms(boxes, scores, iou_threshold, max_output=None):
    """Plain greedy NMS; returns kept indices sorted by score."""
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    order = np.argsort(-np.asarray(scores, np.float32), kind='stable')
    keep = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if max_output and len(keep) >= max_output:
            break
        if order.size == 1:
            break
        overlap = box_iou_matrix(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][overlap <= iou_threshold]
    return np.asarray(keep, np.int64)
