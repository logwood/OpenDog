"""Detector backends reduced to ``Detection`` (box + score + named keypoints).

ONNX decoders (no framework code at runtime, only NumPy + ONNX Runtime):

``YoloPoseDetector``  Ultralytics YOLOv8/YOLO11 pose export, any keypoint count
                      (``tools/train_dog_face_pose.py`` trains a 6-point dog face model).
``AnyFaceDetector``   YOLOv5-face ``export_cat`` graph, e.g. AnyFace YOLOv5l6
                      (``tools/export_anyface_onnx.py``); 5 points.
``ScrfdDetector``     InsightFace SCRFD graph (anchor-free, strides 8/16/32,
                      2 anchors); 5 points.  For an SCRFD retrained on dog faces.
``YoloxBodyGate``     YOLOX COCO detector, used only to confirm the face is a dog's.

Adapters: ``LegacyDogFlwDetector`` wraps the v0.x DogFLW TFLite cascade.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import cv2
import numpy as np

from .runtime import create_session
from .schema import ANYFACE5, CORE3, Detection, get_schema, nms


def letterbox(image, size, center=True, pad_value=114):
    """Keep aspect ratio; returns (canvas, ratio, (pad_x, pad_y))."""
    h, w = image.shape[:2]
    ratio = min(size / h, size / w)
    nw, nh = max(1, round(w * ratio)), max(1, round(h * ratio))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad_value, np.uint8)
    px, py = ((size - nw) // 2, (size - nh) // 2) if center else (0, 0)
    canvas[py:py + nh, px:px + nw] = resized
    return canvas, ratio, (px, py)


def read_sidecar(model_path):
    """Optional ``<model>.json`` written by the export/training tools."""
    sidecar = Path(model_path).with_suffix('.json')
    if sidecar.is_file():
        try:
            return json.loads(sidecar.read_text(encoding='utf8'))
        except (OSError, ValueError):
            return {}
    return {}


class OnnxDetector:
    kind = 'onnx'

    def __init__(self, model, *, input_size=640, score_threshold=0.35, nms_iou=0.45, max_faces=8,
                 providers='auto', threads=0, cache_dir=''):
        self.model_path = Path(model)
        self.session, self.runtime = create_session(model, providers, threads, cache_dir)
        self.input = self.session.get_inputs()[0]
        self.outputs = [o.name for o in self.session.get_outputs()]
        shape = self.input.shape
        fixed = isinstance(shape[2], int) and isinstance(shape[3], int)
        self.size = int(shape[2]) if fixed else int(input_size)
        if fixed and shape[2] != shape[3]:
            raise ValueError(f'Only square detector inputs are supported, got {shape}')
        self.score_threshold, self.nms_iou, self.max_faces = score_threshold, nms_iou, max_faces
        self.metadata = dict(self.session.get_modelmeta().custom_metadata_map or {})
        self.sidecar = read_sidecar(model)
        self.dtype = np.float16 if 'float16' in self.input.type else np.float32

    def warmup(self, runs=2):
        frame = np.zeros((self.size, self.size, 3), np.uint8)
        for _ in range(runs):
            self.detect(frame)

    def _run(self, tensor):
        return self.session.run(self.outputs, {self.input.name: tensor.astype(self.dtype, copy=False)})

    def describe(self):
        return {'kind': self.kind, 'model': str(self.model_path), 'input_size': self.size,
                'score_threshold': self.score_threshold, 'runtime': self.runtime,
                'license': self.sidecar.get('license')}

    def _finish(self, boxes, scores, points, kp_scores, ratio, pad, shape, schema):
        if not len(boxes):
            return []
        keep = nms(boxes, scores, self.nms_iou, self.max_faces)
        px, py = pad
        h, w = shape[:2]
        results = []
        for i in keep:
            box = (boxes[i] - [px, py, px, py]) / ratio
            box = np.clip(box, [0, 0, 0, 0], [w, h, w, h])
            kps = (points[i] - [px, py]) / ratio
            results.append(Detection(box, float(scores[i]), kps, schema,
                                     None if kp_scores is None else kp_scores[i], source=self.kind))
        return results


def _rgb_tensor(canvas, scale=1 / 255.0, mean=0.0):
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
    return ((rgb - mean) * scale).transpose(2, 0, 1)[None]


def _xywh_to_xyxy(b):
    return np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)


class YoloPoseDetector(OnnxDetector):
    kind = 'yolo_pose'

    def __init__(self, model, schema='', **kwargs):
        super().__init__(model, **kwargs)
        kpt = self.sidecar.get('kpt_shape') or self.metadata.get('kpt_shape')
        if isinstance(kpt, str):
            kpt = ast.literal_eval(kpt)
        if not kpt:
            raise ValueError(f'{model}: no kpt_shape in ONNX metadata or sidecar JSON; not a YOLO pose export?')
        self.num_kpts, self.kpt_dims = int(kpt[0]), int(kpt[1])
        self.schema = get_schema(schema or self.sidecar.get('schema') or
                                 {6: 'dogface6', 5: 'anyface5', 3: 'core3'}.get(self.num_kpts, ''))
        if len(self.schema) != self.num_kpts:
            raise ValueError(f'{model}: {self.num_kpts} keypoints do not fit schema {self.schema.name}')
        names = self.metadata.get('names')
        self.num_classes = len(ast.literal_eval(names)) if names else int(self.sidecar.get('num_classes', 1))

    def preprocess(self, frame):
        canvas, ratio, pad = letterbox(frame, self.size)
        return _rgb_tensor(canvas), ratio, pad

    def detect(self, frame):
        tensor, ratio, pad = self.preprocess(frame)
        out = self._run(tensor)[0][0]
        k, d = self.num_kpts, self.kpt_dims
        if out.shape[-1] == 6 + k * d and out.shape[0] <= 1000:  # end-to-end export (NMS in graph)
            out = out[out[:, 4] >= self.score_threshold]
            boxes, scores = out[:, :4], out[:, 4]
            kps = out[:, 6:].reshape(-1, k, d)
        else:
            out = out.T  # (N, 4 + nc + k*d)
            cls = out[:, 4:4 + self.num_classes]
            scores = cls.max(1)
            out, scores = out[scores >= self.score_threshold], scores[scores >= self.score_threshold]
            boxes = _xywh_to_xyxy(out[:, :4])
            kps = out[:, 4 + self.num_classes:].reshape(-1, k, d)
        visibility = kps[:, :, 2] if d == 3 else None
        return self._finish(boxes, scores, kps[:, :, :2], visibility, ratio, pad, frame.shape, self.schema)


class AnyFaceDetector(OnnxDetector):
    """YOLOv5-face ``export_cat`` output: [1, N, 16] = cx cy w h obj 10*landmark cls."""
    kind = 'anyface'

    def preprocess(self, frame):
        canvas, ratio, pad = letterbox(frame, self.size)
        return _rgb_tensor(canvas), ratio, pad

    def detect(self, frame):
        tensor, ratio, pad = self.preprocess(frame)
        out = self._run(tensor)[0]
        out = out.reshape(-1, out.shape[-1])
        if out.shape[1] < 16:
            raise ValueError(f'Unexpected YOLOv5-face output width {out.shape[1]}; export with export_cat')
        scores = out[:, 4] * out[:, 15:].max(1)
        out, scores = out[scores >= self.score_threshold], scores[scores >= self.score_threshold]
        return self._finish(_xywh_to_xyxy(out[:, :4]), scores, out[:, 5:15].reshape(-1, 5, 2), None,
                            ratio, pad, frame.shape, ANYFACE5)


class ScrfdDetector(OnnxDetector):
    """InsightFace SCRFD (MIT) decoding: distance-to-box/keypoint per anchor centre."""
    kind = 'scrfd'

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        count = len(self.outputs)
        if count not in (6, 9, 10, 15):
            raise ValueError(f'Unexpected SCRFD output count {count}')
        self.fmc = 3 if count in (6, 9) else 5
        self.strides = [8, 16, 32] if self.fmc == 3 else [8, 16, 32, 64, 128]
        self.has_kps = count in (9, 15)
        if not self.has_kps:
            raise ValueError('This SCRFD export has no keypoint outputs; the engine needs eyes and nose')
        self.anchors = 2
        self._centres = {}

    def _centre(self, stride):
        if stride not in self._centres:
            n = self.size // stride
            y, x = np.mgrid[:n, :n]
            centres = (np.stack([x, y], -1).reshape(-1, 2) * stride).astype(np.float32)
            self._centres[stride] = np.repeat(centres, self.anchors, axis=0)
        return self._centres[stride]

    def preprocess(self, frame):
        canvas, ratio, pad = letterbox(frame, self.size, center=False, pad_value=0)
        return _rgb_tensor(canvas, 1 / 128.0, 127.5), ratio, pad

    def detect(self, frame):
        tensor, ratio, pad = self.preprocess(frame)
        outputs = self._run(tensor)
        boxes, scores, points = [], [], []
        for i, stride in enumerate(self.strides):
            score = outputs[i].reshape(-1)
            dist = outputs[i + self.fmc].reshape(-1, 4) * stride
            kps = outputs[i + 2 * self.fmc].reshape(-1, 10) * stride
            centres = self._centre(stride)
            keep = score >= self.score_threshold
            c = centres[keep]
            d = dist[keep]
            boxes.append(np.stack([c[:, 0] - d[:, 0], c[:, 1] - d[:, 1], c[:, 0] + d[:, 2], c[:, 1] + d[:, 3]], 1))
            points.append(np.repeat(c, 5, axis=0).reshape(-1, 5, 2) + kps[keep].reshape(-1, 5, 2))
            scores.append(score[keep])
        boxes, scores, points = np.concatenate(boxes), np.concatenate(scores), np.concatenate(points)
        return self._finish(boxes, scores, points, None, ratio, pad, frame.shape, ANYFACE5)


class YoloxBodyGate(OnnxDetector):
    """YOLOX (Apache-2.0) COCO detector; returns dog boxes only."""
    kind = 'yolox'

    def __init__(self, model, class_id=16, legacy=None, **kwargs):
        super().__init__(model, **kwargs)
        self.class_id = class_id
        # The 0.1.1rc0 release ONNX files take raw BGR 0-255 (verified on dog
        # photos); ``legacy`` = RGB/255 + ImageNet mean/std for older exports.
        self.legacy = bool(self.sidecar.get('legacy_preprocess', False) if legacy is None else legacy)
        grids, strides = [], []
        for stride in (8, 16, 32):
            n = self.size // stride
            y, x = np.mgrid[:n, :n]
            grids.append(np.stack([x, y], -1).reshape(-1, 2))
            strides.append(np.full((n * n, 1), stride))
        self.grid = np.concatenate(grids).astype(np.float32)
        self.stride = np.concatenate(strides).astype(np.float32)

    def preprocess(self, frame):
        canvas, ratio, pad = letterbox(frame, self.size, center=False)
        if self.legacy:
            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tensor = ((rgb - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).transpose(2, 0, 1)[None]
        else:
            tensor = canvas.astype(np.float32).transpose(2, 0, 1)[None]
        return tensor.astype(np.float32), ratio, pad

    def detect(self, frame):
        tensor, ratio, _ = self.preprocess(frame)
        out = self._run(tensor)[0][0]
        xy = (out[:, :2] + self.grid) * self.stride
        wh = np.exp(np.clip(out[:, 2:4], -10, 10)) * self.stride
        scores = out[:, 4] * out[:, 5 + self.class_id]
        best = out[:, 5:].argmax(1) == self.class_id  # the class must win, not just pass
        keep = (scores >= self.score_threshold) & best
        boxes = _xywh_to_xyxy(np.concatenate([xy, wh], 1)[keep])
        scores = scores[keep]
        if not len(boxes):
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
        idx = nms(boxes, scores, 0.45, 16)
        h, w = frame.shape[:2]
        boxes = np.clip(boxes[idx] / ratio, 0, [w, h, w, h]).astype(np.float32)
        return boxes, scores[idx].astype(np.float32)


class LegacyDogFlwDetector:
    """v0.x three-model TFLite cascade (dog detector, face localiser, 46 points).

    Runs every frame (no optical flow) and reduces 46 points to eye centres and
    the nose centre.  Heatmap entropy becomes a pseudo keypoint confidence.
    The cascade has its own dog gate, so no body gate is needed.  Weights are
    CC BY-NC 4.0 (DogFLW), i.e. non-commercial only.
    """
    kind = 'dogflw'

    def __init__(self, directory, threads=4, device='cpu'):
        from ..geometry import EYE_A, EYE_B, NOSE
        from ..models import DogFaceModels
        self.models = DogFaceModels(directory, threads, device=device, compact_confidence=True,
                                    landmark_period=0.0)
        self.indices = (EYE_A, EYE_B, NOSE)
        self.directory = str(directory)
        self.clock = 0.0

    @staticmethod
    def _confidence(entropy):
        return float(np.clip((4.2 - entropy) / 1.6, 0, 1))

    def detect(self, frame):
        self.clock += 1.0  # forces a fresh dog detection and landmark pass
        self.models.last_landmarks = -float('inf')
        self.models.retry_at = -float('inf')
        result = self.models.infer(frame, self.clock)
        if not result.get('valid'):
            return []
        points = result['points']
        eye_a, eye_b, nose = (points[i].mean(0) for i in self.indices)
        u = result.get('uncertainty') or {}
        eye_conf = self._confidence(u.get('eye_entropy', 9))
        kp_scores = [eye_conf, eye_conf, self._confidence(u.get('nose_entropy', 9))]
        return [Detection(np.asarray(result['box'], np.float32), float(result.get('dog_score') or 0.5),
                          np.stack([eye_a, eye_b, nose]), CORE3, kp_scores, source=self.kind)]

    def landmarks_in_box(self, frame, box):
        """Run only the 46-point landmarker on a given face box (teacher use).

        Skips the dog detector, the face localiser and the v0.x gates; the box
        is resized to the landmarker's 384 x 384 input like the cascade does.
        """
        from ..landmarks import uncertainty
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (int(round(v)) for v in np.clip(box, 0, [w, h, w, h]))
        if x2 - x1 < 16 or y2 - y1 < 16:
            return None
        crop = cv2.resize(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB), (384, 384))
        outputs = self.models.landmarker(crop[None].astype(np.float32) / 255)
        entropy = uncertainty(outputs[1])
        points = outputs[0].reshape(46, 2) * [x2 - x1, y2 - y1] + [x1, y1]
        eye_a, eye_b, nose = (points[i].mean(0) for i in self.indices)
        eye_conf = self._confidence(entropy['eye_entropy'])
        return Detection(np.asarray([x1, y1, x2, y2], np.float32), 1.0, np.stack([eye_a, eye_b, nose]), CORE3,
                         [eye_conf, eye_conf, self._confidence(entropy['nose_entropy'])], source='dogflw_box')

    def describe(self):
        return {'kind': self.kind, 'model': self.directory, 'device': self.models.device,
                'license': 'CC BY-NC 4.0 (DogFLW landmarks)'}

    def close(self):
        self.models.close()
