"""Detector backends reduced to ``Detection`` (box + score + named keypoints).

ONNX decoders (no framework code at runtime, only NumPy + ONNX Runtime):

``YoloPoseDetector``  Ultralytics YOLOv8/YOLO11 pose export, any keypoint count
                      (``tools/train_dog_face_pose.py`` trains a 6-point dog face model).
``AnyFaceDetector``   YOLOv5-face ``export_cat`` graph, e.g. AnyFace YOLOv5l6
                      (``tools/export_anyface_onnx.py``); 5 points.
``ScrfdDetector``     InsightFace SCRFD graph (anchor-free, strides 8/16/32,
                      2 anchors); 5 points.  For an SCRFD retrained on dog faces.
``CropLandmarker``    keypoints of the one face on a canonical crop, any roll (the refiner's
                      per-frame tracking network; ``tools/train_dog_face_landmarks.py``).
``YoloxBodyGate``     YOLOX COCO detector, used only to confirm the face is a dog's.

Adapters: ``LegacyDogFlwDetector`` wraps the v0.x DogFLW TFLite cascade.
"""
from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path
from threading import Lock

import cv2
import numpy as np

from .runtime import create_cuda_graph_session, create_session
from .resampling import resize_area
from .schema import ANYFACE5, CORE3, SCHEMAS, Detection, KeypointSchema, get_schema, nms


def letterbox(image, size, center=True, pad_value=114):
    """Keep aspect ratio; returns (canvas, ratio, (pad_x, pad_y))."""
    h, w = image.shape[:2]
    ratio = min(size / h, size / w)
    nw, nh = max(1, round(w * ratio)), max(1, round(h * ratio))
    resized = (resize_area(image, (nw, nh)) if ratio < 1 else
               cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR))
    canvas = np.full((size, size, 3), pad_value, np.uint8)
    px, py = ((size - nw) // 2, (size - nh) // 2) if center else (0, 0)
    canvas[py:py + nh, px:px + nw] = resized
    return canvas, ratio, (px, py)


def read_sidecar(model_path):
    """Optional ``<model>.json`` written by the export/training tools."""
    sidecar = Path(model_path).with_suffix('.json')
    if sidecar.is_file():
        value = json.loads(sidecar.read_text(encoding='utf8'))
        if not isinstance(value, dict):
            raise ValueError(f'{sidecar}: model metadata must be a JSON object')
        return value
    return {}


class OnnxDetector:
    kind = 'onnx'

    def __init__(self, model, *, input_size=0, score_threshold=0.35, nms_iou=0.45, max_faces=8,
                 providers='auto', threads=0, cache_dir='', cuda_graph=False, thread_spinning=True,
                 spin_duration_us=-1, spin_backoff_max=1):
        self.model_path = Path(model)
        from .zoo import verify_model
        self.sha256, self.sidecar = verify_model(model)
        self._session_settings = dict(providers=providers, threads=threads, cache_dir=cache_dir,
                                      thread_spinning=thread_spinning, spin_duration_us=spin_duration_us,
                                      spin_backoff_max=spin_backoff_max)
        compiled = self.sidecar.get('cpu_compiled')
        if compiled is not None:
            if not isinstance(compiled, dict) or cuda_graph:
                raise ValueError('Invalid CPU compilation metadata or incompatible CUDA Graph request')
            self._session_settings['cpu_compiled'] = compiled
        self._graph = None
        self._load_session(cuda_graph=cuda_graph)
        self._graph_lock = Lock() if self._graph is not None else None
        shape = self.input.shape
        fixed = isinstance(shape[2], int) and isinstance(shape[3], int)
        # a dynamic export runs full frames at the size asked for, else the one it was made for
        self.size = int(shape[2]) if fixed else int(input_size or self.sidecar.get('input_size') or 640)
        self.dynamic = not fixed   # may also run at other square sizes (refine.crop_size)
        if fixed and shape[2] != shape[3]:
            raise ValueError(f'Only square detector inputs are supported, got {shape}')
        self.score_threshold, self.nms_iou, self.max_faces = score_threshold, nms_iou, max_faces
        self.metadata = dict(self.session.get_modelmeta().custom_metadata_map or {})
        if fixed and self.sidecar.get('input_size', self.size) != self.size:
            raise ValueError(f'{model}: sidecar input_size disagrees with ONNX input {shape}')
        self.dtype = np.float16 if 'float16' in self.input.type else np.float32

    def _load_session(self, *, cuda_graph=False, graph_error=None):
        graph = None
        if cuda_graph:
            try:
                session, info, graph = create_cuda_graph_session(self.model_path, **self._session_settings)
            except Exception as exc:
                graph_error = f'initialization: {exc}'.partition('\n')[0][:300]
        if graph is None:
            session, info = create_session(self.model_path, **self._session_settings)
        if self._graph is not None:
            self._graph.close()
        self._graph, self.session, self.runtime = graph, session, info
        info['cuda_graph'] = {'requested': cuda_graph or graph_error is not None,
                              'active': graph is not None, 'fallback_reason': graph_error}
        self.input = session.get_inputs()[0]
        self.outputs = [output.name for output in session.get_outputs()]
        compiled = self.sidecar.get('cpu_compiled') or {}
        self.conditional_score_input = compiled.get('conditional_score_input')
        embedded = session.get_modelmeta().custom_metadata_map or {}
        if embedded.get('dog_capture.conditional_score_input') != self.conditional_score_input:
            raise ValueError('CPU conditional graph requires its matching compilation sidecar')
        if self.conditional_score_input:
            inputs = session.get_inputs()
            if (len(inputs) != 2 or inputs[1].name != self.conditional_score_input
                    or inputs[1].shape != [] or inputs[1].type != 'tensor(float)'):
                raise ValueError('CPU conditional graph and sidecar input contract disagree')

    def warmup(self, runs=2):
        frame = np.zeros((self.size, self.size, 3), np.uint8)
        if self.conditional_score_input:
            # A blank image normally skips the pose heads. Warm them explicitly
            # so the first real face does not pay their allocation/setup cost.
            tensor = self.preprocess(frame)[0]
            for _ in range(runs):
                self.session.run(self.outputs, self._feed(tensor, threshold=-np.inf))
            return
        for _ in range(runs):
            self.detect(frame)

    def _feed(self, tensor, *, threshold=None):
        feed = {self.input.name: tensor.astype(self.dtype, copy=False)}
        if self.conditional_score_input:
            feed[self.conditional_score_input] = np.asarray(
                self.score_threshold if threshold is None else threshold, np.float32)
        return feed

    def _run(self, tensor):
        tensor = tensor.astype(self.dtype, copy=False)
        if self._graph_lock is not None:
            with self._graph_lock:
                if self._graph is not None:
                    if tensor.shape != self._graph.shape:
                        raise ValueError(f'Expected CUDA Graph input {self._graph.shape}, got {tensor.shape}')
                    try:
                        return self._graph.run(tensor)
                    except Exception as exc:
                        reason = f'replay: {exc}'.partition('\n')[0][:300]
                    self._load_session(graph_error=reason)
        return self.session.run(self.outputs, self._feed(tensor))

    def describe(self):
        return {'kind': self.kind, 'model': str(self.model_path), 'sha256': self.sha256, 'input_size': self.size,
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
    """BGR -> contiguous RGB/NCHW, with one allocation and FP32 arithmetic.

    Write each channel straight to its final plane: an HWC float copy, separate
    subtract/multiply arrays and ORT's implicit contiguous copy are unnecessary.
    Keep subtraction before multiplication (including SCRFD's nonzero mean).
    """
    h, w = canvas.shape[:2]
    tensor = np.empty((1, 3, h, w), np.float32)
    scale, mean = np.float32(scale), np.float32(mean)
    for channel in range(3):
        plane = tensor[0, channel]
        source = canvas[:, :, 2 - channel]
        if mean:
            np.subtract(source, mean, out=plane, dtype=np.float32)
            np.multiply(plane, scale, out=plane)
        else:
            np.multiply(source, scale, out=plane, dtype=np.float32)
    return tensor


def _xywh_to_xyxy(b):
    return np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)


class YoloPoseDetector(OnnxDetector):
    kind = 'yolo_pose'

    def __init__(self, model, schema='', **kwargs):
        super().__init__(model, **kwargs)
        kpt = self.sidecar.get('kpt_shape') or self.metadata.get('kpt_shape')
        if isinstance(kpt, str):
            kpt = ast.literal_eval(kpt)
        embedded_kpt = self.metadata.get('kpt_shape')
        if embedded_kpt and list(ast.literal_eval(embedded_kpt)) != list(kpt):
            raise ValueError(f'{model}: sidecar kpt_shape disagrees with ONNX metadata')
        if not kpt:
            raise ValueError(f'{model}: no kpt_shape in ONNX metadata or sidecar JSON; not a YOLO pose export?')
        self.num_kpts, self.kpt_dims = int(kpt[0]), int(kpt[1])
        named = self._schema_from_names(self.metadata.get('kpt_names'))
        if schema or self.sidecar.get('schema'):
            self.schema = get_schema(schema or self.sidecar.get('schema'))
        elif named is not None:
            self.schema = named
        else:
            self.schema = get_schema({6: 'dogface6', 5: 'anyface5', 3: 'core3'}.get(self.num_kpts, ''))
        if len(self.schema) != self.num_kpts:
            raise ValueError(f'{model}: {self.num_kpts} keypoints do not fit schema {self.schema.name}')
        if self.sidecar.get('yaw_scale') is not None:   # per-model yaw calibration (see schema.DOGFACE6)
            value = float(self.sidecar['yaw_scale'])
            if not 0.2 <= value <= 5:
                raise ValueError(f'{model}: sidecar yaw_scale {value} out of range')
            self.schema = replace(self.schema, yaw_scale=value)
        if named is not None and self.schema.names != named.names:
            raise ValueError(f'{model}: keypoint names/order disagree with schema {self.schema.name}')
        names = self.metadata.get('names')
        self.num_classes = len(ast.literal_eval(names)) if names else int(self.sidecar.get('num_classes', 1))

    def preprocess(self, frame, size=None):
        canvas, ratio, pad = letterbox(frame, size or self.size)
        return _rgb_tensor(canvas), ratio, pad

    def runs_at(self, size):
        """True if this model can run on a square input of ``size`` (refinement crops)."""
        return size == self.size or (self.dynamic and self._graph is None)

    @staticmethod
    def _schema_from_names(value):
        """Ultralytics stores ``kpt_names`` as "{0: ['left_eye', ...]}"; use it if it names eyes and nose."""
        if not value:
            return None
        try:
            parsed = ast.literal_eval(value)
            names = tuple(parsed[0] if isinstance(parsed, dict) else parsed)
        except (ValueError, SyntaxError, KeyError, TypeError):
            return None
        for known in SCHEMAS.values():
            if known.names == names:
                return known
        if not {'left_eye', 'right_eye', 'nose'} <= set(names):
            return None
        pairs = tuple((n, 'right_' + n[5:]) for n in names if n.startswith('left_') and 'right_' + n[5:] in names)
        return KeypointSchema('custom_' + '_'.join(names)[:40], names, pairs)

    def detect(self, frame, size=None):
        tensor, ratio, pad = self.preprocess(frame, size)
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


class CropLandmarker(OnnxDetector):
    """One dog face centred on a canonical crop, at any roll (tools/train_dog_face_landmarks.py).

    The refiner's tracking network: ``detect(crop)`` returns at most one Detection in crop
    pixels -- 6 keypoints, the face box (centre and long side), the presence probability as
    the score, per-keypoint visibility as ``kp_scores`` and the face's roll at any angle as
    ``extra['roll_deg']`` (clockwise, like refine.eye_line_deg).  Keypoints are trained within
    ``keypoint_max_roll`` of upright; beyond that only the roll is meaningful, and the refiner
    re-crops upright.  It does not search the image: the full-frame detector finds faces, this
    follows them.  ``rotation_invariant``: no four-way orientation search is needed.
    """
    kind = 'crop_landmarks'
    rotation_invariant = True

    def __init__(self, model, schema='', **kwargs):
        super().__init__(model, **kwargs)
        self.schema = get_schema(schema or self.sidecar.get('schema') or 'dogface6')
        self.share = float(self.sidecar.get('share', 0.35))
        self.keypoint_max_roll = float(self.sidecar.get('keypoint_max_roll', 45.0))
        k = len(self.schema)
        width = self.session.get_outputs()[0].shape[-1]
        if isinstance(width, int) and width != 3 * k + 6:
            raise ValueError(f'{model}: expected {3 * k + 6} outputs for {k} keypoints, got {width}')

    def runs_at(self, size):
        return size == self.size

    def detect(self, image, size=None):
        if image.shape[:2] != (self.size, self.size):
            raise ValueError(f'CropLandmarker takes {self.size} px crops, got {image.shape[:2]}')
        out = self._run(_rgb_tensor(image))[0][0].astype(np.float64)
        k = len(self.schema)
        points = out[:2 * k].reshape(k, 2)
        cx, cy, side = out[2 * k:2 * k + 3]
        logits = out[2 * k + 3:]
        presence = float(1 / (1 + np.exp(-logits[0])))
        if presence < self.score_threshold or not side > 0:
            return []
        box = np.array([cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2], np.float32)
        visible = (1 / (1 + np.exp(-logits[1:1 + k]))).astype(np.float32)
        cos, sin = logits[1 + k:3 + k]
        det = Detection(box, presence, points.astype(np.float32), self.schema, visible, source=self.kind)
        det.extra['roll_deg'] = float(np.degrees(np.arctan2(sin, cos)))
        return [det]


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
