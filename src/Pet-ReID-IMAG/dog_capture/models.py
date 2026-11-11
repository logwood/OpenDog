"""Pinned pretrained TFLite assets and CPU inference, with original coordinates."""
from __future__ import annotations

import hashlib
from pathlib import Path
import time
import urllib.request
import zipfile
from collections import Counter

import cv2
import numpy as np

from .landmarks import (confidence_model, eye_evidence, landmark_gate, pose_in_crop,
                        rotate_normalized, uncertainty)

HF_REVISION = "ffe69649279e0c226e7061f917550ca8608abbfe"
HF_BASE = f"https://huggingface.co/hugocornellier/dog-face-landmarks/resolve/{HF_REVISION}"
ASSETS = {
    "dog_face_localizer.tflite": (f"{HF_BASE}/dog_face_localizer.tflite", "bf13f179335cec6046f5f158e218d4f5ef577f04af2896dbe8afbddc1e814b9d"),
    "dog_face_landmarks_full.tflite": (f"{HF_BASE}/dog_face_landmarks_full.tflite", "4036cd3d332f75be99e267b414ab74e08d4c3a2d4e123d5df7db543b995dcaca"),
    "efficientdet_lite0_nms.tflite": ("https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/object_detection/rpi/lite-model_efficientdet_lite0_detection_metadata_1.tflite", "2e04c53bfeac0ac2a30c057c7e2a777594ce39baaac35a92f74fb1e8c4fc4e0b"),
}


def file_hash(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def ensure_models(directory, download=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in ASSETS.items():
        path = directory / name
        if path.exists():
            if file_hash(path) != expected:
                raise ValueError(f"Model checksum mismatch: {path}")
            continue
        if not download:
            raise FileNotFoundError(f"Missing {path}; run with --download-models once")
        temporary = path.with_suffix(".download")
        request = urllib.request.Request(url, headers={"User-Agent": "DogCapturePrototype/0.1"})
        print(f"Downloading {name}", flush=True)
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as f:
            while block := response.read(1024 * 1024):
                f.write(block)
        if file_hash(temporary) != expected:
            raise ValueError(f"Downloaded model checksum mismatch: {temporary}")
        temporary.replace(path)
    return directory


def clip_box(box, shape, margin=0.0):
    h, w = shape[:2]
    x1, y1, x2, y2 = map(float, box)
    dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
    return (max(0, int(np.floor(x1 - dx))), max(0, int(np.floor(y1 - dy))),
            min(w, int(np.ceil(x2 + dx))), min(h, int(np.ceil(y2 + dy))))


def letterbox(rgb, size):
    h, w = rgb.shape[:2]
    ratio = size / max(h, w)
    nh, nw = max(1, round(h * ratio)), max(1, round(w * ratio))
    x, y = (size - nw) // 2, (size - nh) // 2
    output = np.zeros((size, size, 3), np.uint8)
    output[y:y + nh, x:x + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR)
    return output, (nw / w, nh / h, x, y)


def undo_letterbox(box, mapping, size):
    sx, sy, px, py = mapping
    x1, y1, x2, y2 = np.asarray(box) * size
    return ((x1 - px) / sx, (y1 - py) / sy, (x2 - px) / sx, (y2 - py) / sy)


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area - inter, 1)


class LiteModel:
    def __init__(self, path=None, threads=4, content=None):
        from ai_edge_litert.interpreter import Interpreter
        options = {'model_content': content} if content else {'model_path': str(path)}
        self.interpreter = Interpreter(**options, num_threads=threads)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.outputs = self.interpreter.get_output_details()

    def __call__(self, value):
        self.interpreter.set_tensor(self.input["index"], value.astype(self.input["dtype"], copy=False))
        self.interpreter.invoke()
        return [self.interpreter.get_tensor(d["index"]) for d in self.outputs]


class DogFaceModels:
    """Dog presence gate + one face localizer + 46-point landmarker.

    Neural refreshes anchor bounded optical-flow tracking. Presence, heatmap
    uncertainty and geometric checks gate each result. Multi-dog frames reject.
    """
    def __init__(self, directory, threads=4, dog_threshold=0.35, detection_period=0.30, landmark_period=0.35, device='auto', compact_confidence=False,
                 trt_directory=None, trt_cuda_graph=True):
        if device not in ('auto', 'cpu', 'gpu', 'trt'):
            raise ValueError(f'Unknown inference backend: {device}')
        directory = Path(directory) if device == 'trt' else ensure_models(directory)
        self.device = 'CPU'
        content = None if device == 'trt' else confidence_model(directory / 'dog_face_landmarks_full.tflite', compact=compact_confidence)
        self.compact_confidence = compact_confidence or device == 'trt'
        self.calls = Counter()
        self.runtime_assets = {}
        self.detector = self.localizer = self.landmarker = None
        if device == 'trt':
            from .trt import TrtModel, TrtDetector
            engines = Path(trt_directory) if trt_directory else directory.parent / 'dog_capture_trt_v3'
            if not all((engines / (name + '.engine')).is_file() for name in ('localizer', 'landmarks', 'detector')):
                raise FileNotFoundError(f'Missing TensorRT engines under {engines}; build them once with tools/prepare_dog_trt.py')
            try:
                self.localizer = TrtModel(engines / 'localizer.engine', use_cuda_graph=trt_cuda_graph)
                self.landmarker = TrtModel(engines / 'landmarks.engine', use_cuda_graph=trt_cuda_graph)
                self.detector = TrtDetector(engines, dog_threshold, use_cuda_graph=trt_cuda_graph)
            except BaseException:
                self.close()
                raise
            self.dog_class = self.detector.spec['dog_class']
            self.runtime_assets = {'localizer': self.localizer.metadata, 'landmarks': self.landmarker.metadata,
                                   'detector': self.detector.model.metadata}
            self.device = f'TensorRT {self.landmarker.precision.upper()}' + (' + CUDA Graph' if trt_cuda_graph else ' (direct enqueue)')
        if device in ('auto', 'gpu'):
            try:
                from .gpu import GpuModel
                self.localizer = GpuModel(directory / 'dog_face_localizer.tflite', threads)
                self.landmarker = GpuModel(threads=threads, content=content)
                self.device = 'LiteRT WebGPU + CPU detector'
            except Exception as exc:
                if device == 'gpu':
                    raise
                print(f'GPU unavailable; using CPU: {exc}', flush=True)
                self.localizer = self.landmarker = None
        if self.landmarker is None:
            self.localizer = LiteModel(directory / 'dog_face_localizer.tflite', threads)
            self.landmarker = LiteModel(threads=threads, content=content)
        if device != 'trt':
            self.detector = LiteModel(directory / "efficientdet_lite0_nms.tflite", threads)
            with zipfile.ZipFile(directory / "efficientdet_lite0_nms.tflite") as z:
                self.dog_class = z.read("labelmap.txt").decode().splitlines().index("dog")
        self.threshold = dog_threshold
        self.period = detection_period
        self.last_detection = -float("inf")
        self.body_box = None
        self.body_score = 0.0
        self.status = "no_dog"
        self.previous_face = None
        self.landmark_period = landmark_period
        self.last_landmarks = -float("inf")
        self.track_gray = None
        self.track_points = None
        self.rotation = 0
        self.next_rotation = 0
        self.retry_at = -float('inf')
        self.last_reason = 'no_face'
        self.confidence = None
        self.anchor_pose = None

    def close(self):
        for model in (self.detector, self.localizer, self.landmarker):
            close = getattr(model, 'close', None)
            if close:
                close()

    def clear_track(self):
        self.track_gray = self.track_points = self.previous_face = None
        self.anchor_pose = None

    def track(self, gray, scale):
        if self.track_gray is None or self.track_points is None or self.track_gray.shape != gray.shape:
            return None
        original = (self.track_points * scale).astype(np.float32).reshape(-1, 1, 2)
        options = dict(winSize=(21, 21), maxLevel=3,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
        moved, status, _ = cv2.calcOpticalFlowPyrLK(self.track_gray, gray, original, None, **options)
        if moved is None:
            return None
        back, backward_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.track_gray, moved, None, **options)
        if back is None:
            return None
        error = np.linalg.norm((back - original).reshape(-1, 2), axis=1)
        good = (status.ravel() > 0) & (backward_status.ravel() > 0) & (error < 1.5)
        if good[16:32].mean() < 0.94 or good.mean() < 0.8:
            return None
        moved = moved.reshape(-1, 2) / scale
        delta = np.median((moved - self.track_points)[good], axis=0)
        moved[~good] = self.track_points[~good] + delta
        return moved, delta

    def detect_dogs(self, rgb):
        size = int(self.detector.input["shape"][1])
        resized, mapping = letterbox(rgb, size)
        self.calls['detector'] += 1
        outputs = self.detector(resized[None])
        # EfficientDet Lite0 v1 has standard TFLite detection-postprocess outputs.
        boxes, classes, scores, count = outputs
        if boxes.ndim != 3 or boxes.shape[-1] != 4:
            raise RuntimeError(f"Unexpected detector outputs: {[x.shape for x in outputs]}")
        detections = []
        for box, cls, score in zip(boxes[0, :int(count[0])], classes[0], scores[0]):
            if int(cls) != self.dog_class or score < self.threshold:
                continue
            y1, x1, y2, x2 = box
            decoded = clip_box(undo_letterbox([x1, y1, x2, y2], mapping, size), rgb.shape)
            if decoded[2] > decoded[0] and decoded[3] > decoded[1]:
                detections.append((decoded, float(score)))
        return detections

    def infer(self, frame, timestamp):
        begin = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        scale = min(1.0, 640 / max(frame.shape[:2]))
        gray = cv2.cvtColor(cv2.resize(frame, (round(frame.shape[1] * scale), round(frame.shape[0] * scale))), cv2.COLOR_BGR2GRAY)
        discontinuity = False
        if timestamp - self.last_detection >= self.period:
            old = self.body_box
            dogs = self.detect_dogs(rgb)
            self.last_detection = timestamp
            if len(dogs) == 1:
                self.body_box, self.body_score = dogs[0]
                self.status = "ok"
                discontinuity = old is not None and box_iou(old, self.body_box) < 0.15
            else:
                self.body_box = None
                self.status = "multiple_dogs" if len(dogs) > 1 else "no_dog"
        detection_ms = (time.perf_counter() - begin) * 1000
        if self.body_box is None:
            self.clear_track()
            return {"valid": False, "reason": self.status, "timings_ms": {"detector": detection_ms}}
        if discontinuity:
            self.clear_track()
        if timestamp < self.retry_at and self.track_points is None:
            return {'valid': False, 'reason': self.last_reason,
                    'discontinuity': discontinuity, 'inference_mode': 'retry_wait',
                    'timings_ms': {'detector': detection_ms}}
        if not discontinuity and timestamp - self.last_landmarks < self.landmark_period:
            tracked = self.track(gray, scale)
            if tracked is not None:
                points, delta = tracked
                box = clip_box(np.asarray(self.previous_face) + np.tile(delta, 2), frame.shape)
                valid, reason, pose = landmark_gate(points, box, self.rotation, self.confidence)
                anchor = self.anchor_pose
                drift = (abs(pose.yaw_cue - anchor.yaw_cue) > .10 or
                         abs(pose.roll_deg - anchor.roll_deg) > 10 or
                         not .80 < pose.eye_distance / max(anchor.eye_distance, 1) < 1.25)
                # Appearance must still support both eyes after tracking.
                # A tracked eye can briefly blink or pass behind fur. Use a
                # looser brightness gate between strict neural anchors.
                if valid and not drift:
                    eye_values = eye_evidence(frame, points)['eye_dark_p20']
                    appearance_ok = (min(eye_values) if abs(pose.yaw_cue) >= .22 else max(eye_values)) <= 165
                else:
                    appearance_ok = False
                if valid and not drift and not appearance_ok:
                    self.track_gray, self.track_points, self.previous_face = gray, points, box
                    return {'valid': False, 'reason': 'eye_visibility_unresolved',
                            'inference_mode': 'optical_flow_rejected',
                            'timings_ms': {'detector': detection_ms,
                                           'tracking': (time.perf_counter() - begin) * 1000 - detection_ms}}
                if valid and not drift and appearance_ok:
                    self.track_gray, self.track_points, self.previous_face = gray, points, box
                    return {"valid": True, "reason": "ok", "box": box, "points": points, 'pose': pose,
                            'rotation_quarters': self.rotation, 'uncertainty': self.confidence,
                            "dog_score": self.body_score, "body_box": self.body_box, "discontinuity": False,
                            "inference_mode": "optical_flow", "timings_ms": {"detector": detection_ms,
                            "tracking": (time.perf_counter() - begin) * 1000 - detection_ms}}
        bx1, by1, bx2, by2 = clip_box(self.body_box, rgb.shape, margin=0.12)
        body = rgb[by1:by2, bx1:bx2]
        local, mapping = letterbox(body, 224)
        self.calls['localizer'] += 1
        result = self.localizer(local[None].astype(np.float32) / 255)[0].reshape(-1)
        # Regressed corners are not ordered by the exported graph. Canonicalize
        # before checking area (verified against the pinned model on CPU).
        rx1, ry1, rx2, ry2 = np.clip(result[:4], 0, 1)
        normalized = [min(rx1, rx2), min(ry1, ry2), max(rx1, rx2), max(ry1, ry2)]
        face = undo_letterbox(normalized, mapping, 224)
        face = (face[0] + bx1, face[1] + by1, face[2] + bx1, face[3] + by1)
        localizer_ms = (time.perf_counter() - begin) * 1000 - detection_ms
        if not np.isfinite(face).all() or face[2] - face[0] < 24 or face[3] - face[1] < 24:
            self.clear_track()
            return {"valid": False, "reason": "no_face", "timings_ms": {"detector": detection_ms, "localizer": localizer_ms}}
        box = clip_box(face, rgb.shape, margin=0.10)
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            self.clear_track()
            return {"valid": False, "reason": "invalid_crop", "timings_ms": {}}
        crop = cv2.resize(rgb[y1:y2, x1:x2], (384, 384), interpolation=cv2.INTER_LINEAR)
        rotation = self.rotation if self.track_points is not None else self.next_rotation
        def hypothesis(k):
            self.calls['landmarker'] += 1
            outputs = self.landmarker(np.rot90(crop, k)[None].astype(np.float32) / 255)
            confidence = uncertainty(outputs[1])
            points = rotate_normalized(outputs[0].reshape(46, 2), -k)
            points = points * [x2 - x1, y2 - y1] + [x1, y1]
            valid, reason, pose = landmark_gate(points, box, k, confidence)
            evidence = eye_evidence(frame, points)
            # A real side turn naturally hides the far eye. Front capture still
            # requires both eyes; side pose may use one visible eye plus the
            # separate heatmap, nose and temporal consistency checks.
            if valid and abs(pose.yaw_cue) >= .22 and min(evidence['eye_dark_p20']) <= 105:
                evidence['accepted'] = True
            confidence.update(evidence)
            if valid and not evidence['accepted']:
                valid, reason = False, 'eye_visibility_unresolved'
            return valid, reason, pose, points, confidence
        valid, reason, pose, points, confidence = hypothesis(rotation)
        # Reacquisition tests alternate orientations only when facial detail
        # is present. A blank/back-of-head crop does not trigger four forwards.
        if not valid and reason != 'eye_visibility_unresolved' and confidence['core_entropy'] < 5.0:
            candidates = []
            for k in range(4):
                if k == rotation:
                    continue
                candidate = hypothesis(k)
                if candidate[0]:
                    candidates.append((candidate[4]['core_entropy'], k, candidate))
            if candidates:
                _, rotation, candidate = min(candidates, key=lambda x: x[0])
                valid, reason, pose, points, confidence = candidate
        if not valid:
            self.clear_track()
            self.retry_at = timestamp + max(.12, self.landmark_period)
            self.last_reason = reason
            # Occlusion does not imply that the camera has rotated.
            self.next_rotation = rotation if reason == 'eye_visibility_unresolved' else (rotation + 1) % 4
            return {'valid': False, 'reason': reason, 'discontinuity': discontinuity,
                    'inference_mode': 'neural_rejected', 'uncertainty': confidence,
                    'rotation_quarters': rotation,
                    'timings_ms': {'detector': detection_ms, 'localizer': localizer_ms,
                                   'landmarks': (time.perf_counter() - begin) * 1000 - detection_ms - localizer_ms}}
        if self.previous_face is not None and box_iou(self.previous_face, box) < 0.12:
            discontinuity = True
        self.previous_face = box
        discontinuity = discontinuity or self.rotation != rotation
        self.rotation = self.next_rotation = rotation
        self.confidence = confidence
        self.anchor_pose = pose
        self.track_gray, self.track_points = gray, points
        self.last_landmarks = timestamp
        elapsed = (time.perf_counter() - begin) * 1000
        return {"valid": True, "reason": "ok", "box": box, "points": points, 'pose': pose,
                'rotation_quarters': rotation, 'uncertainty': confidence,
                "dog_score": self.body_score, "body_box": self.body_box,
                "discontinuity": discontinuity, "inference_mode": "neural",
                "timings_ms": {"detector": detection_ms, "localizer": localizer_ms,
                               "landmarks": elapsed - detection_ms - localizer_ms}}

    def verify_candidate(self, frame, record):
        """Check only a potential winning frame, without changing live tracking.

        Bypass detector/localizer: its accepted face ROI is already available.
        The original candidate's pixels, not the later confirmation frame, run
        through the landmarker. This prevents a sharp flow drift from winning.
        """
        box, rotation = record['box'], record.get('rotation_quarters', 0)
        x1, y1, x2, y2 = box
        crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        crop = cv2.resize(crop, (384, 384), interpolation=cv2.INTER_LINEAR)
        self.calls['candidate_verification'] += 1
        self.calls['landmarker'] += 1
        outputs = self.landmarker(np.rot90(crop, rotation)[None].astype(np.float32) / 255)
        confidence = uncertainty(outputs[1])
        points = rotate_normalized(outputs[0].reshape(46, 2), -rotation)
        points = points * [x2 - x1, y2 - y1] + [x1, y1]
        valid, reason, pose = landmark_gate(points, box, rotation, confidence)
        evidence = eye_evidence(frame, points)
        if valid and abs(pose.yaw_cue) >= .22 and min(evidence['eye_dark_p20']) <= 105:
            evidence['accepted'] = True
        confidence.update(evidence)
        if valid and not evidence['accepted']:
            valid, reason = False, 'eye_visibility_unresolved'
        return {'valid': valid, 'reason': reason, 'pose': pose, 'points': points,
                'box': box, 'rotation_quarters': rotation, 'uncertainty': confidence,
                'inference_mode': 'candidate_neural', 'timings_ms': {}}
