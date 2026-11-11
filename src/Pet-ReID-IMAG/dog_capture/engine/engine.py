"""CaptureEngine: detection -> species gate -> tracking -> pose -> quality -> best shot.

The same causal pipeline as a face-capture SDK:

1. a one-stage face detector returns boxes + keypoints for every face, and each
   face is re-examined on a canonical upright crop (refine.py: scale + quarter turn);
2. an optional dog-body detector rejects human/cat faces (species gate);
3. ByteTrack gives each dog a stable track ID (several dogs no longer void a frame);
4. per track: keypoint geometry -> debounced view -> aligned-crop quality;
5. per track and view: best-shot policy (fast / best / hybrid);
6. guidance for the operator from the primary (largest, sticky) dog.

``process()`` never touches disk or network and holds bounded memory.  Call it
from a single worker with capture timestamps.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
import math
import time

import cv2
import numpy as np

from .alignment import align_face, load_template
from .config import CaptureConfig
from .guidance import guidance
from .pose import ViewClassifier, face_geometry
from .quality import QualityAssessor, QualityReport
from .refine import FaceRefiner, eye_contrast, transform_detection, upright_frame
from .schema import Detection, box_iou_matrix
from .selection import Shot, TrackShots
from .tracking import ByteTracker, Track

ROTATION_SEARCH = (0, 90, 270, 180)


@dataclass
class FaceResult:
    track_id: int
    detection: Detection
    geometry: object
    view_state: object
    quality: QualityReport | None
    accepted: bool = False          # entered a slot or the pending buffer
    speed_iod_per_s: float = 0.0

    def to_dict(self):
        return {'track_id': self.track_id, 'detection': self.detection.to_dict(),
                'pose': self.geometry.to_dict() if self.geometry else None,
                'view': self.view_state.to_dict() if self.view_state else None,
                'quality': self.quality.to_dict() if self.quality else None,
                'accepted': self.accepted, 'speed_iod_per_s': round(self.speed_iod_per_s, 3)}


@dataclass
class FrameResult:
    frame_index: int
    time_s: float
    rotation_deg: int
    mirrored: bool
    faces: list
    primary_track: int | None
    guidance: dict
    captured_views: list
    missing_views: list
    events: list = field(default_factory=list)
    rejected: dict = field(default_factory=dict)
    timings_ms: dict = field(default_factory=dict)

    def to_dict(self):
        return {'frame_index': self.frame_index, 'time_s': round(self.time_s, 6),
                'rotation_deg': self.rotation_deg, 'mirrored': self.mirrored, 'primary_track': self.primary_track,
                'guidance': self.guidance, 'captured_views': self.captured_views,
                'missing_views': self.missing_views, 'events': self.events, 'rejected': self.rejected,
                'faces': [f.to_dict() for f in self.faces],
                'timings_ms': {k: round(v, 3) for k, v in self.timings_ms.items()}}


class OrientationLock:
    """Find the frame rotation at which faces are upright, cheaply and stably.

    Most videos are already upright (OpenCV applies container rotation
    metadata), so the current rotation is always analysed.  Only after
    ``probe_after_s`` without a face does one frame in ``probe_every`` try
    another quarter turn; two successful probes switch to it.  A confident
    face whose eye line is near-vertical gives the correction directly
    (keypoints know where "up" is).  Two upright detections lock the
    rotation; ``relock_after_s`` without a face unlocks it.
    """

    def __init__(self, mode, probe_every=10, probe_after_s=1.0, relock_after_s=3.0):
        self.fixed = None if str(mode) == 'auto' else int(mode)
        self.rotation = self.fixed or 0
        self.locked = self.fixed is not None
        self.probe_every, self.probe_after, self.relock_after = probe_every, probe_after_s, relock_after_s
        self.frame = 0
        self.probe_index = 0
        self.start = self.last_face = None
        self.hits = self.sideways = 0
        self.probe_hits = {}

    def next(self, timestamp):
        """(rotation, is_probe) to use for this frame."""
        self.frame += 1
        if self.fixed is not None:
            return self.rotation, False
        if self.start is None:
            self.start = timestamp
        since = timestamp - (self.last_face if self.last_face is not None else self.start)
        if self.locked and since <= self.relock_after:
            return self.rotation, False
        if since >= self.probe_after and self.frame % self.probe_every == 0:
            others = [r for r in ROTATION_SEARCH if r != self.rotation]
            rotation = others[self.probe_index % len(others)]
            self.probe_index += 1
            return rotation, True
        return self.rotation, False

    def report(self, rotation, probe, found, timestamp, roll_deg=None):
        """Feed back whether an upright confident face was found; True if the base rotation changed."""
        if self.fixed is not None:
            return False
        if probe:
            if found:
                self.probe_hits[rotation] = self.probe_hits.get(rotation, 0) + 1
                if self.probe_hits[rotation] >= 2:
                    self.rotation, self.locked, self.hits, self.probe_hits = rotation, False, 0, {}
                    self.last_face = timestamp
                    return True
            return False
        if not found:
            self.sideways = 0
            if self.locked and self.last_face is not None and timestamp - self.last_face > self.relock_after:
                self.locked, self.hits = False, 0
            return False
        self.last_face = timestamp
        quarter = round(roll_deg / 90.0) % 4 if roll_deg is not None and abs(roll_deg) > 65 else 0
        if quarter and not self.locked:
            self.sideways += 1
            if self.sideways >= 2:  # sustained sideways face: rotate it upright
                self.rotation = (self.rotation - 90 * quarter) % 360
                self.hits = self.sideways = 0
                return True
            return False
        self.sideways, self.probe_hits = 0, {}
        if not self.locked:
            self.hits += 1
            self.locked = self.hits >= 2
        return False


def _place(det):
    x1, y1, x2, y2 = (float(v) for v in det.box)
    return (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1)


def rotate_frame(frame, degrees_cw):
    k = (degrees_cw // 90) % 4
    return frame if k == 0 else np.ascontiguousarray(np.rot90(frame, -k))


def as_bgr(frame):
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        raise ValueError(f'Expected uint8 pixels, got {frame.dtype}')
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 16:
        raise ValueError(f'Expected an HxWx3 BGR frame, got shape {frame.shape}')
    return frame


class _TrackContext:
    def __init__(self, track_id, engine):
        still = engine.config.runtime.still_images
        self.classifier = ViewClassifier(engine.config.pose, still=still)
        # A photo set has no time axis: fast/hybrid locking would ignore later, better photos.
        selection = replace(engine.config.selection, mode='best') if still else engine.config.selection
        self.shots = TrackShots(track_id, engine.view_names, selection,
                                {v.name: v.shots for v in engine.config.pose.views})



class CaptureEngine:
    def __init__(self, config: CaptureConfig | None = None, detector=None, body_gate=None):
        self.config = config or CaptureConfig()
        self.config.validate()
        self.detector = detector
        self.body_gate = body_gate
        # Crop refinement needs a model that can be run on an image (not the legacy DogFLW cascade).
        refinable = detector is not None and hasattr(detector, 'size') and self.config.refine.enabled
        self.refiner = FaceRefiner(detector, self.config.refine) if refinable else None
        self.full_detected = -math.inf
        self.full_detected_frame = -10 ** 9
        self.views = {v.name: v for v in self.config.pose.views}
        self.view_names = tuple(self.views)
        self.required_views = frozenset(v.name for v in self.config.pose.views if v.required)
        self.template = load_template(self.config.output.template)
        self.assessor = QualityAssessor(self.config.quality, self.template)
        self.tracker = ByteTracker(self.config.tracker)
        self.orientation = OrientationLock(self.config.runtime.rotate)
        self.contexts: dict[int, _TrackContext] = {}
        self.finished: list[TrackShots] = []
        self.events = deque(maxlen=1024)
        self.previous_time = None
        self.primary_id = None
        self.frames = 0
        self.dropped = {'non_increasing_timestamp': 0}
        self.body_boxes = np.zeros((0, 4), np.float32)
        self.body_scores = np.zeros((0,), np.float32)
        self.body_time = -math.inf
        self.body_checked = -math.inf
        self.roi_checked = -math.inf
        self._external = False
        self._timings = {}
        # Per view: the best credible candidate that missed only the quality level (for review).
        self.near_misses: dict[str, dict] = {}
        self.gate_history = deque()   # (time, cx, cy, size, 'pass'|'fail'|'present') per analysed face
        self.frame_times = deque()    # times of processed frames
        self._gate_status = {}        # id(detection) -> this frame's check result
        self.guidance_history = deque()
        self.shown_code = None

    # ---- helpers ------------------------------------------------------
    def _context(self, track_id):
        if track_id not in self.contexts:
            self.contexts[track_id] = _TrackContext(track_id, self)
        return self.contexts[track_id]

    def _archive(self, track_id, timestamp):
        context = self.contexts.pop(track_id, None)
        if context is None:
            return
        context.shots.finalize(timestamp)
        if context.shots.captured():
            self.finished.append(context.shots)
            self.finished.sort(key=lambda s: s.rank(), reverse=True)
            del self.finished[4:]
            self.events.append({'event': 'track_finished', 'track_id': track_id, 'time_s': timestamp,
                                'views': context.shots.captured()})

    def _refresh_bodies(self, frame, timestamp, timings):
        g = self.config.body_gate
        if timestamp - self.body_checked >= g.period_s or timestamp < self.body_checked:
            started = time.perf_counter()
            boxes, scores = self.body_gate.detect(frame)
            timings['body_gate'] = timings.get('body_gate', 0) + (time.perf_counter() - started) * 1000
            self.body_checked = timestamp
            if len(boxes):
                self.body_boxes, self.body_scores, self.body_time = boxes, scores, timestamp

    def _check_dog(self, frame, det, timestamp, timings):
        """True if the face belongs to a dog: inside a recent dog-body box, or a
        dog is detected in an expanded head crop (close-ups often show no body)."""
        g = self.config.body_gate
        self._refresh_bodies(frame, timestamp, timings)
        cx, cy = (det.box[0] + det.box[2]) / 2, (det.box[1] + det.box[3]) / 2
        if timestamp - self.body_time <= g.hold_s:
            for box, score in zip(self.body_boxes, self.body_scores):
                mx, my = (box[2] - box[0]) * g.margin, (box[3] - box[1]) * g.margin
                if box[0] - mx <= cx <= box[2] + mx and box[1] - my <= cy <= box[3] + my:
                    det.body_box, det.body_score = box, float(score)
                    return True
        h, w = frame.shape[:2]
        size = max(det.width, det.height) * 1.5
        x1, y1 = int(max(0, cx - size)), int(max(0, cy - size))
        x2, y2 = int(min(w, cx + size)), int(min(h, cy + size))
        if x2 - x1 < 32 or y2 - y1 < 32:
            return False
        started = time.perf_counter()
        boxes, scores = self.body_gate.detect(frame[y1:y2, x1:x2])
        timings['head_gate'] = timings.get('head_gate', 0) + (time.perf_counter() - started) * 1000
        for box, score in zip(boxes, scores):
            if score >= g.head_score_threshold and box[0] <= cx - x1 <= box[2] and box[1] <= cy - y1 <= box[3]:
                det.body_box, det.body_score = box + np.array([x1, y1, x1, y1], np.float32), float(score)
                return True
        return False

    def _roi_zoom(self, frame, timestamp, timings):
        """Two-stage fallback for distant dogs (face SDK "detect in ROI").

        No face in the full frame, but the body gate sees a dog: run the face
        detector once more on a square crop around the largest dog, which
        magnifies a small face.  Costs one extra detector call per frame only
        while a dog is visible without a detectable face.
        """
        d = self.config.detector
        if not d.roi_zoom or self.body_gate is None or self.detector is None:
            return []
        if timestamp - self.roi_checked < d.roi_period_s and timestamp >= self.roi_checked:
            return []
        self.roi_checked = timestamp
        self._refresh_bodies(frame, timestamp, timings)
        if timestamp - self.body_time > self.config.body_gate.hold_s or not len(self.body_boxes):
            return []
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(self.body_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        side = max(x2 - x1, y2 - y1) * 1.15
        if max(h, w) / max(side, 1) < d.roi_min_zoom:
            return []  # the dog already fills the frame: a crop would not magnify it
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        left, top = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        right, bottom = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
        if right - left < 32 or bottom - top < 32:
            return []
        started = time.perf_counter()
        found = self.detector.detect(frame[top:bottom, left:right])
        timings['roi_detector'] = (time.perf_counter() - started) * 1000
        offset = np.array([left, top], np.float32)
        kept = []
        for det in found:
            det.box = det.box + np.tile(offset, 2)
            det.keypoints = det.keypoints + offset
            det.source = 'roi_zoom'
            if det.validate() is None:
                kept.append(det)
        return kept

    def _dog_seen(self, timestamp):
        return (self.body_gate is not None and len(self.body_boxes) > 0
                and timestamp - self.body_time <= self.config.body_gate.hold_s)

    def _track_is_dog(self, frame, track, timestamp, timings):
        """Species verdict per *track*, not per frame (stable and cheap).

        A confirmed track stays a dog for its lifetime -- ByteTrack only links
        overlapping boxes, so a track cannot silently become a person.  An
        unconfirmed track is re-checked every ``period_s``; the gate model runs
        only while such tracks exist.
        """
        if self.body_gate is None or getattr(track, 'is_dog', False):
            return True
        if timestamp < getattr(track, 'dog_retry', -math.inf):
            return False
        track.is_dog = self._check_dog(frame, track.detection, timestamp, timings)
        if not track.is_dog:
            track.dog_retry = timestamp + self.config.body_gate.period_s
        return track.is_dog

    def _species_gate(self, frame, detections, timestamp, timings):
        """Photo mode (no tracks): check every face of the image."""
        if self.body_gate is None or not detections:
            return detections, 0
        kept = [d for d in detections if self._check_dog(frame, d, timestamp, timings)]
        return kept, len(detections) - len(kept)

    def _choose_primary(self, tracks):
        def size(t):
            return t.detection.width * t.detection.height * max(t.score, 0.05)
        current = next((t for t in tracks if t.id == self.primary_id), None)
        best = max(tracks, key=size, default=None)
        if current is None and self.primary_id is not None:
            # A one-frame miss must not hand guidance to another dog.
            lost = next((t for t in self.tracker.tracks if t.id == self.primary_id), None)
            if lost is not None and lost.state == Track.LOST and self.previous_time - lost.last_time < 0.5:
                return
        if best is None:
            self.primary_id = None
            return
        if current is None or size(best) > 1.5 * size(current):
            self.primary_id = best.id

    # ---- main API -----------------------------------------------------
    def process(self, frame, timestamp, frame_index=None, detections=None):
        """Analyse one frame; returns FrameResult, or None if the frame was dropped.

        ``detections`` (list of Detection or dicts) bypasses the detector for
        integration with an upstream model; they must use this frame's
        coordinates, and rotation/mirroring are then not applied.
        """
        started = time.perf_counter()
        timestamp = float(timestamp)
        if not math.isfinite(timestamp) or (self.previous_time is not None and timestamp <= self.previous_time):
            if self.config.runtime.timestamps == 'raise':
                raise ValueError('Capture timestamps must be finite and strictly increasing')
            self.dropped['non_increasing_timestamp'] += 1
            return None
        frame = as_bgr(frame)
        index = self.frames if frame_index is None else int(frame_index)
        self.previous_time = timestamp
        self.frames += 1
        timings = {}
        self._timings = timings
        rejected = {}
        skipped = False
        external = self._external = detections is not None
        window, keep = self.config.refine.gate_window_s, self.config.refine.gate_min_frames
        self.frame_times.append(timestamp)
        while len(self.frame_times) > keep and timestamp - self.frame_times[0] > window:
            self.frame_times.popleft()
        while self.gate_history and self.gate_history[0][0] < self.frame_times[0]:
            self.gate_history.popleft()
        if external:
            rotation, probe = 0, False
        elif self.config.runtime.still_images:  # photos are searched per image instead
            rotation, probe = self.orientation.fixed or 0, False
        else:
            rotation, probe = self.orientation.next(timestamp)
        work = rotate_frame(frame, rotation)
        if self.config.runtime.mirror and not external:
            work = np.ascontiguousarray(work[:, ::-1])
        t0 = time.perf_counter()
        if external:
            found = []
            for item in detections:
                try:
                    found.append(item if isinstance(item, Detection) else Detection.from_dict(item))
                except (TypeError, ValueError, KeyError):
                    rejected['malformed_detection'] = rejected.get('malformed_detection', 0) + 1
        elif self.detector is None:
            raise RuntimeError('No detector configured; pass detections= for upstream observations')
        elif self._skip_full_detection(timestamp, probe):
            found = []   # edge mode: faces are followed on their crops (refine.full_detect_period_s)
            skipped = True
        else:
            found = self.detector.detect(work)
            self.full_detected, self.full_detected_frame = timestamp, self.frames
            timings['detector'] = (time.perf_counter() - t0) * 1000
        valid = []
        for det in found:
            reason = det.validate()
            if reason:
                rejected[reason] = rejected.get(reason, 0) + 1
            else:
                valid.append(det)
        if probe:  # orientation probe: detection only, no tracking or species gate in these coordinates
            return self._probe(valid, rotation, timestamp, index, rejected, timings, started)
        followed = self.refiner is not None and not external and any(
            r['misses'] < self.config.refine.carry_frames and r['score'] >= self.config.tracker.high_score
            for r in self.refiner.rois)
        if not external and not followed and not any(d.score >= self.config.tracker.high_score for d in valid):
            # Only weak or unrelated faces (e.g. a person): look for the dog's face up close.
            valid = valid + self._roi_zoom(work, timestamp, timings)
        if self.config.runtime.still_images:
            any_face = bool(valid)
            valid, not_dog = self._species_gate(work, valid, timestamp, timings)
            if not_dog:
                rejected['not_dog'] = not_dog
            return self._process_still(work, valid, timestamp, index, rotation, rejected, timings, started,
                                       search=not any_face and not external)
        if self.refiner is not None and not external:
            t0 = time.perf_counter()
            valid = self.refiner.refine(work, valid, timestamp, full_ran=not skipped)
            timings['refine'] = (time.perf_counter() - t0) * 1000
        found, roll = self._upright_evidence(valid)
        if not external and self.orientation.report(rotation, False, found, timestamp, roll):
            for track in self.tracker.tracks:  # coordinates change with rotation
                self._archive(track.id, timestamp)
            self.tracker.reset()
            if self.refiner is not None:
                self.refiner.reset()
            self.body_time = self.body_checked = -math.inf
        t0 = time.perf_counter()
        tracks = self.tracker.update(valid, timestamp)
        for track in self.tracker.removed:
            self._archive(track.id, timestamp)
        timings['tracking'] = (time.perf_counter() - t0) * 1000
        dogs = [t for t in tracks if self._track_is_dog(work, t, timestamp, timings)]
        if len(dogs) < len(tracks):
            rejected['not_dog'] = len(tracks) - len(dogs)
        tracks = dogs
        if not tracks and self.body_gate is not None:
            self._refresh_bodies(work, timestamp, timings)  # keeps "dog seen, face not found" current
        tracks = sorted(tracks, key=lambda t: -t.detection.width * t.detection.height)
        tracks = tracks[:self.config.detector.max_faces]
        self._choose_primary(tracks)
        t0 = time.perf_counter()
        faces, events = [], []
        for track in tracks:
            face, new_events = self._analyse(work, track, timestamp, index, rotation)
            faces.append(face)
            events += new_events
        timings['analysis'] = (time.perf_counter() - t0) * 1000
        return self._result(faces, events, index, timestamp, rotation, rejected, timings, started)

    def _skip_full_detection(self, timestamp, probe):
        period = self.config.refine.full_detect_period_s
        if self.refiner is None or period <= 0 or probe or self.config.runtime.still_images:
            return False
        live = any(r['misses'] == 0 and r['score'] >= self.config.tracker.high_score for r in self.refiner.rois)
        recent = self.frames - self.full_detected_frame < self.config.refine.full_detect_min_frames
        return live and (0 <= timestamp - self.full_detected < period or recent)

    def _upright_evidence(self, detections):
        confident = [d for d in detections if d.score >= self.config.tracker.new_track_score]
        if not confident:
            return False, None
        best = max(confident, key=lambda d: d.width * d.height)
        return True, face_geometry(best, self.config.pose).roll_deg

    def _probe(self, detections, rotation, timestamp, index, rejected, timings, started):
        found, roll = self._upright_evidence(detections)
        upright = found and roll is not None and abs(roll) <= 45
        if self.orientation.report(rotation, True, upright, timestamp, roll):
            for track in self.tracker.tracks:  # coordinates change with rotation
                self._archive(track.id, timestamp)
            self.tracker.reset()
            if self.refiner is not None:
                self.refiner.reset()
            self.body_time = self.body_checked = -math.inf
        timings['orientation_probe'] = 1.0
        # Report the base rotation: a probe frame shows nothing and must not look rotated.
        return self._result([], [], index, timestamp, self.orientation.rotation, rejected, timings, started)

    def _analyse(self, frame, track, timestamp, index, rotation, still=False):
        context = self._context(track.id)
        det = track.detection
        geometry = face_geometry(det, self.config.pose)
        state = context.classifier.update(timestamp, geometry)
        events = context.shots.observe(state, timestamp)
        target = self.views.get(state.raw)
        speed = 0.0 if still else float(np.linalg.norm(track.velocity(self.config.tracker.nominal_fps)))
        in_band = state.view is not None and state.raw == state.view and state.view not in context.shots.locked
        # Pixel metrics are needed to select (in-band faces) or to guide (the primary dog).
        full = in_band or track.id == self.primary_id
        quality, aligned = self.assessor.assess(frame, det, geometry, target, speed, full=full)
        face = FaceResult(track.id, det, geometry, state, quality,
                          speed_iod_per_s=speed / max(geometry.iod, 1e-6))
        eligible = in_band and quality.passed and context.shots.eligible_level(quality.level)
        if in_band and quality.passed and not eligible:
            self._near_miss(context, frame, det, target, state.view, quality, track.id, index, timestamp, still)
        if eligible and context.shots.wants(state.view, quality.score):
            # Landmarks first, sharpness second: a confident but hallucinated face (the model on a
            # dog lying sideways with closed eyes) must not win on texture or exposure.
            problem = self._trusted(context, frame, det, target, timestamp, still)
            if problem:
                quality.reasons.append(problem)
                quality.passed, eligible = False, False
        if eligible and context.shots.wants(state.view, quality.score):
            size = self.config.output.aligned_size
            crop = aligned.image if size == aligned.image.shape[0] else align_face(
                frame, geometry.points, self.template[0], self.template[1], size).image
            # The kept frame is turned (losslessly) so the face is upright for the recognizer.
            k = geometry.upright_k
            kept, shot_det, shot_geometry = frame, det, geometry
            if k:
                kept, matrix = upright_frame(frame, k)
                shot_det = transform_detection(det, matrix)
                shot_det.extra['upright_k'] = 0
                shot_geometry = face_geometry(shot_det, self.config.pose)
            shot = Shot(state.view, track.id, index, timestamp, quality.score, quality, shot_geometry, shot_det,
                        crop.copy(), kept.copy() if self.config.output.keep_frames else None,
                        (rotation + 90 * k) % 360, 'still_image' if still else 'stable_view')
            new, face.accepted = context.shots.offer(shot, state, timestamp)
            events += new
        self.events.extend(events)
        if self.refiner is not None and not still and det.source != 'external':
            self.gate_history.append((timestamp, *_place(det), self._gate_status.pop(id(det), 'present')))
        return face, events

    def _unreliable(self, frame, det, target=None):
        """Reason code if this face's landmarks cannot be trusted for a shot, else None.

        Cheapest first: refined at all and its crop score; image evidence of open eyes;
        then an open-loop second look on the nearest quarter-turn crop (two model runs):
        score, eye confidence for the target view, mirror test, agreement.
        """
        if self.refiner is None or det.source == 'external':
            return None
        f = self.config.refine
        if not det.extra.get('refined') or det.score < f.min_shot_score:
            return 'landmarks_unreliable'
        front = target is not None and target.lower < 0 < target.upper
        pick = min if front else max      # frontal: both eyes; side: the visible one
        started = time.perf_counter()
        contrast = eye_contrast(frame, det)
        det.extra['eye_contrast'] = contrast
        if pick(contrast) < f.min_eye_contrast:
            self._timing('shot_check', started)
            return 'eyes_not_visible'
        check = self.refiner.verify(frame, det)
        det.extra['verify'] = check
        self._timing('shot_check', started)
        if (check['score'] < f.min_shot_score
                or pick(check['eyes']) < (f.min_front_eye_conf if front else f.min_side_eye_conf)
                or not check['mirror'] <= (f.max_mirror_disagreement if front else f.max_side_mirror_disagreement)
                or not check['agreement'] <= (f.max_agreement if front else f.max_side_agreement)):
            return 'landmarks_unreliable'
        return None

    def _trusted(self, context, frame, det, target, timestamp, still=False):
        """Shot gate: this frame's landmark checks, and (video) a consistent recent history."""
        problem = self._unreliable(frame, det, target)
        if still or self.refiner is None or det.source == 'external':
            return problem
        f = self.config.refine
        self._gate_status[id(det)] = 'fail' if problem else 'pass'
        if problem:
            return problem
        cx, cy, size = _place(det)
        here = [e for e in self.gate_history if math.hypot(e[1] - cx, e[2] - cy) <= 0.5 * max(e[3], size)]
        passes = 1 + sum(1 for e in here if e[4] == 'pass')
        fails = sum(1 for e in here if e[4] == 'fail')
        missing = max(0, len(self.frame_times) - 1 - len({e[0] for e in here}))   # frames with no face here
        if passes < f.gate_min_passes or passes < f.gate_min_ratio * (passes + fails + missing):
            return 'landmarks_unstable'
        return None

    def _near_miss(self, context, frame, det, target, view, quality, track_id, index, timestamp, still=False):
        best = self.near_misses.get(view)
        if best is not None and best['score'] >= quality.score:
            return
        if self._trusted(context, frame, det, target, timestamp, still):
            return   # not a credible face: never shown as "almost"
        self.near_misses[view] = {'view': view, 'frame_index': index, 'time_s': round(timestamp, 4),
                                  'track_id': track_id, 'score': round(quality.score, 4), 'level': quality.level,
                                  'needed': self.config.selection.min_level,
                                  'weakest': min(quality.factors, key=quality.factors.get) if quality.factors else None}

    def _timing(self, name, started):
        self._timings[name] = self._timings.get(name, 0) + (time.perf_counter() - started) * 1000

    def _process_still(self, frame, detections, timestamp, index, rotation, rejected, timings, started,
                       search=False):
        """Photo-album mode: largest dog face per image, one subject, no dwell."""
        if search and self.orientation.fixed is None and self.detector is not None:
            for degrees in ROTATION_SEARCH[1:]:  # no face at all: the photo may be sideways
                rotated = rotate_frame(frame, degrees)
                found = [d for d in self.detector.detect(rotated) if d.validate() is None]
                self.body_checked = -math.inf  # body boxes are per orientation
                found, not_dog = self._species_gate(rotated, found, timestamp, timings)
                if not_dog:
                    rejected['not_dog'] = rejected.get('not_dog', 0) + not_dog
                if found:
                    frame, detections, rotation = rotated, found, degrees
                    break
        faces, events = [], []
        if detections and self.refiner is not None and not self._external:
            detections = self.refiner.refine(frame, detections, timestamp, still=True)
        if detections:
            det = max(detections, key=lambda d: d.width * d.height * d.score)
            track = Track(0, det, timestamp, self.tracker.kalman)
            self.primary_id = 0
            face, events = self._analyse(frame, track, timestamp, index, rotation, still=True)
            faces.append(face)
        return self._result(faces, events, index, timestamp, rotation, rejected, timings, started)

    def _result(self, faces, events, index, timestamp, rotation, rejected, timings, started):
        primary = next((f for f in faces if f.track_id == self.primary_id), None)
        context = self.contexts.get(self.primary_id)
        captured = context.shots.captured() if context else []
        missing = [v for v in self.view_names if v not in captured]
        # Guide towards views that can still change (a captured view keeps improving
        # until it locks), required ones first; "done" only once nothing is outstanding.
        targets = (context.shots.outstanding(self.required_views) if context
                   else sorted(self.view_names, key=lambda v: v not in self.required_views))
        advice = guidance(primary, targets, rejected_species=bool(rejected.get('not_dog')),
                          speed_iod=primary.speed_iod_per_s if primary else 0.0,
                          motion_ref=self.config.quality.motion_ref_iod_per_s,
                          dog_seen=primary is None and self._dog_seen(timestamp))
        advice = self._debounce(advice, timestamp)
        timings['total'] = (time.perf_counter() - started) * 1000
        mirrored = bool(self.config.runtime.mirror and not self._external)
        return FrameResult(index, timestamp, rotation, mirrored, faces, self.primary_id, advice, captured, missing,
                           events, rejected, timings)

    def _debounce(self, advice, timestamp, window_s=0.6):
        """Majority vote over the last ``window_s``: prompts must not flicker.

        'done' and a view that just became capturable are passed through at
        once; everything else must dominate the recent window to be shown.
        """
        self.guidance_history.append((timestamp, advice))
        while self.guidance_history and timestamp - self.guidance_history[0][0] > window_s:
            self.guidance_history.popleft()
        if advice['code'] in ('done', 'hold_still') or self.config.runtime.still_images:
            self.shown_code = advice['code']
            return advice
        counts = {}
        for _, item in self.guidance_history:
            counts[item['code']] = counts.get(item['code'], 0) + 1
        # Ties keep the prompt already on screen; a new prompt must out-vote it.
        code = max(counts, key=lambda k: (counts[k], k == self.shown_code, k == advice['code']))
        self.shown_code = code
        return next(item for _, item in reversed(self.guidance_history) if item['code'] == code)

    # ---- server-side verification ---------------------------------------
    def verify_shot(self, frame, claimed_box=None, claimed_view=None, exit_margin=None):
        """Independently re-check one uploaded frame (no tracking, no dwell).

        Re-detects faces, keeps dog faces, picks the one overlapping the
        client's claimed box (else the largest), and recomputes geometry, view
        band and quality with this engine's configuration.  Returns a dict
        with ``ok`` and machine-readable ``reasons``.
        """
        frame = as_bgr(frame)
        timings = {}
        if self.detector is None:
            raise RuntimeError('verify_shot needs a face detector')
        found = [d for d in self.detector.detect(frame) if d.validate() is None]
        if found and self.refiner is not None:
            found = self.refiner.refine(frame, found, 0.0, still=True)
        reasons = []
        if not found:
            return {'ok': False, 'reasons': ['no_face'], 'faces': 0}
        if self.body_gate is not None:
            # Every upload is independent: a body box from the preceding JPEG
            # must not authorise a face in this one when its gate finds no dog.
            self.body_time = self.body_checked = -math.inf
            self.body_boxes = np.zeros((0, 4), np.float32)
            self.body_scores = np.zeros((0,), np.float32)
            found = [d for d in found if self._check_dog(frame, d, self.previous_time or 0.0, timings)]
            if not found:
                return {'ok': False, 'reasons': ['not_dog'], 'faces': 0}
        iou = 0.0
        if claimed_box is not None:
            overlaps = box_iou_matrix(np.asarray(claimed_box, np.float32)[None], np.stack([d.box for d in found]))[0]
            best = int(overlaps.argmax())
            iou = float(overlaps[best])
            if iou < 0.3:
                reasons.append('claimed_face_not_found')
            det = found[best]
        else:
            det = max(found, key=lambda d: d.width * d.height * d.score)
        geometry = face_geometry(det, self.config.pose)
        margin = self.config.pose.exit_margin if exit_margin is None else exit_margin
        view = None
        for spec in self.config.pose.views:
            if (spec.lower - margin <= geometry.yaw <= spec.upper + margin
                    and abs(geometry.roll_deg) <= spec.max_roll_deg):
                view = spec.name
                break
        target = self.views.get(claimed_view) if claimed_view else self.views.get(view)
        quality, _ = self.assessor.assess(frame, det, geometry, target, 0.0)
        reasons += quality.reasons
        self._timings = timings
        problem = self._unreliable(frame, det, target) if quality.passed else None
        if problem:
            reasons.append(problem)
        if claimed_view is not None and view != claimed_view:
            reasons.append('view_mismatch')
        if quality.passed and not TrackShots.level_ok(quality.level, self.config.selection.min_level):
            reasons.append('quality_too_low')
        return {'ok': not reasons, 'reasons': reasons, 'faces': len(found), 'box_iou': iou, 'view': view,
                'detection': det.to_dict(), 'pose': geometry.to_dict(), 'quality': quality.to_dict()}

    # ---- results ------------------------------------------------------
    def bundles(self):
        """All per-track results (active and finished), best first."""
        items = [c.shots for c in self.contexts.values() if c.shots.captured()] + list(self.finished)
        unique = {id(s): s for s in items}
        return sorted(unique.values(), key=lambda s: s.rank(), reverse=True)

    def complete(self):
        """True once the primary dog has every view and no view can still improve.

        fast/hybrid: every view is locked; best: every view is captured (it
        never locks, so "complete" means coverage).
        """
        context = self.contexts.get(self.primary_id)
        if context is None:
            return False
        shots = context.shots
        required = [v.name for v in self.config.pose.views if v.required]
        if not all(shots.full(v) for v in required):
            return False
        return self.config.selection.mode == 'best' or all(v in shots.locked for v in required)

    def best_bundle(self):
        bundles = self.bundles()
        return bundles[0] if bundles else None

    def finalize(self):
        """End of capture: drop unconfirmed pending shots, return the best bundle."""
        timestamp = self.previous_time
        for context in self.contexts.values():
            context.shots.finalize(timestamp)
        return self.best_bundle()

    def describe(self):
        return {'config': self.config.to_dict(),
                'detector': self.detector.describe() if self.detector is not None else {'kind': 'external'},
                'body_gate': self.body_gate.describe() if self.body_gate is not None else None,
                'orientation': {'mode': self.config.runtime.rotate, 'rotation_deg': self.orientation.rotation,
                                'locked': self.orientation.locked},
                'frames': self.frames, 'dropped': dict(self.dropped)}
