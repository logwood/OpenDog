"""Second-stage keypoints on a canonical, upright face crop.

The face models are trained on upright dogs whose faces cover roughly a third
of the picture.  A dog lying down presents its face rolled by 90 degrees, and a
phone held close fills the frame with it; on such input the model still
reports a face, with confident but wrong keypoints (validation, 400 faces:
upright 75 % within 0.25 IOD, rotated by a quarter turn 0 %).

So, as face trackers do (detector + landmark model on a normalised region of
interest, MediaPipe-style), every face box is re-examined on a square crop in
which the face covers ``share`` of the side, rotated by the quarter turn that
makes the face upright.  The quarter turn is chosen by the highest detector
score over the four orientations (96 % correct on validation), searched when a
face appears and every ``search_period_s``, and kept with a switching margin.
Crops also carry a face across frames where the full-frame detector misses it.

Keypoints are mapped back to frame coordinates; ``extra['upright_angle']`` is the
clockwise turn that makes the face upright and ``extra['upright_k']`` the nearest
quarter turn (used to store the shot frame upright, losslessly).  ``consistency``
re-runs the model on the mirrored crop: real faces give the same landmarks,
hallucinated ones usually do not.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .config import RefineConfig
from .schema import Detection, box_iou_matrix


def rotation_matrix(angle_deg, size):
    """3x3: a size x size image turned angle_deg clockwise about its centre (90 = cv2.ROTATE_90_CLOCKWISE)."""
    a = math.radians(angle_deg)
    c, s = round(math.cos(a), 12), round(math.sin(a), 12)
    o = (size - 1) / 2
    return np.array([[c, -s, o - c * o + s * o], [s, c, o - s * o - c * o], [0.0, 0.0, 1.0]])


def crop_matrix(box, angle_deg, share, size):
    """3x3 affine: frame -> square crop (face long side = share * size) turned angle_deg clockwise."""
    x1, y1, x2, y2 = (float(v) for v in box)
    side = max(x2 - x1, y2 - y1, 1e-6) / share
    s = size / side
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    scale = np.array([[s, 0.0, size / 2 - s * cx], [0.0, s, size / 2 - s * cy], [0.0, 0.0, 1.0]])
    return rotation_matrix(angle_deg, size) @ scale


def apply(matrix, points):
    p = np.asarray(points, np.float64).reshape(-1, 2)
    return p @ matrix[:2, :2].T + matrix[:2, 2]


def wrap_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0


def eye_line_deg(detection):
    """Clockwise tilt of the eye line (degrees, y down), oriented so the nose is below it."""
    a, b, n = (np.asarray(detection.point(k), np.float64) for k in ('left_eye', 'right_eye', 'nose'))
    axis = b - a
    if float((n - (a + b) / 2) @ np.array([-axis[1], axis[0]])) < 0:
        axis = -axis
    return math.degrees(math.atan2(axis[1], axis[0]))


EVIDENCE_SIZE = 96   # canvas on which the eye distance is 0.42 * EVIDENCE_SIZE
DARK_RING = 50.0     # grey level below which the surroundings are a dark mask


def eye_contrast(frame, det):
    """Image evidence of an open eye at each predicted eye point (no model involved).

    An open eye is darker than the fur around it (iris, pupil); plain fur, a cheek, a
    hand or a closed lid is not.  On a canvas where the eye distance is fixed:
      darker = (median of a ring at 1.6-2.6 r - 10th percentile of the disc r = 0.09 n) / ring median.
    Inside a dark mask (ring median < DARK_RING) "darker" cannot work, so there the
    brightness range of the eye itself also counts (iris against a catch-light):
      range  = (95th - 5th percentile of the disc r = 0.06 n) / 255.
    Validation (422 dogface6-val faces): ~96 % of correctly placed eyes reach 0.20; a
    black-masked puppy's eye 0.54 (darker-only: 0.14); eyes placed on a hand 0.08.
    """
    n = EVIDENCE_SIZE
    le, re = (np.asarray(det.point(k), np.float64) for k in ('left_eye', 'right_eye'))
    iod = max(float(np.linalg.norm(re - le)), 1.0)
    s = 0.42 * n / iod
    mid = (le + re) / 2
    matrix = np.float32([[s, 0, n / 2 - s * mid[0]], [0, s, n / 2 - s * mid[1]]])
    canvas = cv2.warpAffine(frame, matrix, (n, n), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY).astype(np.float32)
    yy, xx = np.mgrid[:n, :n]
    out = []
    for point in (le, re):
        q = (point - mid) * s + n / 2
        r = np.hypot(xx - q[0], yy - q[1])
        radius = 0.09 * n
        inner, ring = gray[r <= radius], gray[(r > 1.6 * radius) & (r <= 2.6 * radius)]
        core = gray[r <= (2 / 3) * radius]
        if inner.size < 5 or ring.size < 5 or core.size < 5:
            out.append(0.0)
            continue
        level = float(np.median(ring))
        darker = (level - float(np.percentile(inner, 10))) / max(level, 20.0)
        spread = (float(np.percentile(core, 95)) - float(np.percentile(core, 5))) / 255.0
        out.append(max(darker, spread) if level < DARK_RING else darker)
    return tuple(out)


def flip_index(schema):
    index = list(range(len(schema)))
    for left, right in schema.pairs:
        a, b = schema.index(left), schema.index(right)
        if a is not None and b is not None:
            index[a], index[b] = b, a
    return index


class FaceRefiner:
    """Per-face region of interest: canonical scale and a continuous upright angle.

    Each face carries ``angle``, the clockwise turn that makes it upright.  Video: the
    crop uses the angle carried from the previous frame and the measured residual tilt
    updates it for the next frame (one model run per face per frame); the four quarter
    turns are re-checked every ``search_period_s``.  Photos (and the server re-check):
    quarter-turn search, then one immediate re-crop at the measured residual.
    """

    def __init__(self, detector, config: RefineConfig | None = None):
        self.detector = detector
        self.config = config or RefineConfig()
        self.size = int(getattr(detector, 'size', 320))
        self.rois = []   # [{'box', 'region', 'angle', 'searched_at', 'searched_frame', 'score', 'misses'}]
        self.frame = 0

    def reset(self):
        self.rois = []

    # ---- one crop -----------------------------------------------------
    def crop(self, frame, box, angle):
        matrix = crop_matrix(box, angle, self.config.share, self.size)
        image = cv2.warpAffine(frame, matrix[:2], (self.size, self.size), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
        return image, matrix

    def _pick(self, detections):
        """The face at the crop centre (the one being refined), not a neighbour."""
        if not detections:
            return None
        centre = self.size / 2

        def key(d):
            cx, cy = (d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2
            return d.score / (1 + math.hypot(cx - centre, cy - centre) / (0.25 * self.size))
        return max(detections, key=key)

    def examine(self, frame, box, angle):
        """Detect on the crop turned ``angle``; returns a Detection in frame coordinates or None."""
        image, matrix = self.crop(frame, box, angle)
        det = self._pick([d for d in self.detector.detect(image) if d.validate() is None])
        if det is None:
            return None
        inverse = np.linalg.inv(matrix)
        x1, y1, x2, y2 = det.box
        corners = apply(inverse, [[x1, y1], [x2, y1], [x1, y2], [x2, y2]])
        box_back = [corners[:, 0].min(), corners[:, 1].min(), corners[:, 0].max(), corners[:, 1].max()]
        # The face's own extent (square, centred on it): the next crop is built from this, not
        # from box_back, which bounds a rotated rectangle and would grow frame after frame.
        centre = apply(inverse, [[(x1 + x2) / 2, (y1 + y2) / 2]])[0]
        half = max(x2 - x1, y2 - y1) / 2 / math.hypot(matrix[0, 0], matrix[0, 1])
        region = [centre[0] - half, centre[1] - half, centre[0] + half, centre[1] + half]
        tilt = eye_line_deg(det)
        upright = wrap_deg(angle - tilt)
        return Detection(box_back, det.score, apply(inverse, det.keypoints), det.schema,
                         kp_scores=det.kp_scores, source=f'{det.source or "detector"}+refined',
                         extra={'upright_angle': upright, 'upright_k': int(round(upright / 90.0)) % 4,
                                'crop_angle': wrap_deg(angle), 'crop_tilt': tilt, 'crop_score': float(det.score),
                                'region': region})

    def _mirror_points(self, image):
        other = self._pick([d for d in self.detector.detect(np.ascontiguousarray(image[:, ::-1]))
                            if d.validate() is None])
        if other is None:
            return None
        points = np.asarray(other.keypoints, np.float64).copy()
        points[:, 0] = self.size - 1 - points[:, 0]
        return points[flip_index(other.schema)]

    def verify(self, frame, det):
        """Open-loop second look before a shot is taken.

        The face is re-detected on the crop turned by the nearest *quarter* turn, i.e.
        an input chosen from four fixed orientations rather than adapted to this
        prediction's own eye line (a continuously re-rotated crop can lock a
        hallucination in: a closed-eye profile then looks like a frontal face).
        Returns crop score, eye confidences, mirror disagreement and agreement with
        ``det`` (both in units of the predicted eye distance; inf if lost).
        """
        quarter = 90.0 * round(det.extra.get('upright_angle', 0.0) / 90.0)
        image, matrix = self.crop(frame, det.extra.get('region', det.box), quarter)
        check = self._pick([d for d in self.detector.detect(image) if d.validate() is None])
        eyes = [det.schema.index('left_eye'), det.schema.index('right_eye')]
        own = apply(matrix, det.keypoints)
        span = max(float(np.linalg.norm(own[eyes[0]] - own[eyes[1]])), 1e-6)
        if check is None:
            return {'score': 0.0, 'eyes': (0.0, 0.0), 'mirror': math.inf, 'agreement': math.inf}
        points = np.asarray(check.keypoints, np.float64)
        mirrored = self._mirror_points(image)
        conf = check.kp_scores if check.kp_scores is not None else np.ones(len(points))
        return {'score': float(check.score), 'eyes': (float(conf[eyes[0]]), float(conf[eyes[1]])),
                'mirror': math.inf if mirrored is None else float(np.mean(np.linalg.norm(points - mirrored, axis=1)) / span),
                'agreement': float(np.mean(np.linalg.norm(points - own, axis=1)) / span)}

    # ---- orientation --------------------------------------------------
    def _search(self, frame, box, base, base_detection):
        """Compare the base crop already examined on this frame with three quarter turns."""
        results = {0: base_detection} if base_detection is not None else {}
        for j in range(1, 4):
            det = self.examine(frame, box, base + 90.0 * j)
            if det is not None:
                results[j] = det
        if not results:
            return None
        best = max(results.values(), key=lambda d: d.score)
        chosen = results[0] if 0 in results and results[0].score + self.config.switch_margin >= best.score else best
        others = [d.score for d in results.values() if d is not chosen]
        chosen.extra['search_margin'] = chosen.score - max(others, default=0.0)
        chosen.extra['search_kept'] = chosen is results.get(0)
        return chosen

    def _settle(self, frame, box, det):
        """One re-crop at the measured residual tilt (validation: 30-45 deg residual 18 -> 49 % good)."""
        if det is None or abs(det.extra['crop_tilt']) <= self.config.fine_min_deg:
            return det
        again = self.examine(frame, box, det.extra['upright_angle'])
        return again if again is not None and again.score >= det.score - 0.05 else det

    def refine(self, frame, detections, timestamp, still=False, full_ran=True):
        """Refined detections for this frame: full-frame ones plus faces carried from the last frame.

        ``full_ran``: the full-frame detector ran on this frame.  Then a carried face is one it
        missed and is followed for at most ``carry_frames``; when it was skipped on purpose
        (refine.full_detect_period_s) a confident crop alone keeps the face.
        """
        c = self.config
        self.frame += 1
        previous = [] if still else self.rois
        boxes = [(d.box, d) for d in detections]
        if previous:
            current = np.stack([d.box for d in detections]) if detections else np.zeros((0, 4), np.float32)
            for roi in previous:
                if roi['misses'] >= c.carry_frames:
                    continue
                if len(current) and box_iou_matrix(roi['box'][None], current)[0].max() >= c.match_iou:
                    continue
                boxes.append((roi['region'], None))   # carried: the full-frame detector missed it
        out, rois = [], []
        for box, source in boxes:
            match = None
            if previous:
                overlaps = box_iou_matrix(np.asarray(box, np.float32)[None], np.stack([r['box'] for r in previous]))[0]
                if overlaps.max() >= c.match_iou:
                    match = previous[int(overlaps.argmax())]
            base = 0.0 if match is None else match['angle']
            age = None if match is None else timestamp - match['searched_at']
            # Re-check the quarter turn when a face is new or doubtful; a confidently followed
            # face only as a safety net (each check costs four model runs).
            frames = 0 if match is None else self.frame - match.get('searched_frame', 0)
            det = self.examine(frame, box, base)
            backoff = 1 if match is None else match.get('backoff', 1)
            due = (still or match is None or match['score'] < c.research_below
                   or det is None or det.score < c.research_below
                   or (frames >= c.search_min_frames * backoff
                       and ((age >= c.search_period_s * backoff and match['score'] < c.confident_score)
                            or age >= c.max_search_period_s)))
            searched_at = timestamp if due else match['searched_at']
            if due:
                det = self._search(frame, box, base, det)
                det = self._settle(frame, box, det)
            if det is None:
                if source is not None:       # keep the full-frame face, marked as not refined
                    source.extra = {**source.extra, 'upright_k': 0, 'refined': False}
                    out.append(source)
                if match is not None:
                    rois.append({**match, 'misses': match['misses'] + 1})
                continue
            if source is not None:
                det.body_box, det.body_score = source.body_box, source.body_score
                det.extra['full_score'] = float(source.score)
            det.extra['refined'] = True
            det.extra['carried'] = source is None
            out.append(det)
            confirmed = det.score >= c.research_below
            previous_misses = match['misses'] if match else 0
            if source is not None:
                misses = 0
            elif full_ran or not confirmed:
                misses = previous_misses + 1          # the detector missed it, or the crop is weak
            else:
                misses = previous_misses              # skipped detector: the confident crop vouches
            rois.append({'box': np.asarray(det.box, np.float32), 'region': det.extra['region'],
                         'angle': det.extra['upright_angle'],
                         'searched_at': searched_at,
                         'searched_frame': self.frame if due else match.get('searched_frame', 0),
                         'backoff': _backoff(c, backoff, det) if due else backoff,
                         'score': det.score, 'misses': misses})
        if not still:
            self.rois = rois
        return _dedupe(out)


def _backoff(config, backoff, det):
    """Next search-interval multiplier after a search: grows while the turn is clearly confirmed."""
    if det.extra.get('search_kept') and det.extra.get('search_margin', 0.0) >= config.backoff_margin:
        return min(backoff * 2, config.max_backoff)
    return 1


def _dedupe(detections, iou=0.5):
    """Two candidate boxes can converge on one face after refinement: keep the best."""
    if len(detections) < 2:
        return detections
    order = sorted(detections, key=lambda d: -d.score)
    kept = []
    for d in order:
        if kept and box_iou_matrix(d.box[None], np.stack([k.box for k in kept]))[0].max() >= iou:
            continue
        kept.append(d)
    return kept


def upright_frame(frame, k):
    """Frame turned k quarter turns clockwise (lossless) and the 3x3 map of its pixel coordinates."""
    h, w = frame.shape[:2]
    k %= 4
    if k == 0:
        return frame, np.eye(3)
    image = cv2.rotate(frame, {1: cv2.ROTATE_90_CLOCKWISE, 2: cv2.ROTATE_180, 3: cv2.ROTATE_90_COUNTERCLOCKWISE}[k])
    if k == 1:
        m = np.array([[0.0, -1.0, h - 1.0], [1.0, 0.0, 0.0], [0, 0, 1]])
    elif k == 2:
        m = np.array([[-1.0, 0.0, w - 1.0], [0.0, -1.0, h - 1.0], [0, 0, 1]])
    else:
        m = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, w - 1.0], [0, 0, 1]])
    return image, m


def transform_detection(det, matrix):
    """Detection mapped by a 3x3 pixel map (axis-aligned box of the mapped box)."""
    x1, y1, x2, y2 = det.box
    corners = apply(matrix, [[x1, y1], [x2, y1], [x1, y2], [x2, y2]])
    extra = dict(det.extra)
    out = Detection([corners[:, 0].min(), corners[:, 1].min(), corners[:, 0].max(), corners[:, 1].max()],
                    det.score, apply(matrix, det.keypoints), det.schema, kp_scores=det.kp_scores,
                    body_box=None if det.body_box is None else apply(matrix, np.reshape(det.body_box, (2, 2))).reshape(-1),
                    body_score=det.body_score, source=det.source, extra=extra)
    if det.body_box is not None:
        b = out.body_box.reshape(2, 2)
        out.body_box = np.array([b[:, 0].min(), b[:, 1].min(), b[:, 0].max(), b[:, 1].max()], np.float32)
    return out
