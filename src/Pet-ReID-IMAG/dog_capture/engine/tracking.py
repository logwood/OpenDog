"""Multi-target tracking: ByteTrack association + constant-velocity Kalman filter.

This follows ByteTrack (Zhang et al., ECCV 2022, MIT) and the SORT/DeepSORT
Kalman model: state (cx, cy, aspect, h) plus velocities, two-stage matching
where low-score detections may only continue existing tracks.  Two changes
for capture use: prediction uses *real timestamps* (frames can be skipped by
a latest-frame queue), and every track carries its last keypoint detection.

Also contains the One-Euro filter (Casiez et al., CHI 2012), the standard
landmark/pose smoother used by MediaPipe face and hand pipelines.
"""
from __future__ import annotations

import math

import numpy as np

from .schema import box_iou_matrix


class OneEuroFilter:
    """Speed-adaptive low-pass filter for scalars or arrays, driven by timestamps."""

    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = float(min_cutoff), float(beta), float(d_cutoff)
        self.reset()

    def reset(self):
        self.t = self.x = self.dx = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, t, x):
        x = np.asarray(x, np.float64)
        if self.t is None or t <= self.t:
            self.t, self.x, self.dx = t, x, np.zeros_like(x)
            return x.copy()
        dt = t - self.t
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = self.dx + a_d * (dx - self.dx)
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx)
        a = self._alpha(cutoff, dt)
        self.x = self.x + a * (x - self.x)
        self.t = t
        return self.x.copy()


def xyxy_to_xyah(box):
    x1, y1, x2, y2 = box
    w, h = max(x2 - x1, 1e-3), max(y2 - y1, 1e-3)
    return np.array([x1 + w / 2, y1 + h / 2, w / h, h], np.float64)


def xyah_to_xyxy(state):
    cx, cy, a, h = state[:4]
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], np.float32)


class KalmanBoxFilter:
    """DeepSORT/ByteTrack constant-velocity model; ``dt`` in nominal frame units."""
    std_position = 1.0 / 20
    std_velocity = 1.0 / 160

    def initiate(self, box):
        mean = np.r_[xyxy_to_xyah(box), np.zeros(4)]
        h = mean[3]
        std = [2 * self.std_position * h, 2 * self.std_position * h, 1e-2, 2 * self.std_position * h,
               10 * self.std_velocity * h, 10 * self.std_velocity * h, 1e-5, 10 * self.std_velocity * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean, cov, dt):
        dt = float(np.clip(dt, 0.0, 30.0))
        motion = np.eye(8)
        motion[:4, 4:] = np.eye(4) * dt
        h = max(mean[3], 1e-3)
        std = [self.std_position * h, self.std_position * h, 1e-2, self.std_position * h,
               self.std_velocity * h, self.std_velocity * h, 1e-5, self.std_velocity * h]
        q = np.diag(np.square(std)) * max(dt, 1e-3)
        return motion @ mean, motion @ cov @ motion.T + q

    def update(self, mean, cov, box):
        measurement = xyxy_to_xyah(box)
        h = max(mean[3], 1e-3)
        r = np.diag(np.square([self.std_position * h, self.std_position * h, 1e-1, self.std_position * h]))
        projected = cov[:4, :4] + r
        gain = np.linalg.solve(projected, cov[:4, :]).T
        mean = mean + gain @ (measurement - mean[:4])
        cov = cov - gain @ projected @ gain.T
        return mean, cov


def assign(cost, threshold):
    """Greedy min-cost assignment with a gate.

    Deliberately greedy (stable sort by cost, then row, then column) instead
    of Hungarian: with a handful of dogs the results are the same in practice,
    and every port (TypeScript, Kotlin, ...) produces identical pairs.
    """
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    pairs = []
    order = np.argsort(cost, axis=None, kind='stable')
    used_r, used_c = set(), set()
    for flat in order:
        r, c = divmod(int(flat), cost.shape[1])
        if cost[r, c] > threshold:
            break
        if r not in used_r and c not in used_c:
            pairs.append((r, c))
            used_r.add(r)
            used_c.add(c)
    matched_r = {r for r, _ in pairs}
    matched_c = {c for _, c in pairs}
    return (pairs, [r for r in range(cost.shape[0]) if r not in matched_r],
            [c for c in range(cost.shape[1]) if c not in matched_c])


class Track:
    NEW, TRACKED, LOST, REMOVED = 'new', 'tracked', 'lost', 'removed'

    def __init__(self, track_id, detection, timestamp, kalman):
        self.id = track_id
        self.kalman = kalman
        self.mean, self.cov = kalman.initiate(detection.box)
        self.detection = detection
        self.score = detection.score
        self.state = Track.NEW
        self.hits = 1
        self.start_time = self.last_time = self.predict_time = timestamp
        self.updated = True  # matched in the most recent update

    @property
    def box(self):
        return xyah_to_xyxy(self.mean)

    @property
    def confirmed(self):
        return self.state in (Track.TRACKED, Track.LOST) or self.hits >= 2

    def velocity(self, nominal_fps):
        """Centre velocity in pixels per second."""
        return self.mean[4:6] * nominal_fps

    def predict(self, timestamp, nominal_fps):
        dt = (timestamp - self.predict_time) * nominal_fps
        if dt > 0:
            mean = self.mean.copy()
            if self.state != Track.TRACKED:
                mean[7] = 0  # ByteTrack: frozen height velocity while lost
            self.mean, self.cov = self.kalman.predict(mean, self.cov, dt)
            self.predict_time = timestamp

    def update(self, detection, timestamp):
        self.mean, self.cov = self.kalman.update(self.mean, self.cov, detection.box)
        self.detection = detection
        self.score = detection.score
        self.state = Track.TRACKED
        self.hits += 1
        self.last_time = timestamp
        self.updated = True


class ByteTracker:
    def __init__(self, config):
        self.config = config
        self.kalman = KalmanBoxFilter()
        self.tracks: list[Track] = []
        self.next_id = 1
        self.removed: list[Track] = []

    def reset(self):
        self.removed.extend(self.tracks)
        self.tracks = []

    def update(self, detections, timestamp):
        """Associate detections; returns active (confirmed, updated) tracks.

        ``self.removed`` collects tracks that ended in this call so the caller
        can finalise their state; it is cleared at the start of each update.
        """
        c = self.config
        self.removed = []
        for track in self.tracks:
            track.updated = False
            track.predict(timestamp, c.nominal_fps)
        high = [d for d in detections if d.score >= c.high_score]
        low = [d for d in detections if c.low_score <= d.score < c.high_score]
        confirmed = [t for t in self.tracks if t.state in (Track.TRACKED, Track.LOST)]
        unconfirmed = [t for t in self.tracks if t.state == Track.NEW]

        def cost(tracks, dets, fuse=True):
            if not tracks or not dets:
                return np.zeros((len(tracks), len(dets)))
            iou = box_iou_matrix(np.stack([t.box for t in tracks]), np.stack([d.box for d in dets]))
            if fuse:  # ByteTrack fuse_score: confident detections win ties
                iou = iou * np.asarray([d.score for d in dets])[None, :]
            return 1.0 - iou

        # 1) high-score detections against all confirmed (tracked + lost) tracks
        pairs, rest_t, rest_d = assign(cost(confirmed, high), 1 - c.match_iou)
        for ti, di in pairs:
            confirmed[ti].update(high[di], timestamp)
        remaining_high = [high[i] for i in rest_d]
        # 2) low-score detections only continue tracks that were tracked last frame
        candidates = [confirmed[i] for i in rest_t if confirmed[i].state == Track.TRACKED]
        pairs, rest_low_t, _ = assign(cost(candidates, low, fuse=False), 1 - c.low_match_iou)
        for ti, di in pairs:
            candidates[ti].update(low[di], timestamp)
        for i in rest_low_t:
            candidates[i].state = Track.LOST
        # 3) unconfirmed tracks need a second high-score detection
        pairs, rest_u, rest_d = assign(cost(unconfirmed, remaining_high), 1 - c.unconfirmed_match_iou)
        for ti, di in pairs:
            unconfirmed[ti].update(remaining_high[di], timestamp)
        for i in rest_u:
            unconfirmed[i].state = Track.REMOVED
        # 4) births
        for i in rest_d:
            det = remaining_high[i]
            if det.score >= c.new_track_score:
                self.tracks.append(Track(self.next_id, det, timestamp, self.kalman))
                self.next_id += 1
        # 5) deaths
        for track in self.tracks:
            if track.state == Track.LOST and timestamp - track.last_time > c.max_lost_s:
                track.state = Track.REMOVED
        if len(self.tracks) > c.max_tracks:  # bounded memory: drop the stalest
            alive = sorted((t for t in self.tracks if t.state != Track.REMOVED),
                           key=lambda t: t.last_time, reverse=True)
            for track in alive[c.max_tracks:]:
                track.state = Track.REMOVED
        self.removed = [t for t in self.tracks if t.state == Track.REMOVED]
        self.tracks = [t for t in self.tracks if t.state != Track.REMOVED]
        return [t for t in self.tracks if t.updated and t.state == Track.TRACKED]
