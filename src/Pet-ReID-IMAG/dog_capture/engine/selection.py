"""Per-track best-shot selection, after face-capture camera policies.

Face capture cameras keep, per tracked person, the best face seen so far and
emit it according to a policy:

``fast``    the first frame whose score reaches ``good_enough`` (low latency);
``best``    the best frame of the whole track (highest quality);
``hybrid``  emit at the first good-enough frame, keep upgrading for
            ``improve_window_s``, then lock (the common default).

Here this is done per target view.  A shot only enters a view slot while the
debounced view is that view *and* the frame's own pose is in the band.  Shots
seen before the dwell is confirmed wait in a pending buffer and are promoted
when it is, so the sharp first frame of a short dwell is not lost; if the
visit ends unconfirmed they are dropped.  Tracks are never mixed: one bundle =
one track = one dog.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

LEVEL_ORDER = {'reject': 0, 'low': 1, 'medium': 2, 'high': 3}


@dataclass
class Shot:
    view: str
    track_id: int
    frame_index: int
    time_s: float
    score: float
    quality: object          # QualityReport
    geometry: object         # FaceGeometry
    detection: object        # Detection
    aligned: np.ndarray      # aligned BGR crop (output size)
    frame: np.ndarray | None  # full upright frame, if kept
    rotation_deg: int = 0
    basis: str = 'stable_view'
    confirmed_at: float | None = None

    def summary(self):
        return {'view': self.view, 'track_id': self.track_id, 'frame_index': self.frame_index,
                'time_s': round(self.time_s, 4), 'score': round(self.score, 4),
                'level': self.quality.level, 'basis': self.basis,
                'confirmed_at_s': None if self.confirmed_at is None else round(self.confirmed_at, 4),
                'rotation_deg': self.rotation_deg,
                'detection': self.detection.to_dict(), 'pose': self.geometry.to_dict(),
                'quality': self.quality.to_dict()}


class TrackShots:
    def __init__(self, track_id, view_names, config, shots=None):
        self.track_id = track_id
        self.view_names = tuple(view_names)
        self.config = config
        # images kept per view (ViewSpec.shots), default selection.top_k
        self.k = {v: int((shots or {}).get(v) or config.top_k) for v in self.view_names}
        self.slots = {v: [] for v in self.view_names}
        self.pending = {v: [] for v in self.view_names}
        self.first_good = {v: None for v in self.view_names}
        self.locked = set()
        self.visit = None
        self.first_time = self.last_time = None

    # ---- queries ------------------------------------------------------
    @staticmethod
    def level_ok(level, minimum):
        return LEVEL_ORDER[level] >= LEVEL_ORDER[minimum]

    def eligible_level(self, level):
        return self.level_ok(level, self.config.min_level)

    def wants(self, view, score):
        """Cheap pre-check so a full-frame copy is only made when it can matter."""
        if view not in self.slots or view in self.locked:
            return False
        k = self.k[view]
        for bucket in (self.slots[view], self.pending[view]):
            if len(bucket) >= k and score <= min(s.score for s in bucket):
                return False  # could never enter the slot (or the pending buffer)
        return True

    def captured(self):
        return [v for v in self.view_names if self.slots[v]]

    def full(self, view):
        return len(self.slots[view]) >= self.k[view]

    def best(self, view):
        return self.slots[view][0] if self.slots.get(view) else None

    def rank(self):
        views = self.captured()
        return (len(views), int(bool(views) and self.view_names[0] in views),
                sum(self.slots[v][0].score for v in views))

    # ---- updates ------------------------------------------------------
    def observe(self, view_state, timestamp):
        """Track the debounced visit; unconfirmed visits drop their pending shots."""
        if self.first_time is None:
            self.first_time = timestamp
        self.last_time = timestamp
        if view_state.view != self.visit:
            if self.visit is not None:
                self.pending[self.visit] = []
            self.visit = view_state.view
        events = []
        if self.visit is not None and view_state.stable and self.pending[self.visit]:
            for shot in self.pending[self.visit]:
                shot.confirmed_at = timestamp
                events += self._insert(shot, timestamp)
            self.pending[self.visit] = []
        self._update_locks(timestamp)
        return events

    def offer(self, shot, view_state, timestamp):
        """Returns (events, kept): kept = the shot entered the slot or the pending buffer."""
        if shot.view in self.locked or view_state.view != shot.view:
            return [], False
        if not view_state.stable:
            bucket = self.pending[shot.view]
            bucket.append(shot)
            bucket.sort(key=lambda s: -s.score)
            del bucket[self.k[shot.view]:]
            return [], shot in bucket
        shot.confirmed_at = timestamp
        events = self._insert(shot, timestamp)
        self._update_locks(timestamp)
        return events, shot in self.slots[shot.view]

    def _insert(self, shot, timestamp):
        c = self.config
        if shot.view in self.locked:
            return []
        slot = self.slots[shot.view]
        was_empty = not slot
        previous_best = slot[0].score if slot else None
        k = self.k[shot.view]
        if k > 1:
            near = [s for s in slot if abs(s.time_s - shot.time_s) < c.min_gap_s]
            if near:
                if shot.score <= max(s.score for s in near):
                    return []
                for s in near:
                    slot.remove(s)
        slot.append(shot)
        slot.sort(key=lambda s: -s.score)
        del slot[k:]
        if shot not in slot:
            return []
        events = []
        good = shot.score >= c.good_enough
        if good and self.first_good[shot.view] is None:
            self.first_good[shot.view] = timestamp
        if was_empty:
            events.append({'event': 'view_captured', 'view': shot.view, 'track_id': self.track_id,
                           'frame_index': shot.frame_index, 'time_s': shot.time_s, 'score': shot.score})
        elif previous_best is not None and shot.score > previous_best + 0.01:
            events.append({'event': 'view_improved', 'view': shot.view, 'track_id': self.track_id,
                           'frame_index': shot.frame_index, 'time_s': shot.time_s, 'score': shot.score})
        if c.mode == 'fast' and len(slot) >= k and min(s.score for s in slot) >= c.good_enough:
            self.locked.add(shot.view)
            events.append({'event': 'view_locked', 'view': shot.view, 'track_id': self.track_id,
                           'time_s': timestamp})
        return events

    def _update_locks(self, timestamp):
        if self.config.mode != 'hybrid':
            return
        for view, first in self.first_good.items():
            if (first is not None and view not in self.locked and self.full(view)
                    and timestamp - first >= self.config.improve_window_s):
                self.locked.add(view)

    def finalize(self, timestamp=None):
        """Pending shots of an unconfirmed final visit are discarded."""
        self.pending = {v: [] for v in self.view_names}

    def summary(self):
        return {'track_id': self.track_id, 'views': {v: [s.summary() for s in self.slots[v]]
                                                     for v in self.view_names if self.slots[v]},
                'missing_views': [v for v in self.view_names if not self.slots[v]],
                'locked_views': sorted(self.locked),
                'first_time_s': self.first_time, 'last_time_s': self.last_time}
