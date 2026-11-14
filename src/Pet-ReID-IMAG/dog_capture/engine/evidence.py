"""Temporal landmark evidence owned by ONE track in ONE coordinate system.

Presence is not a successful check. One processed frame contributes at most
one observation; a second nearby dog cannot provide extra temporal votes.
The engine's processed-frame clock supplies the existing missing-frame/window
policy, while the track context owns and releases the observations.
"""
from collections import deque
import math


class TemporalEvidence:
    def __init__(self):
        self.observations = deque()  # (time, cx, cy, size, 'pass'|'fail'|'present')

    def trim(self, earliest):
        while self.observations and self.observations[0][0] < earliest:
            self.observations.popleft()

    def record(self, timestamp, place, status):
        if status not in ('pass', 'fail', 'present'):
            raise ValueError('Unknown landmark evidence status')
        if self.observations and timestamp < self.observations[-1][0]:
            raise ValueError('Landmark evidence timestamps must not decrease')
        if self.observations and timestamp == self.observations[-1][0]:
            self.observations.pop()  # replacing this frame never creates a second vote
        self.observations.append((timestamp, *place, status))

    def reject_reason(self, place, frame_count, config):
        """Called only AFTER this frame's own spatial/model checks pass.

        Keep the prior spatial gate and ratio semantics: 'present' means not
        checked, so it is neither a pass nor a fail. Absent frames are missing.
        The current frame has not been recorded yet and contributes one pass.
        """
        cx, cy, size = place
        passes, fails, seen = 1, 0, 0
        for _, x, y, previous_size, status in self.observations:
            if math.hypot(x - cx, y - cy) <= 0.5 * max(previous_size, size):
                seen += 1
                passes += status == 'pass'
                fails += status == 'fail'
        missing = max(0, frame_count - 1 - seen)
        if (passes < config.gate_min_passes
                or passes < config.gate_min_ratio * (passes + fails + missing)):
            return 'landmarks_unstable'
        return None
