"""Bounded recovery after failing to improve a verified, pending shot."""
import math


class CandidateRetry:
    def __init__(self, period_s):
        self.period_s = period_s
        self.view = None
        self.visit = 0
        self.failed_visit = None
        self.retry_at = -math.inf
        self.recovery_used = False
        self.recoverable = False

    def observe(self, view):
        # The pose classifier, not raw yaw noise, defines a new visit. A broken
        # dwell followed by reacquisition is also a new attempt at that view.
        if view != self.view:
            self.view = view
            if view is not None:
                self.visit += 1

    def begin(self, timestamp):
        """Call only for an eligible candidate that can improve the shot/buffer."""
        if self.view is None:
            return False
        if timestamp >= self.retry_at:
            self.recovery_used = False
            return True
        if self.recoverable and self.visit != self.failed_visit and not self.recovery_used:
            self.recovery_used = True
            return True
        return False

    def finish(self, timestamp, problem, passed, had_pending=False):
        if problem and problem != 'landmarks_unstable':
            self.failed_visit = self.visit
            self.retry_at = timestamp + self.period_s
            # Only a failed replacement of an already checked pending shot earns
            # recovery. Blindly retrying every new visit changes the checks that
            # feed temporal evidence and can make later, sharper shots lose out.
            self.recoverable = had_pending
            # An unsuccessful recovery does not mint another recovery token.
        elif passed:
            # A spatial/model check that passed must continue collecting the
            # original temporal evidence, even if it is not stable enough yet.
            self.failed_visit = None
            self.retry_at = -math.inf
            self.recovery_used = False
            self.recoverable = False
