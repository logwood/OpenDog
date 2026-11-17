/** Bounded recovery after failing to improve a verified pending shot (port of retry.py). */
export class CandidateRetry {
  view: string | null = null;
  visit = 0;
  failedVisit: number | null = null;
  retryAt = -Infinity;
  recoveryUsed = false;
  recoverable = false;

  constructor(readonly periodS: number) {}

  observe(view: string | null) {
    if (view !== this.view) {
      this.view = view;
      if (view !== null) this.visit++;
    }
  }

  /** Only reserve recovery for a qualified candidate that can improve the retained shot/buffer. */
  begin(t: number): boolean {
    if (this.view === null) return false;
    if (t >= this.retryAt) { this.recoveryUsed = false; return true; }
    if (this.recoverable && this.visit !== this.failedVisit && !this.recoveryUsed) {
      this.recoveryUsed = true;
      return true;
    }
    return false;
  }

  finish(t: number, problem: string | null, passed: boolean, hadPending = false) {
    if (problem && problem !== 'landmarks_unstable') {
      this.failedVisit = this.visit;
      this.retryAt = t + this.periodS;
      // A verified pending shot was already available before this replacement
      // failed. Ordinary acquisition failures keep the original backoff.
      this.recoverable = hadPending;
    } else if (passed) {
      this.failedVisit = null;
      this.retryAt = -Infinity;
      this.recoveryUsed = false;
      this.recoverable = false;
    }
  }
}
