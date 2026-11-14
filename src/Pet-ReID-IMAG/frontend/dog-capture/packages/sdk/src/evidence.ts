/** Temporal landmark evidence for ONE track in ONE coordinate system. */
import type { CaptureConfig } from './config';

export type GateStatus = 'pass' | 'fail' | 'present';
export type Place = [number, number, number];
export interface GateCheck { problem: string | null; status: GateStatus }

export class TemporalEvidence {
  readonly observations: [number, number, number, number, GateStatus][] = [];

  trim(earliest: number) {
    while (this.observations.length && this.observations[0][0] < earliest) this.observations.shift();
  }

  record(t: number, place: Place, status: GateStatus) {
    const last = this.observations.at(-1);
    if (last && t < last[0]) throw new Error('Landmark evidence timestamps must not decrease');
    if (last && t === last[0]) this.observations.pop(); // replacing this frame cannot add a vote
    this.observations.push([t, ...place, status]);
  }

  /** The current frame passed its own checks and has not been recorded yet. */
  rejectReason([cx, cy, size]: Place, frameCount: number, config: CaptureConfig['refine']): string | null {
    let passes = 1, fails = 0, seen = 0;
    for (const [, x, y, oldSize, status] of this.observations) {
      if (Math.hypot(x - cx, y - cy) <= 0.5 * Math.max(oldSize, size)) {
        seen++;
        if (status === 'pass') passes++;
        if (status === 'fail') fails++;
      }
    }
    // 'present' means not checked: neither a pass nor a fail, as in Python.
    const missing = Math.max(0, frameCount - 1 - seen);
    return passes < config.gate_min_passes || passes < config.gate_min_ratio * (passes + fails + missing)
      ? 'landmarks_unstable' : null;
  }
}
