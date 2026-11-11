/** Head pose from keypoints and the debounced view classifier (port of pose.py). */
import type { CaptureConfig, ViewSpec } from './config';
import { height, width, type Detection, type Point } from './schema';
import { OneEuroFilter } from './tracking';

export interface FaceGeometry {
  points: Record<string, Point>;
  scores: Record<string, number>;
  iod: number;
  roll_deg: number;
  yaw: number;
  pitch: number;
  lower_yaw: number | null;
  approx_deg: number;
  swapped: boolean;
  /** quarter turns (clockwise) that make the face upright; roll_deg is measured after them */
  upright_k: number;
  valid: boolean;
  reason: string;
}

export function faceGeometry(det: Detection, config: CaptureConfig['pose']): FaceGeometry {
  const points: Record<string, Point> = {};
  const scores: Record<string, number> = {};
  det.schema.names.forEach((name, i) => {
    points[name] = [det.keypoints[i][0], det.keypoints[i][1]];
    if (det.kpScores) scores[name] = det.kpScores[i];
  });
  const uprightK = ((Number(det.extra?.upright_k ?? 0) % 4) + 4) % 4;
  const base: FaceGeometry = { points, scores, iod: 0, roll_deg: 0, yaw: 0, pitch: 0, lower_yaw: null, approx_deg: 0,
    swapped: false, upright_k: uprightK, valid: false, reason: 'no_face' };
  let a = points.left_eye, b = points.right_eye;
  const nose = points.nose;
  let axis = [b[0] - a[0], b[1] - a[1]];
  const span = Math.hypot(axis[0], axis[1]);
  if (span < 1e-6) return { ...base, reason: 'eyes_coincide' };
  let normal = [-axis[1] / span, axis[0] / span];
  const swapped = (nose[0] - (a[0] + b[0]) / 2) * normal[0] + (nose[1] - (a[1] + b[1]) / 2) * normal[1] < 0;
  if (swapped) {
    for (const [l, r] of det.schema.pairs) {
      if (l in points && r in points) {
        [points[l], points[r]] = [points[r], points[l]];
        if (l in scores && r in scores) [scores[l], scores[r]] = [scores[r], scores[l]];
      }
    }
    a = points.left_eye; b = points.right_eye;
    axis = [b[0] - a[0], b[1] - a[1]];
    normal = [-axis[1] / span, axis[0] / span];
  }
  const h = [axis[0] / span, axis[1] / span];
  const mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
  const off = [nose[0] - mid[0], nose[1] - mid[1]];
  const scale = det.schema.yawScale;
  const yaw = (off[0] * h[0] + off[1] * h[1]) / span / scale;
  const pitch = (off[0] * normal[0] + off[1] * normal[1]) / span;
  // roll in the face's upright orientation (a dog lying down is upright after its quarter turn)
  let roll = Math.atan2(h[1], h[0]) * 180 / Math.PI + 90 * uprightK;
  roll = ((((roll + 180) % 360) + 360) % 360) - 180;
  let lower: number | null = null;
  if ('left_mouth' in points && 'right_mouth' in points) {
    const m = [(points.left_mouth[0] + points.right_mouth[0]) / 2 - mid[0], (points.left_mouth[1] + points.right_mouth[1]) / 2 - mid[1]];
    lower = (m[0] * h[0] + m[1] * h[1]) / span / scale;
  } else if ('chin' in points && (Object.keys(scores).length === 0 || (scores.chin ?? 1) >= 0.3)) {
    lower = ((points.chin[0] - mid[0]) * h[0] + (points.chin[1] - mid[1]) * h[1]) / span / scale;
  }
  const g: FaceGeometry = { points, scores, iod: span, roll_deg: roll, yaw, pitch, lower_yaw: lower,
    approx_deg: Math.atan2(yaw, config.snout_factor) * 180 / Math.PI, swapped, upright_k: uprightK, valid: false,
    reason: 'ok' };
  const boxW = Math.max(width(det), height(det));
  if (span < config.min_iod_box_ratio * boxW) g.reason = 'eyes_unresolved';
  else if (span > config.max_iod_box_ratio * boxW) g.reason = 'implausible_eye_span';
  else if (!(config.pitch_range[0] <= pitch && pitch <= config.pitch_range[1])) g.reason = 'implausible_nose_position';
  else if (Math.abs(yaw) > config.max_abs_yaw) g.reason = 'extreme_pose';
  else if (lower !== null && Math.abs(yaw) > 0.18 && Math.abs(lower) > 0.18 && Math.sign(lower) !== Math.sign(yaw)) g.reason = 'inconsistent_landmarks';
  else g.valid = true;
  if (!g.valid) return g;
  g.reason = 'ok';
  return g;
}

export interface ViewState {
  raw: string | null;
  view: string | null;
  stable: boolean;
  smoothed_yaw: number | null;
  dwell_s: number;
  observations: number;
  changed: boolean;
}

export class ViewClassifier {
  private filter: OneEuroFilter;
  private views: Map<string, ViewSpec>;
  view: string | null = null;
  private since: number | null = null;
  private count = 0;
  private lastTime: number | null = null;

  constructor(private config: CaptureConfig['pose']) {
    this.views = new Map(config.views.map((v) => [v.name, v]));
    this.filter = new OneEuroFilter(config.smooth_min_cutoff, config.smooth_beta);
  }

  reset() { this.filter.reset(); this.view = null; this.since = null; this.count = 0; this.lastTime = null; }

  band(yaw: number, roll: number, margin = 0): string | null {
    for (const v of this.views.values()) {
      if (v.lower - margin <= yaw && yaw <= v.upper + margin && Math.abs(roll) <= v.max_roll_deg) return v.name;
    }
    return null;
  }

  update(t: number, g: FaceGeometry | null): ViewState {
    const c = this.config;
    const previous = this.view;
    if (this.lastTime !== null && t - this.lastTime > c.max_gap_s) this.reset();
    if (!g || !g.valid) {
      this.view = null; this.since = null; this.count = 0; this.filter.reset(); this.lastTime = t;
      return { raw: null, view: null, stable: false, smoothed_yaw: null, dwell_s: 0, observations: 0, changed: previous !== null };
    }
    const smoothed = this.filter.filter(t, g.yaw);
    const roll = g.roll_deg;
    const raw = this.band(g.yaw, roll);
    const entry = raw !== null && this.band(smoothed, roll) === raw ? raw : null;
    if (this.view !== null) {
      const spec = this.views.get(this.view)!;
      const m = c.exit_margin;
      const keep = spec.lower - m <= g.yaw && g.yaw <= spec.upper + m && spec.lower - m <= smoothed
        && smoothed <= spec.upper + m && Math.abs(roll) <= spec.max_roll_deg;
      if (keep && (entry === null || entry === this.view)) this.count += 1;
      else if (entry !== null) { this.view = entry; this.since = t; this.count = 1; }
      else { this.view = null; this.since = null; this.count = 0; }
    } else if (entry !== null) { this.view = entry; this.since = t; this.count = 1; }
    this.lastTime = t;
    const dwell = this.since === null ? 0 : t - this.since;
    const stable = this.view !== null && this.count >= c.min_dwell_obs && dwell >= c.min_dwell_s;
    return { raw, view: this.view, stable, smoothed_yaw: smoothed, dwell_s: dwell, observations: this.count,
      changed: this.view !== previous };
  }
}
