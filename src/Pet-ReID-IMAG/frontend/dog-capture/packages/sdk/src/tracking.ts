/** ByteTrack + constant-velocity Kalman filter + One-Euro filter (port of tracking.py). */
import type { CaptureConfig } from './config';
import { iou, type Box, type Detection } from './schema';

export class OneEuroFilter {
  private t: number | null = null;
  private x = 0;
  private dx = 0;
  constructor(private minCutoff = 1.0, private beta = 0.0, private dCutoff = 1.0) {}

  reset() { this.t = null; }

  private static alpha(cutoff: number, dt: number) {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }

  filter(t: number, x: number): number {
    if (this.t === null || t <= this.t) { this.t = t; this.x = x; this.dx = 0; return x; }
    const dt = t - this.t;
    const dx = (x - this.x) / dt;
    this.dx += OneEuroFilter.alpha(this.dCutoff, dt) * (dx - this.dx);
    const cutoff = this.minCutoff + this.beta * Math.abs(this.dx);
    this.x += OneEuroFilter.alpha(cutoff, dt) * (x - this.x);
    this.t = t;
    return this.x;
  }
}

type Vec = number[];
type Mat = number[][];

const zeros = (r: number, c: number): Mat => Array.from({ length: r }, () => new Array(c).fill(0));
const eye = (n: number): Mat => zeros(n, n).map((row, i) => { row[i] = 1; return row; });
const matmul = (a: Mat, b: Mat): Mat => a.map((row) => b[0].map((_, j) => row.reduce((s, v, k) => s + v * b[k][j], 0)));
const transpose = (a: Mat): Mat => a[0].map((_, j) => a.map((row) => row[j]));
const matvec = (a: Mat, v: Vec): Vec => a.map((row) => row.reduce((s, x, k) => s + x * v[k], 0));

/** Solve A X = B for X (A n x n, B n x m), Gaussian elimination with partial pivoting. */
function solve(A: Mat, B: Mat): Mat {
  const n = A.length, m = B[0].length;
  const a = A.map((row, i) => [...row, ...B[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(a[r][col]) > Math.abs(a[pivot][col])) pivot = r;
    [a[col], a[pivot]] = [a[pivot], a[col]];
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = a[r][col] / a[col][col];
      for (let c = col; c < n + m; c++) a[r][c] -= factor * a[col][c];
    }
  }
  return a.map((row, i) => row.slice(n).map((v) => v / a[i][i]));
}

export const xyxyToXyah = (b: Box): Vec => {
  const w = Math.max(b[2] - b[0], 1e-3), h = Math.max(b[3] - b[1], 1e-3);
  return [b[0] + w / 2, b[1] + h / 2, w / h, h];
};
export const xyahToXyxy = (s: Vec): Box => {
  const w = s[2] * s[3];
  return [s[0] - w / 2, s[1] - s[3] / 2, s[0] + w / 2, s[1] + s[3] / 2].map(Math.fround) as Box;
};

export class KalmanBoxFilter {
  static stdPosition = 1 / 20;
  static stdVelocity = 1 / 160;

  initiate(box: Box): [Vec, Mat] {
    const mean = [...xyxyToXyah(box), 0, 0, 0, 0];
    const h = mean[3], p = KalmanBoxFilter.stdPosition, v = KalmanBoxFilter.stdVelocity;
    const std = [2 * p * h, 2 * p * h, 1e-2, 2 * p * h, 10 * v * h, 10 * v * h, 1e-5, 10 * v * h];
    return [mean, std.map((s, i) => { const row = new Array(8).fill(0); row[i] = s * s; return row; })];
  }

  predict(mean: Vec, cov: Mat, dt: number): [Vec, Mat] {
    dt = Math.min(Math.max(dt, 0), 30);
    const motion = eye(8);
    for (let i = 0; i < 4; i++) motion[i][i + 4] = dt;
    const h = Math.max(mean[3], 1e-3), p = KalmanBoxFilter.stdPosition, v = KalmanBoxFilter.stdVelocity;
    const std = [p * h, p * h, 1e-2, p * h, v * h, v * h, 1e-5, v * h];
    const scale = Math.max(dt, 1e-3);
    const next = matmul(matmul(motion, cov), transpose(motion));
    for (let i = 0; i < 8; i++) next[i][i] += std[i] * std[i] * scale;
    return [matvec(motion, mean), next];
  }

  update(mean: Vec, cov: Mat, box: Box): [Vec, Mat] {
    const z = xyxyToXyah(box);
    const h = Math.max(mean[3], 1e-3), p = KalmanBoxFilter.stdPosition;
    const r = [p * h, p * h, 1e-1, p * h].map((s) => s * s);
    const projected = cov.slice(0, 4).map((row, i) => row.slice(0, 4).map((v, j) => v + (i === j ? r[i] : 0)));
    const gain = transpose(solve(projected, cov.slice(0, 4)));            // 8 x 4
    const innovation = z.map((v, i) => v - mean[i]);
    const newMean = mean.map((v, i) => v + gain[i].reduce((s, g, k) => s + g * innovation[k], 0));
    const correction = matmul(matmul(gain, projected), transpose(gain));
    const newCov = cov.map((row, i) => row.map((v, j) => v - correction[i][j]));
    return [newMean, newCov];
  }
}

/** Greedy assignment: stable sort by cost (then row, then column), identical to tracking.py. */
export function assign(cost: number[][], threshold: number, cols = cost.length ? cost[0].length : 0) {
  const rows = cost.length;
  const order: [number, number, number][] = [];
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) order.push([cost[r][c], r, c]);
  order.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
  const usedR = new Set<number>(), usedC = new Set<number>(), pairs: [number, number][] = [];
  for (const [value, r, c] of order) {
    if (value > threshold) break;
    if (!usedR.has(r) && !usedC.has(c)) { pairs.push([r, c]); usedR.add(r); usedC.add(c); }
  }
  return {
    pairs,
    restRows: [...Array(rows).keys()].filter((r) => !usedR.has(r)),
    restCols: [...Array(cols).keys()].filter((c) => !usedC.has(c)),
  };
}

export type TrackState = 'new' | 'tracked' | 'lost' | 'removed';

export class Track {
  mean: Vec;
  cov: Mat;
  state: TrackState = 'new';
  hits = 1;
  startTime: number;
  lastTime: number;
  predictTime: number;
  updated = true;
  score: number;
  isDog: boolean | null = null;
  dogRetry = -Infinity;

  constructor(public id: number, public detection: Detection, timestamp: number, private kalman: KalmanBoxFilter) {
    [this.mean, this.cov] = kalman.initiate(detection.box);
    this.score = detection.score;
    this.startTime = this.lastTime = this.predictTime = timestamp;
  }

  get box(): Box { return xyahToXyxy(this.mean); }

  /** centre velocity in pixels per second */
  velocity(nominalFps: number): [number, number] { return [this.mean[4] * nominalFps, this.mean[5] * nominalFps]; }

  predict(timestamp: number, nominalFps: number) {
    const dt = (timestamp - this.predictTime) * nominalFps;
    if (dt > 0) {
      const mean = [...this.mean];
      if (this.state !== 'tracked') mean[7] = 0;
      [this.mean, this.cov] = this.kalman.predict(mean, this.cov, dt);
      this.predictTime = timestamp;
    }
  }

  update(detection: Detection, timestamp: number) {
    [this.mean, this.cov] = this.kalman.update(this.mean, this.cov, detection.box);
    this.detection = detection;
    this.score = detection.score;
    this.state = 'tracked';
    this.hits += 1;
    this.lastTime = timestamp;
    this.updated = true;
  }
}

export class ByteTracker {
  tracks: Track[] = [];
  removed: Track[] = [];
  readonly kalman = new KalmanBoxFilter();
  private nextId = 1;
  constructor(private config: CaptureConfig['tracker']) {}

  reset() { this.removed.push(...this.tracks); this.tracks = []; }

  update(detections: Detection[], timestamp: number): Track[] {
    const c = this.config;
    this.removed = [];
    for (const t of this.tracks) { t.updated = false; t.predict(timestamp, c.nominal_fps); }
    const high = detections.filter((d) => d.score >= c.high_score);
    const low = detections.filter((d) => d.score >= c.low_score && d.score < c.high_score);
    const confirmed = this.tracks.filter((t) => t.state === 'tracked' || t.state === 'lost');
    const unconfirmed = this.tracks.filter((t) => t.state === 'new');
    const cost = (tracks: Track[], dets: Detection[], fuse = true) =>
      tracks.map((t) => dets.map((d) => 1 - iou(t.box, d.box) * (fuse ? d.score : 1)));

    let result = assign(cost(confirmed, high), 1 - c.match_iou, high.length);
    for (const [ti, di] of result.pairs) confirmed[ti].update(high[di], timestamp);
    const remainingHigh = result.restCols.map((i) => high[i]);
    const candidates = result.restRows.map((i) => confirmed[i]).filter((t) => t.state === 'tracked');
    result = assign(cost(candidates, low, false), 1 - c.low_match_iou, low.length);
    for (const [ti, di] of result.pairs) candidates[ti].update(low[di], timestamp);
    for (const i of result.restRows) candidates[i].state = 'lost';
    result = assign(cost(unconfirmed, remainingHigh), 1 - c.unconfirmed_match_iou, remainingHigh.length);
    for (const [ti, di] of result.pairs) unconfirmed[ti].update(remainingHigh[di], timestamp);
    for (const i of result.restRows) unconfirmed[i].state = 'removed';
    for (const i of result.restCols) {
      const det = remainingHigh[i];
      if (det.score >= c.new_track_score) this.tracks.push(new Track(this.nextId++, det, timestamp, this.kalman));
    }
    for (const t of this.tracks) if (t.state === 'lost' && timestamp - t.lastTime > c.max_lost_s) t.state = 'removed';
    if (this.tracks.length > c.max_tracks) {
      const alive = this.tracks.filter((t) => t.state !== 'removed').sort((a, b) => b.lastTime - a.lastTime);
      for (const t of alive.slice(c.max_tracks)) t.state = 'removed';
    }
    this.removed = this.tracks.filter((t) => t.state === 'removed');
    this.tracks = this.tracks.filter((t) => t.state !== 'removed');
    return this.tracks.filter((t) => t.updated && t.state === 'tracked');
  }
}
