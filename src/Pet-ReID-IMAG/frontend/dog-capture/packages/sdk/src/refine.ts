/**
 * Second-stage keypoints on a canonical, upright face crop — port of refine.py.
 *
 * The face model is trained on upright dogs whose faces cover about a third of
 * the picture; a dog lying down (face turned a quarter turn) or a phone held
 * close gets confident but wrong keypoints.  Each face box is re-examined on a
 * square crop where the face covers `share` of the side, turned so the face is
 * upright (quarter-turn search by detector score, then the measured residual
 * tilt), as face trackers do with a normalised region of interest.
 */
import type { CaptureConfig } from './config';
import type { FaceDetector } from './detectors';
import { mirror, rotate, toGray, warpAffine, type Frame } from './image';
import { detectionReason, iou, type Box, type Detection, type KeypointSchema, type Point } from './schema';

export type Mat3 = number[][];
type RefineConfig = CaptureConfig['refine'];

const round12 = (v: number) => Math.round(v * 1e12) / 1e12;

export function matmul(a: Mat3, b: Mat3): Mat3 {
  return a.map((row) => [0, 1, 2].map((j) => row[0] * b[0][j] + row[1] * b[1][j] + row[2] * b[2][j]));
}

export function invert(m: Mat3): Mat3 {
  const [[a, b, c], [d, e, f]] = m;
  const det = a * e - b * d;
  const ia = e / det, ib = -b / det, id = -d / det, ie = a / det;
  return [[ia, ib, -(ia * c + ib * f)], [id, ie, -(id * c + ie * f)], [0, 0, 1]];
}

export function apply(m: Mat3, points: Point[]): Point[] {
  return points.map(([x, y]) => [m[0][0] * x + m[0][1] * y + m[0][2], m[1][0] * x + m[1][1] * y + m[1][2]] as Point);
}

/** A size x size image turned `angle` degrees clockwise about its centre (90 = cv2.ROTATE_90_CLOCKWISE). */
export function rotationMatrix(angle: number, size: number): Mat3 {
  const a = angle * Math.PI / 180, c = round12(Math.cos(a)), s = round12(Math.sin(a)), o = (size - 1) / 2;
  return [[c, -s, o - c * o + s * o], [s, c, o - s * o - c * o], [0, 0, 1]];
}

/** Frame -> square crop (face long side = share * size) turned `angle` degrees clockwise. */
export function cropMatrix(box: Box, angle: number, share: number, size: number): Mat3 {
  const [x1, y1, x2, y2] = box;
  const side = Math.max(x2 - x1, y2 - y1, 1e-6) / share;
  const s = size / side, cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
  return matmul(rotationMatrix(angle, size), [[s, 0, size / 2 - s * cx], [0, s, size / 2 - s * cy], [0, 0, 1]]);
}

export const wrapDeg = (a: number) => ((((a + 180) % 360) + 360) % 360) - 180;

const pointOf = (d: Detection, name: string) => d.keypoints[d.schema.names.indexOf(name)];

/** Clockwise tilt of the eye line (degrees, y down), oriented so the nose is below it. */
export function eyeLineDeg(d: Detection): number {
  const a = pointOf(d, 'left_eye'), b = pointOf(d, 'right_eye'), n = pointOf(d, 'nose');
  let ax = b[0] - a[0], ay = b[1] - a[1];
  if ((n[0] - (a[0] + b[0]) / 2) * -ay + (n[1] - (a[1] + b[1]) / 2) * ax < 0) { ax = -ax; ay = -ay; }
  return Math.atan2(ay, ax) * 180 / Math.PI;
}

export function flipIndex(schema: KeypointSchema): number[] {
  const index = schema.names.map((_, i) => i);
  for (const [l, r] of schema.pairs) {
    const a = schema.names.indexOf(l), b = schema.names.indexOf(r);
    if (a >= 0 && b >= 0) { index[a] = b; index[b] = a; }
  }
  return index;
}

/** numpy.percentile (linear interpolation) of an unsorted array. */
export function percentile(values: number[], q: number): number {
  const a = [...values].sort((x, y) => x - y);
  const pos = (a.length - 1) * q / 100, lo = Math.floor(pos), hi = Math.ceil(pos);
  return a[lo] + (a[hi] - a[lo]) * (pos - lo);
}

export const EVIDENCE_SIZE = 96;
/** grey level below which the eye's surroundings are a dark mask (see refine.py eye_contrast) */
export const DARK_RING = 50;

/** Image evidence of an open eye at each predicted eye point (no model involved); see refine.py. */
export function eyeContrast(frame: Frame, d: Detection): [number, number] {
  const n = EVIDENCE_SIZE;
  const le = pointOf(d, 'left_eye'), re = pointOf(d, 'right_eye');
  const iod = Math.max(Math.hypot(re[0] - le[0], re[1] - le[1]), 1);
  const s = 0.42 * n / iod, mx = (le[0] + re[0]) / 2, my = (le[1] + re[1]) / 2;
  const canvas = warpAffine(frame, [[s, 0, n / 2 - s * mx], [0, s, n / 2 - s * my]], n, n, 114);
  const gray = toGray(canvas);
  const radius = 0.09 * n;
  return [le, re].map((p) => {
    const qx = (p[0] - mx) * s + n / 2, qy = (p[1] - my) * s + n / 2;
    const inner: number[] = [], ring: number[] = [], core: number[] = [];
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        const r = Math.hypot(x - qx, y - qy);
        if (r <= radius) inner.push(gray[y * n + x]);
        else if (r > 1.6 * radius && r <= 2.6 * radius) ring.push(gray[y * n + x]);
        if (r <= (2 / 3) * radius) core.push(gray[y * n + x]);
      }
    }
    if (inner.length < 5 || ring.length < 5 || core.length < 5) return 0;
    const level = percentile(ring, 50);
    const darker = (level - percentile(inner, 10)) / Math.max(level, 20);
    const spread = (percentile(core, 95) - percentile(core, 5)) / 255;
    return level < DARK_RING ? Math.max(darker, spread) : darker;
  }) as [number, number];
}

/** Frame turned k quarter turns clockwise (lossless) and the 3x3 map of its pixel coordinates. */
export function uprightFrame(frame: Frame, k: number): { frame: Frame; matrix: Mat3 } {
  const w = frame.width, h = frame.height;
  k = ((k % 4) + 4) % 4;
  if (k === 0) return { frame, matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]] };
  const matrix = k === 1 ? [[0, -1, h - 1], [1, 0, 0], [0, 0, 1]]
    : k === 2 ? [[-1, 0, w - 1], [0, -1, h - 1], [0, 0, 1]] : [[0, 1, 0], [-1, 0, w - 1], [0, 0, 1]];
  return { frame: rotate(frame, 90 * k), matrix };
}

function boundsOf(points: Point[]): Box {
  const xs = points.map((p) => p[0]), ys = points.map((p) => p[1]);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}

const corners = (b: Box): Point[] => [[b[0], b[1]], [b[2], b[1]], [b[0], b[3]], [b[2], b[3]]];

/** Detection mapped by a 3x3 pixel map (axis-aligned box of the mapped box). */
export function transformDetection(d: Detection, m: Mat3): Detection {
  return { ...d, box: boundsOf(apply(m, corners(d.box))), keypoints: apply(m, d.keypoints),
    bodyBox: d.bodyBox ? boundsOf(apply(m, corners(d.bodyBox))) : d.bodyBox, extra: { ...(d.extra ?? {}) } };
}

export interface Verification { score: number; eyes: [number, number]; mirror: number; agreement: number }

interface Roi { box: Box; region: Box; angle: number; searchedAt: number; searchedFrame: number; score: number; misses: number }

export class FaceRefiner {
  rois: Roi[] = [];
  readonly size: number;
  private frame = 0;

  constructor(private detector: FaceDetector, private config: RefineConfig) {
    this.size = detector.size ?? 320;
  }

  reset() { this.rois = []; }

  crop(frame: Frame, box: Box, angle: number): { image: Frame; matrix: Mat3 } {
    const matrix = cropMatrix(box, angle, this.config.share, this.size);
    return { image: warpAffine(frame, matrix.slice(0, 2), this.size, this.size, 114), matrix };
  }

  private pick(found: Detection[]): Detection | null {
    const c = this.size / 2;
    let best: Detection | null = null, bestKey = -Infinity;
    for (const d of found) {
      if (detectionReason(d) !== null) continue;
      const key = d.score / (1 + Math.hypot((d.box[0] + d.box[2]) / 2 - c, (d.box[1] + d.box[3]) / 2 - c) / (0.25 * this.size));
      if (key > bestKey) { bestKey = key; best = d; }
    }
    return best;
  }

  /** Detect on the crop turned `angle`; a Detection in frame coordinates or null. */
  async examine(frame: Frame, box: Box, angle: number): Promise<Detection | null> {
    const { image, matrix } = this.crop(frame, box, angle);
    const d = this.pick(await this.detector.detect(image));
    if (!d) return null;
    const inverse = invert(matrix);
    const tilt = eyeLineDeg(d), upright = wrapDeg(angle - tilt);
    // the face's own extent (square, centred on it): the next crop is built from this, not from
    // the bound of a rotated rectangle, which would grow frame after frame
    const [[cx, cy]] = apply(inverse, [[(d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2]]);
    const half = Math.max(d.box[2] - d.box[0], d.box[3] - d.box[1]) / 2 / Math.hypot(matrix[0][0], matrix[0][1]);
    return { box: boundsOf(apply(inverse, corners(d.box))), score: d.score, keypoints: apply(inverse, d.keypoints),
      schema: d.schema, kpScores: d.kpScores, source: `${d.source || 'detector'}+refined`,
      extra: { upright_angle: upright, upright_k: ((Math.round(upright / 90) % 4) + 4) % 4, crop_angle: wrapDeg(angle),
        crop_tilt: tilt, crop_score: d.score, region: [cx - half, cy - half, cx + half, cy + half] as Box } };
  }

  /**
   * Open-loop second look before a shot: the crop turned by the nearest *quarter* turn,
   * not re-rotated to this prediction's own eye line (that can lock a hallucination in).
   */
  async verify(frame: Frame, d: Detection): Promise<Verification> {
    const quarter = 90 * Math.round(Number(d.extra?.upright_angle ?? 0) / 90);
    const { image, matrix } = this.crop(frame, (d.extra?.region as Box | undefined) ?? d.box, quarter);
    const check = this.pick(await this.detector.detect(image));
    const ie = d.schema.names.indexOf('left_eye'), je = d.schema.names.indexOf('right_eye');
    const own = apply(matrix, d.keypoints);
    const span = Math.max(Math.hypot(own[ie][0] - own[je][0], own[ie][1] - own[je][1]), 1e-6);
    if (!check) return { score: 0, eyes: [0, 0], mirror: Infinity, agreement: Infinity };
    const mean = (a: Point[], b: Point[]) => a.reduce((s, p, i) => s + Math.hypot(p[0] - b[i][0], p[1] - b[i][1]), 0) / a.length;
    const other = this.pick(await this.detector.detect(mirror(image)));
    let mirrorGap = Infinity;
    if (other) {
      const flip = flipIndex(other.schema);
      const back = flip.map((j) => [this.size - 1 - other.keypoints[j][0], other.keypoints[j][1]] as Point);
      mirrorGap = mean(check.keypoints, back) / span;
    }
    const conf = check.kpScores ?? check.keypoints.map(() => 1);
    return { score: check.score, eyes: [conf[ie], conf[je]], mirror: mirrorGap, agreement: mean(check.keypoints, own) / span };
  }

  private async search(frame: Frame, box: Box, base: number, baseDetection: Detection | null): Promise<Detection | null> {
    const results = new Map<number, Detection>();
    if (baseDetection) results.set(0, baseDetection);
    for (let j = 1; j < 4; j++) {
      const d = await this.examine(frame, box, base + 90 * j);
      if (d) results.set(j, d);
    }
    if (!results.size) return null;
    const best = [...results.values()].reduce((a, b) => (b.score > a.score ? b : a));
    const current = results.get(0);
    return current && current.score + this.config.switch_margin >= best.score ? current : best;
  }

  private async settle(frame: Frame, box: Box, d: Detection | null): Promise<Detection | null> {
    if (!d || Math.abs(Number(d.extra!.crop_tilt)) <= this.config.fine_min_deg) return d;
    const again = await this.examine(frame, box, Number(d.extra!.upright_angle));
    return again && again.score >= d.score - 0.05 ? again : d;
  }

  /** Refined detections for this frame: full-frame ones plus faces carried from the last frame. */
  /** fullRan: the full-frame detector ran on this frame (then a carried face is a real miss). */
  async refine(frame: Frame, detections: Detection[], t: number, still = false, fullRan = true): Promise<Detection[]> {
    const c = this.config;
    this.frame += 1;
    const previous = still ? [] : this.rois;
    const boxes: [Box, Detection | null][] = detections.map((d) => [d.box, d]);
    for (const roi of previous) {
      if (roi.misses >= c.carry_frames) continue;
      if (detections.some((d) => iou(roi.box, d.box) >= c.match_iou)) continue;
      boxes.push([roi.region, null]);   // carried: the full-frame detector missed it
    }
    const out: Detection[] = [], rois: Roi[] = [];
    for (const [box, source] of boxes) {
      let match: Roi | null = null, bestOverlap = -1;
      for (const roi of previous) {
        const o = iou(box, roi.box);
        if (o > bestOverlap) { bestOverlap = o; match = roi; }
      }
      if (bestOverlap < c.match_iou) match = null;
      const base = match ? match.angle : 0;
      const age = match ? t - match.searchedAt : 0, frames = match ? this.frame - match.searchedFrame : 0;
      // re-check the quarter turn when new or doubtful, else periodically (seconds and frames)
      let d = await this.examine(frame, box, base);
      const due = still || !match || match.score < c.research_below || !d || d.score < c.research_below
        || (frames >= c.search_min_frames
          && ((age >= c.search_period_s && match.score < c.confident_score) || age >= c.max_search_period_s));
      const searchedAt = due ? t : match!.searchedAt;
      if (due) {
        d = await this.search(frame, box, base, d);
        d = await this.settle(frame, box, d);
      }
      if (!d) {
        if (source) out.push({ ...source, extra: { ...(source.extra ?? {}), upright_k: 0, refined: false } });
        if (match) rois.push({ ...match, misses: match.misses + 1 });
        continue;
      }
      if (source) { d.bodyBox = source.bodyBox; d.bodyScore = source.bodyScore; d.extra!.full_score = source.score; }
      d.extra!.refined = true;
      d.extra!.carried = source === null;
      out.push(d);
      const confirmed = d.score >= c.research_below, before = match ? match.misses : 0;
      // missed by a detector that ran (or a weak crop): count it; skipped detector: the crop vouches
      const misses = source ? 0 : (fullRan || !confirmed ? before + 1 : before);
      rois.push({ box: d.box, region: d.extra!.region as Box, angle: Number(d.extra!.upright_angle), searchedAt,
        searchedFrame: due ? this.frame : (match?.searchedFrame ?? 0), score: d.score, misses });
    }
    if (!still) this.rois = rois;
    return dedupe(out);
  }
}

function dedupe(detections: Detection[], threshold = 0.5): Detection[] {
  const kept: Detection[] = [];
  for (const d of [...detections].sort((a, b) => b.score - a.score)) {
    if (!kept.some((k) => iou(d.box, k.box) >= threshold)) kept.push(d);
  }
  return kept;
}
