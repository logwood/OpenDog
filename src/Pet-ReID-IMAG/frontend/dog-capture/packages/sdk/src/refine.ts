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
import { crop as cropFrame, mirror, resize, rotate, toGray, warpAffine, type Frame } from './image';
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

/**
 * The size x size crop `matrix` maps to (border 114), port of refine.warp_crop: downsampling more
 * than 2x is done by area averaging the covered region first and warping at ~1:1 (bilinear point
 * sampling at 10-18x aliases, and the model then sees noise that moves with the box).
 */
export function warpCrop(frame: Frame, matrix: Mat3, size: number, antialias = true): Frame {
  const scale = Math.hypot(matrix[0][0], matrix[0][1]);
  if (antialias && scale < 0.5) {
    const corners = apply(invert(matrix), [[0, 0], [size, 0], [0, size], [size, size]]);
    const xs = corners.map((p) => p[0]), ys = corners.map((p) => p[1]);
    const x1 = Math.max(0, Math.floor(Math.min(...xs)) - 1), y1 = Math.max(0, Math.floor(Math.min(...ys)) - 1);
    const x2 = Math.min(frame.width, Math.ceil(Math.max(...xs)) + 1), y2 = Math.min(frame.height, Math.ceil(Math.max(...ys)) + 1);
    if (x2 - x1 >= 2 && y2 - y1 >= 2) {
      const nw = Math.max(1, Math.floor((x2 - x1) * scale + 0.5)), nh = Math.max(1, Math.floor((y2 - y1) * scale + 0.5));
      const small = resize(cropFrame(frame, x1, y1, x2, y2), nw, nh);
      const rx = nw / (x2 - x1), ry = nh / (y2 - y1);
      // pixel centres: small x = rx * (u - x1 + 0.5) - 0.5   (cv2 INTER_AREA)
      const back: Mat3 = [[1 / rx, 0, x1 + 0.5 / rx - 0.5], [0, 1 / ry, y1 + 0.5 / ry - 0.5], [0, 0, 1]];
      return warpAffine(small, matmul(matrix, back).slice(0, 2), size, size, 114);
    }
  }
  return warpAffine(frame, matrix.slice(0, 2), size, size, 114);
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

interface Roi {
  box: Box; region: Box; angle: number; searchedAt: number; searchedFrame: number; backoff: number; score: number;
  misses: number;
}

/** The per-frame crop network (detectors.CropLandmarker): one face at any roll on a canonical crop. */
export interface Landmarker extends FaceDetector {
  readonly size: number; readonly share: number; readonly keypointMaxRoll: number; readonly rotationInvariant: boolean;
}

/**
 * Per-face region of interest (port of refine.FaceRefiner).  With a `landmarker` it does every
 * per-frame crop run (tracker profile) and the four-way search is not needed; the detector keeps
 * the precise profile: `precise()` on shot candidates and `verify()` (calibrated on it).
 */
export class FaceRefiner {
  rois: Roi[] = [];
  /** Searches that found no face at any turn: [t, frame, cx, cy, size] (refine.empty_search_*). */
  empty: [number, number, number, number, number][] = [];
  readonly size: number;
  readonly preciseSize: number;
  readonly share: number;
  private frame = 0;

  constructor(private detector: FaceDetector, private config: RefineConfig, readonly landmarker: Landmarker | null = null) {
    const native = detector.size ?? 320, wanted = config.crop_size || native;
    this.preciseSize = wanted === native || detector.runsAt?.(wanted) ? wanted : native;
    this.size = landmarker ? landmarker.size : this.preciseSize;
    this.share = landmarker?.share || config.share;
  }

  private get rotationInvariant() { return !!this.landmarker?.rotationInvariant; }

  /** (size, share) of the crops for the tracker or the precise model. */
  private profile(precise: boolean): [number, number] {
    return precise || !this.landmarker ? [this.preciseSize, this.config.share] : [this.size, this.share];
  }

  /** The crop model: the landmarker (tracker profile), else the face detector (refine.crop_size). */
  private detect(image: Frame, precise = false) {
    if (this.landmarker && !precise) return this.landmarker.detect(image);
    return this.preciseSize === (this.detector.size ?? this.preciseSize) ? this.detector.detect(image)
      : this.detector.detect(image, this.preciseSize);
  }

  reset() { this.rois = []; this.empty = []; }

  /** A search found no face at this place a moment ago. */
  private searchedEmpty(box: Box, t: number) {
    const c = this.config;
    if (c.empty_search_period_s <= 0) return false;
    this.empty = this.empty.filter((e) => t - e[0] < c.empty_search_period_s || this.frame - e[1] < c.empty_search_min_frames);
    const [cx, cy, size] = boxPlace(box);
    return this.empty.some((e) => Math.hypot(e[2] - cx, e[3] - cy) <= 0.5 * Math.max(e[4], size));
  }

  crop(frame: Frame, box: Box, angle: number, precise = false): { image: Frame; matrix: Mat3 } {
    const [size, share] = this.profile(precise);
    const matrix = cropMatrix(box, angle, share, size);
    const mode = this.config.antialias;
    const smooth = mode === 'all' || (mode === 'tracker' && !!this.landmarker && !precise);
    return { image: warpCrop(frame, matrix, size, smooth), matrix };
  }

  /** The face at the crop centre (the one being refined), not a neighbour. */
  private pick(found: Detection[], size: number): Detection | null {
    const c = size / 2;
    let best: Detection | null = null, bestKey = -Infinity;
    for (const d of found) {
      if (detectionReason(d) !== null) continue;
      const key = d.score / (1 + Math.hypot((d.box[0] + d.box[2]) / 2 - c, (d.box[1] + d.box[3]) / 2 - c) / (0.25 * size));
      if (key > bestKey) { bestKey = key; best = d; }
    }
    return best;
  }

  private async look(image: Frame, precise = false) { return this.pick(await this.detect(image, precise), image.width); }

  /** Detect on the crop turned `angle`; a Detection in frame coordinates or null. */
  async examine(frame: Frame, box: Box, angle: number, precise = false): Promise<Detection | null> {
    const { image, matrix } = this.crop(frame, box, angle, precise);
    const d = await this.look(image, precise);
    if (!d) return null;
    const inverse = invert(matrix);
    const roll = precise ? undefined : d.extra?.roll_deg as number | undefined;
    // a landmarker beyond its keypoint range: its roll head says which way is up
    const tilt = roll !== undefined && Math.abs(roll) > (this.landmarker?.keypointMaxRoll ?? 45) ? roll : eyeLineDeg(d);
    const upright = wrapDeg(angle - tilt);
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
    const { image, matrix } = this.crop(frame, (d.extra?.region as Box | undefined) ?? d.box, quarter, true);
    const check = await this.look(image, true);
    const ie = d.schema.names.indexOf('left_eye'), je = d.schema.names.indexOf('right_eye');
    const own = apply(matrix, d.keypoints);
    const span = Math.max(Math.hypot(own[ie][0] - own[je][0], own[ie][1] - own[je][1]), 1e-6);
    if (!check) return { score: 0, eyes: [0, 0], mirror: Infinity, agreement: Infinity };
    const mean = (a: Point[], b: Point[]) => a.reduce((s, p, i) => s + Math.hypot(p[0] - b[i][0], p[1] - b[i][1]), 0) / a.length;
    const other = await this.look(mirror(image), true);
    let mirrorGap = Infinity;
    if (other) {
      const flip = flipIndex(other.schema);
      const back = flip.map((j) => [image.width - 1 - other.keypoints[j][0], other.keypoints[j][1]] as Point);
      mirrorGap = mean(check.keypoints, back) / span;
    }
    const conf = check.kpScores ?? check.keypoints.map(() => 1);
    return { score: check.score, eyes: [conf[ie], conf[je]], mirror: mirrorGap, agreement: mean(check.keypoints, own) / span };
  }

  /**
   * Shot candidate with a landmarker: the face detector once on the tracker's upright crop (plus a
   * re-crop if its eye line still tilts more than fine_min_deg).  Without one: `d` itself.
   */
  async precise(frame: Frame, d: Detection): Promise<Detection | null> {
    if (!this.landmarker || d.source === 'external') return d;
    const region = (d.extra?.region as Box | undefined) ?? d.box;
    let sharp = await this.examine(frame, region, Number(d.extra?.upright_angle ?? 0), true);
    if (sharp && Math.abs(Number(sharp.extra!.crop_tilt)) > this.config.fine_min_deg) {
      const again = await this.examine(frame, region, Number(sharp.extra!.upright_angle), true);
      if (again && again.score >= sharp.score - 0.05) sharp = again;
    }
    if (!sharp) return null;
    sharp.bodyBox = d.bodyBox; sharp.bodyScore = d.bodyScore;
    for (const k of ['full_score', 'carried']) if (d.extra && k in d.extra) sharp.extra![k] = d.extra[k];
    sharp.extra!.refined = true;
    sharp.extra!.tracker_score = d.score;
    return sharp;
  }

  private async search(frame: Frame, box: Box, base: number, baseDetection: Detection | null): Promise<Detection | null> {
    if (this.rotationInvariant) {   // the landmarker reads any roll: the base crop is the answer
      if (baseDetection) { baseDetection.extra!.search_margin = 1; baseDetection.extra!.search_kept = true; }
      return baseDetection;
    }
    const results = new Map<number, Detection>();
    if (baseDetection) results.set(0, baseDetection);
    for (let j = 1; j < 4; j++) {
      const d = await this.examine(frame, box, base + 90 * j);
      if (d) results.set(j, d);
    }
    if (!results.size) return null;
    const best = [...results.values()].reduce((a, b) => (b.score > a.score ? b : a));
    const current = results.get(0);
    const chosen = current && current.score + this.config.switch_margin >= best.score ? current : best;
    const others = [...results.values()].filter((d) => d !== chosen).map((d) => d.score);
    chosen.extra!.search_margin = chosen.score - Math.max(0, ...others);
    chosen.extra!.search_kept = chosen === current;
    return chosen;
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
      const backoff = match ? match.backoff : 1;
      // re-check the quarter turn when new or doubtful, else periodically (seconds and frames, stretched
      // while the turn keeps being confirmed clearly)
      let d = await this.examine(frame, box, base);
      let due = still || !match || match.score < c.research_below || !d || d.score < c.research_below
        || (frames >= c.search_min_frames * backoff
          && ((age >= c.search_period_s * backoff && match.score < c.confident_score) || age >= c.max_search_period_s));
      // this crop is unusable and no turn here showed a face a moment ago: wait (empty_search_*)
      if (due && !still && (!d || d.score < c.research_below) && this.searchedEmpty(box, t)) due = false;
      const fresh = due || !match;
      const searchedAt = fresh ? t : match!.searchedAt;
      if (due) {
        d = await this.search(frame, box, base, d);
        d = await this.settle(frame, box, d);
        if (!d) this.empty.push([t, this.frame, ...boxPlace(box)]);
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
        searchedFrame: fresh ? this.frame : (match?.searchedFrame ?? 0),
        backoff: due ? nextBackoff(c, backoff, d) : backoff, score: d.score, misses });
    }
    if (!still) this.rois = rois;
    return dedupe(out);
  }
}

/** Centre and long side of a box (where a face is, independent of track ids). */
function boxPlace(box: Box): [number, number, number] {
  return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, Math.max(box[2] - box[0], box[3] - box[1])];
}

/** Next search-interval multiplier after a search: grows while the turn is clearly confirmed. */
function nextBackoff(c: RefineConfig, backoff: number, d: Detection) {
  return d.extra?.search_kept && Number(d.extra?.search_margin ?? 0) >= c.backoff_margin ? Math.min(backoff * 2, c.max_backoff) : 1;
}

function dedupe(detections: Detection[], threshold = 0.5): Detection[] {
  const kept: Detection[] = [];
  for (const d of [...detections].sort((a, b) => b.score - a.score)) {
    if (!kept.some((k) => iou(d.box, k.box) >= threshold)) kept.push(d);
  }
  return kept;
}
