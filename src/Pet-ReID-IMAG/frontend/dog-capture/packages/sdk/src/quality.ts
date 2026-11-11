/** FIQA-style quality on the 224 x 224 aligned crop (port of quality.py; same constants). */
import { alignFace, coreMask, eyePatches, PETFACE_DOG_224, TEMPLATE_SIZE, type AlignedFace } from './alignment';
import type { CaptureConfig, ViewSpec } from './config';
import { toGray, type Frame } from './image';
import type { FaceGeometry } from './pose';
import type { Detection } from './schema';

export const QUALITY_SIZE = 224;
export const LAPLACIAN_REF = 80;
export const EYE_LAPLACIAN_REF = 100;
export const CRETE_SHARP = 0.30;
export const CRETE_BLURRY = 0.70;

export interface QualityReport {
  score: number;
  level: 'reject' | 'low' | 'medium' | 'high';
  passed: boolean;
  reasons: string[];
  factors: Record<string, number>;
  metrics: Record<string, number | number[]>;
}

const reflect101 = (i: number, n: number) => (n === 1 ? 0 : i < 0 ? -i : i >= n ? 2 * n - 2 - i : i);
const reflect = (i: number, n: number) => (i < 0 ? -i - 1 : i >= n ? 2 * n - 1 - i : i);

/** GaussianBlur(3x3, sigma 0.6) on uint8 then 3x3 Laplacian, variance over mask (or all). */
export function laplacianEnergy(gray: Uint8Array, w: number, h: number, mask?: Uint8Array | null,
  rect?: [number, number, number, number]): number {
  const [x0, y0, x1, y1] = rect ?? [0, 0, w, h];
  const rw = x1 - x0, rh = y1 - y0;
  const g = Math.exp(-1 / (2 * 0.6 * 0.6));
  const k = [g / (1 + 2 * g), 1 / (1 + 2 * g), g / (1 + 2 * g)];
  const at = (x: number, y: number) => gray[(y0 + reflect101(y, rh)) * w + x0 + reflect101(x, rw)];
  const tmp = new Float64Array(rw * rh);
  for (let y = 0; y < rh; y++) for (let x = 0; x < rw; x++) tmp[y * rw + x] = k[0] * at(x - 1, y) + k[1] * at(x, y) + k[2] * at(x + 1, y);
  const smooth = new Uint8Array(rw * rh);
  for (let y = 0; y < rh; y++) for (let x = 0; x < rw; x++) {
    const v = k[0] * tmp[reflect101(y - 1, rh) * rw + x] + k[1] * tmp[y * rw + x] + k[2] * tmp[reflect101(y + 1, rh) * rw + x];
    smooth[y * rw + x] = Math.min(255, Math.max(0, Math.round(v)));
  }
  let sum = 0, sum2 = 0, n = 0;
  for (let y = 0; y < rh; y++) for (let x = 0; x < rw; x++) {
    if (mask && !mask[(y0 + y) * w + x0 + x]) continue;
    const s = (xx: number, yy: number) => smooth[reflect101(yy, rh) * rw + reflect101(xx, rw)];
    const v = s(x - 1, y) + s(x + 1, y) + s(x, y - 1) + s(x, y + 1) - 4 * s(x, y);
    sum += v; sum2 += v * v; n += 1;
  }
  if (!n) return 0;
  const mean = sum / n;
  return sum2 / n - mean * mean;
}

/** Crete et al. 2007 re-blur metric on a rectangle of a gray image (box filter, BORDER_REFLECT). */
export function blurEffect(gray: Uint8Array, w: number, rect: [number, number, number, number], hSize = 11): number {
  const [x0, y0, x1, y1] = rect;
  const rw = x1 - x0, rh = y1 - y0;
  const at = (x: number, y: number) => gray[(y0 + y) * w + x0 + x];
  const half = Math.floor(hSize / 2);
  const values: number[] = [];
  for (const axis of [0, 1]) {
    const blurred = new Float64Array(rw * rh);
    for (let y = 0; y < rh; y++) for (let x = 0; x < rw; x++) {
      let s = 0;
      for (let t = -half; t <= half; t++) {
        s += axis === 0 ? at(x, reflect(y + t, rh)) : at(reflect(x + t, rw), y);
      }
      blurred[y * rw + x] = s / hSize;
    }
    let total = 0, variation = 0;
    const ny = axis === 0 ? rh - 1 : rh, nx = axis === 0 ? rw : rw - 1;
    for (let y = 2; y < ny - 2; y++) for (let x = 2; x < nx - 2; x++) {
      const dx = axis === 0 ? 0 : 1, dy = axis === 0 ? 1 : 0;
      const df = Math.abs(at(x + dx, y + dy) - at(x, y));
      const db = Math.abs(blurred[(y + dy) * rw + x + dx] - blurred[y * rw + x]);
      total += df;
      variation += Math.max(df - db, 0);
    }
    values.push(total <= 1e-6 ? 1 : 1 - variation / total);
  }
  return Math.max(...values);
}

export class QualityAssessor {
  readonly mask: Uint8Array;
  readonly maskRect: [number, number, number, number];
  readonly eyes: [number, number, number, number][];

  constructor(private config: CaptureConfig['quality'], mask?: Uint8Array) {
    this.mask = mask ?? coreMask(QUALITY_SIZE, PETFACE_DOG_224, TEMPLATE_SIZE);
    let x0 = QUALITY_SIZE, y0 = QUALITY_SIZE, x1 = 0, y1 = 0;
    for (let y = 0; y < QUALITY_SIZE; y++) for (let x = 0; x < QUALITY_SIZE; x++) {
      if (this.mask[y * QUALITY_SIZE + x]) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x + 1); y1 = Math.max(y1, y + 1); }
    }
    this.maskRect = [x0, y0, x1, y1];
    this.eyes = eyePatches(QUALITY_SIZE);
  }

  align(frame: Frame, g: FaceGeometry): AlignedFace { return alignFace(frame, g.points, QUALITY_SIZE); }

  assess(frame: Frame, det: Detection, g: FaceGeometry, target: ViewSpec | null, speedPxS = 0, full = true):
    { report: QualityReport; aligned: AlignedFace | null } {
    const c = this.config;
    const report: QualityReport = { score: 0, level: 'reject', passed: false, reasons: [], factors: {}, metrics: {} };
    const m = report.metrics;
    m.iod_px = g.iod; m.detector_score = det.score;
    if (!g.valid) { report.reasons.push(g.reason); return { report, aligned: null }; }
    if (g.iod < c.min_iod_px) report.reasons.push('face_too_small');
    const margin = c.border_margin * Math.max(frame.width, frame.height);
    const core = ['left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth'].filter((k) => k in g.points).map((k) => g.points[k]);
    if (core.some(([x, y]) => x < margin || y < margin || x > frame.width - 1 - margin || y > frame.height - 1 - margin)) {
      report.reasons.push('face_clipped');
    }
    const front = target !== null && target.lower < 0 && 0 < target.upper;
    const kp = g.scores;
    if (Object.keys(kp).length) {
      const eyes = [kp.left_eye ?? 0, kp.right_eye ?? 0];
      const eye = front ? Math.min(...eyes) : Math.max(...eyes);
      m.eye_visibility = eye; m.nose_visibility = kp.nose ?? 0;
      if (!(eye >= c.min_kp_score && (kp.nose ?? 0) >= c.min_kp_score)) report.reasons.push('eyes_or_nose_occluded');
    }
    if (report.reasons.length || !full) { report.passed = full && !report.reasons.length; return { report, aligned: null }; }
    const aligned = this.align(frame, g);
    m.alignment_residual = aligned.residual;
    if (aligned.residual > c.max_alignment_residual) report.reasons.push('alignment_failed');
    const gray = toGray(aligned.image);
    const metrics = this.pixelMetrics(gray);
    Object.assign(m, metrics, { warp_scale: aligned.scale });
    const { brightness, contrast, clipped, laplacian, blur_effect: crete, eye_laplacian: eyeLaps } = metrics;
    const sLap = 1 - Math.exp(-laplacian / LAPLACIAN_REF);
    const sCrete = Math.min(1, Math.max(0, (CRETE_BLURRY - crete) / (CRETE_BLURRY - CRETE_SHARP)));
    const sharpness = 0.5 * (sLap + sCrete);
    const eyeLap = front ? Math.min(...eyeLaps) : Math.max(...eyeLaps);
    const eyeDetail = 1 - Math.exp(-eyeLap / EYE_LAPLACIAN_REF);
    const lo = c.min_brightness, hi = c.max_brightness;
    let exposure = brightness < 80 ? (brightness - lo) / Math.max(80 - lo, 1) : brightness > 170 ? (hi - brightness) / Math.max(hi - 170, 1) : 1;
    exposure = Math.min(1, Math.max(0, exposure)) * (1 - Math.min(clipped / Math.max(c.max_clipped, 1e-6), 1) * 0.5);
    exposure *= Math.min(1, Math.max(0.2, contrast / (2.5 * c.min_contrast)));
    if (sharpness < c.min_sharpness) report.reasons.push('blurred');
    if (brightness < lo) report.reasons.push('too_dark');
    else if (brightness > hi || clipped > c.max_clipped) report.reasons.push('overexposed');
    else if (contrast < c.min_contrast) report.reasons.push('low_contrast');
    const resolution = (g.iod - c.min_iod_px) / (c.good_iod_px - c.min_iod_px);
    let confidence = Math.min(1, Math.max(0, (det.score - 0.2) / 0.6));
    if (Object.keys(kp).length) confidence = Math.sqrt(confidence * ((m.eye_visibility as number) + (m.nose_visibility as number)) / 2);
    const speed = speedPxS / Math.max(g.iod, 1e-6);
    m.speed_iod_per_s = speed;
    const motion = 1 / (1 + (speed / c.motion_ref_iod_per_s) ** 2);
    let fit = 0.5;
    if (target) {
      const half = Math.max(target.center - target.lower, target.upper - target.center);
      fit = (1 - 0.5 * Math.min(Math.abs(g.yaw - target.center) / half, 1)) * (1 - 0.3 * Math.min(Math.abs(g.roll_deg) / target.max_roll_deg, 1));
    }
    const alignment = 1 - Math.min(aligned.residual / c.max_alignment_residual, 1) * 0.8;
    report.factors = { sharpness, eye_detail: eyeDetail, exposure, resolution: Math.min(1, Math.max(0, resolution)),
      confidence, motion, pose_fit: fit, alignment };
    let total = 0, logSum = 0;
    for (const [k, v] of Object.entries(report.factors)) {
      const w = c.weights[k] ?? 0;
      total += w; logSum += w * Math.log(Math.max(v, 0.02));
    }
    report.score = Math.exp(logSum / total);
    report.passed = report.reasons.length === 0;
    if (report.passed) report.level = report.score >= c.high_level ? 'high' : report.score >= c.medium_level ? 'medium' : 'low';
    return { report, aligned };
  }

  /** Brightness/contrast/clipping (uint8 histogram), Laplacian energy, Crete blur, eye-patch energies. */
  pixelMetrics(gray: Uint8Array) {
    const hist = new Uint32Array(256);
    let n = 0, sum = 0;
    for (let i = 0; i < gray.length; i++) if (this.mask[i]) { hist[gray[i]]++; n++; sum += gray[i]; }
    const cumulative = new Float64Array(256);
    let acc = 0;
    for (let v = 0; v < 256; v++) { acc += hist[v]; cumulative[v] = acc / n; }
    const search = (q: number) => { let i = 0; while (i < 255 && cumulative[i] < q) i++; return i; };
    const laplacian = laplacianEnergy(gray, QUALITY_SIZE, QUALITY_SIZE, this.mask);
    const eyeLaplacian = this.eyes.map((rect) => laplacianEnergy(gray, QUALITY_SIZE, QUALITY_SIZE, null, rect));
    return { brightness: sum / n, contrast: search(0.95) - search(0.05), clipped: 1 - cumulative[249] + cumulative[3],
      laplacian, blur_effect: blurEffect(gray, QUALITY_SIZE, this.maskRect), eye_laplacian: eyeLaplacian };
  }
}
