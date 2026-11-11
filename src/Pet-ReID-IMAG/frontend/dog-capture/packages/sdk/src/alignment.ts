/** Five-point similarity alignment to the PetFace dog template (port of alignment.py). */
import { crop, resize, warpAffine, type Frame } from './image';
import type { Point } from './schema';

export const PETFACE_DOG_224: Record<string, Point> = {
  left_eye: [56.0, 94.75487264], right_eye: [168.0, 94.57845561], nose: [112.2124176, 150.36327108],
  left_mouth: [60.33797073, 165.05586751], right_mouth: [163.13275146, 165.04858398],
};
export const TEMPLATE_SIZE = 224;
const ORDER = ['left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth'];

/** Least-squares similarity (Umeyama without reflection), closed form in 2-D. */
export function similarity(src: Point[], dst: Point[]): number[][] {
  const n = src.length;
  const ms = [0, 0], md = [0, 0];
  for (let i = 0; i < n; i++) { ms[0] += src[i][0] / n; ms[1] += src[i][1] / n; md[0] += dst[i][0] / n; md[1] += dst[i][1] / n; }
  let sxx = 0, a = 0, b = 0;
  for (let i = 0; i < n; i++) {
    const sx = src[i][0] - ms[0], sy = src[i][1] - ms[1], dx = dst[i][0] - md[0], dy = dst[i][1] - md[1];
    sxx += sx * sx + sy * sy;
    a += sx * dx + sy * dy;
    b += sx * dy - sy * dx;
  }
  if (sxx < 1e-12) throw new Error('Degenerate source points');
  a /= sxx; b /= sxx;
  return [[a, -b, md[0] - (a * ms[0] - b * ms[1])], [b, a, md[1] - (b * ms[0] + a * ms[1])]];
}

export interface AlignedFace {
  image: Frame;
  matrix: number[][];
  scale: number;
  residual: number;
}

export function alignFace(frame: Frame, points: Record<string, Point>, outSize = TEMPLATE_SIZE,
  template = PETFACE_DOG_224, templateSize = TEMPLATE_SIZE): AlignedFace {
  const names = ORDER.filter((k) => k in points && k in template);
  if (names.length < 3) throw new Error('Alignment needs at least eyes and nose');
  const factor = outSize / templateSize;
  const src = names.map((k) => points[k]);
  const dst = names.map((k) => [template[k][0] * factor, template[k][1] * factor] as Point);
  const m = similarity(src, dst);
  const scale = Math.sqrt(Math.abs(m[0][0] * m[1][1] - m[0][1] * m[1][0]));
  const projected = src.map(([x, y]) => [m[0][0] * x + m[0][1] * y + m[0][2], m[1][0] * x + m[1][1] * y + m[1][2]]);
  const le = dst[names.indexOf('left_eye')], re = dst[names.indexOf('right_eye')];
  const span = Math.hypot(le[0] - re[0], le[1] - re[1]);
  const residual = projected.reduce((s, p, i) => s + Math.hypot(p[0] - dst[i][0], p[1] - dst[i][1]), 0) / names.length / span;
  let warp = m, source = frame;
  if (scale < 0.5) {
    // Area-resample a source ROI first, as alignment.py does, so the crop is not aliased.
    const det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    const inv = [[m[1][1] / det, -m[0][1] / det], [-m[1][0] / det, m[0][0] / det]];
    const corners = [[0, 0], [outSize, 0], [0, outSize], [outSize, outSize]].map(([x, y]) => {
      const u = x - m[0][2], v = y - m[1][2];
      return [inv[0][0] * u + inv[0][1] * v, inv[1][0] * u + inv[1][1] * v];
    });
    const x1 = Math.max(0, Math.floor(Math.min(...corners.map((c) => c[0]))) - 2);
    const y1 = Math.max(0, Math.floor(Math.min(...corners.map((c) => c[1]))) - 2);
    const x2 = Math.min(frame.width, Math.ceil(Math.max(...corners.map((c) => c[0]))) + 2);
    const y2 = Math.min(frame.height, Math.ceil(Math.max(...corners.map((c) => c[1]))) + 2);
    if (x2 - x1 > 1 && y2 - y1 > 1) {
      const f = scale / 0.8;
      const rw = Math.max(1, Math.round((x2 - x1) * f)), rh = Math.max(1, Math.round((y2 - y1) * f));
      source = resize(crop(frame, x1, y1, x2, y2), rw, rh);
      const fx = rw / (x2 - x1), fy = rh / (y2 - y1);
      warp = [[m[0][0] / fx, m[0][1] / fy, m[0][0] * x1 + m[0][1] * y1 + m[0][2]],
              [m[1][0] / fx, m[1][1] / fy, m[1][0] * x1 + m[1][1] * y1 + m[1][2]]];
    }
  }
  return { image: warpAffine(source, warp, outSize, outSize), matrix: m, scale, residual };
}

/** Fixed elliptical eyes-nose-mouth region (true = inside), aligned coordinates. */
export function coreMask(size = TEMPLATE_SIZE, template = PETFACE_DOG_224, templateSize = TEMPLATE_SIZE): Uint8Array {
  const f = size / templateSize;
  const eyesY = (template.left_eye[1] + template.right_eye[1]) / 2;
  const mouthY = (template.left_mouth[1] + template.right_mouth[1]) / 2;
  const cx = Math.round((template.left_eye[0] + template.right_eye[0]) / 2 * f);
  const top = eyesY - 26, bottom = mouthY + 16;
  const cy = Math.round((top + bottom) / 2 * f);
  const ax = Math.round((template.right_eye[0] - template.left_eye[0]) * 0.72 * f), ay = Math.round((bottom - top) / 2 * f);
  const mask = new Uint8Array(size * size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const u = (x - cx) / (ax + 0.5), v = (y - cy) / (ay + 0.5);
      if (u * u + v * v <= 1) mask[y * size + x] = 1;
    }
  }
  return mask;
}

export function eyePatches(size = TEMPLATE_SIZE, template = PETFACE_DOG_224, radius = 18, templateSize = TEMPLATE_SIZE) {
  const f = size / templateSize;
  return (['left_eye', 'right_eye'] as const).map((name) => {
    const x = template[name][0] * f, y = template[name][1] * f, r = radius * f;
    return [Math.max(0, Math.round(x - r)), Math.max(0, Math.round(y - r)), Math.min(size, Math.round(x + r)),
      Math.min(size, Math.round(y + r))] as [number, number, number, number];
  });
}
