/** Detector-to-engine contract: one face box, its score and named keypoints (see schema.py). */
export interface KeypointSchema {
  name: string;
  names: string[];
  pairs: [string, string][];
  /** raw yaw cue of this schema's nose / reference (DogFLW nose-pad centre) yaw cue */
  yawScale: number;
}

export const ANYFACE5: KeypointSchema = {
  name: 'anyface5', names: ['left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth'],
  pairs: [['left_eye', 'right_eye'], ['left_mouth', 'right_mouth']], yawScale: 1.0,
};
/** yawScale fitted on the student's predictions (see schema.py DOGFACE6); a model sidecar may override it. */
export const DOGFACE6: KeypointSchema = {
  name: 'dogface6', names: ['left_eye', 'right_eye', 'nose', 'chin', 'left_ear_base', 'right_ear_base'],
  pairs: [['left_eye', 'right_eye'], ['left_ear_base', 'right_ear_base']], yawScale: 1.05,
};
export const CORE3: KeypointSchema = {
  name: 'core3', names: ['left_eye', 'right_eye', 'nose'], pairs: [['left_eye', 'right_eye']], yawScale: 1.0,
};
export const SCHEMAS: Record<string, KeypointSchema> = { anyface5: ANYFACE5, dogface6: DOGFACE6, core3: CORE3 };

export function getSchema(value: string | KeypointSchema): KeypointSchema {
  if (typeof value !== 'string') return value;
  const schema = SCHEMAS[value];
  if (!schema) throw new Error(`Unknown keypoint schema ${value}`);
  return schema;
}

export type Box = [number, number, number, number];
export type Point = [number, number];

export interface Detection {
  box: Box;
  score: number;
  keypoints: Point[];
  schema: KeypointSchema;
  kpScores?: number[] | null;
  bodyBox?: Box | null;
  bodyScore?: number | null;
  source?: string;
  /** refinement results: upright_angle, upright_k, refined, carried, eye_contrast, verify, ... */
  extra?: Record<string, unknown>;
}

export function detectionReason(d: Detection): string | null {
  if (d.keypoints.length !== d.schema.names.length) return 'keypoint_count_mismatch';
  if (d.kpScores && d.kpScores.length !== d.schema.names.length) return 'keypoint_score_count_mismatch';
  const finite = [...d.box, d.score, ...d.keypoints.flat()].every(Number.isFinite);
  if (!finite) return 'nonfinite_detection';
  if (d.box[2] <= d.box[0] || d.box[3] <= d.box[1]) return 'empty_box';
  if (d.score < 0 || d.score > 1) return 'score_out_of_range';
  for (const k of ['left_eye', 'right_eye', 'nose']) if (!d.schema.names.includes(k)) return 'schema_without_eyes_and_nose';
  return null;
}

export const width = (d: { box: Box }) => d.box[2] - d.box[0];
export const height = (d: { box: Box }) => d.box[3] - d.box[1];

export function detectionToJSON(d: Detection) {
  const r2 = (v: number) => Math.round(v * 100) / 100;
  const out: Record<string, unknown> = {
    box: d.box.map(r2), score: Math.round(d.score * 1e4) / 1e4, schema: d.schema.name,
    keypoints: d.keypoints.map(([x, y]) => [r2(x), r2(y)]),
  };
  if (d.kpScores) out.kp_scores = d.kpScores.map((v) => Math.round(v * 1e4) / 1e4);
  if (d.bodyBox) { out.body_box = d.bodyBox.map(r2); out.body_score = d.bodyScore ?? null; }
  if (d.source) out.source = d.source;
  for (const key of ['upright_k', 'upright_angle', 'refined', 'eye_contrast', 'verify']) {
    if (d.extra && key in d.extra) out[key] = plain(d.extra[key]);
  }
  return out;
}

/** JSON-friendly copy of a small diagnostic value (numbers rounded, non-finite -> null). */
function plain(item: unknown): unknown {
  if (Array.isArray(item)) return item.map(plain);
  if (item && typeof item === 'object') return Object.fromEntries(Object.entries(item).map(([k, v]) => [k, plain(v)]));
  if (typeof item === 'number') return Number.isFinite(item) ? Math.round(item * 1e4) / 1e4 : null;
  return item;
}

/** IoU of two xyxy boxes, computed in float32 like the reference. */
export function iou(a: Box, b: Box): number {
  const f = Math.fround;
  const x1 = Math.max(f(a[0]), f(b[0])), y1 = Math.max(f(a[1]), f(b[1]));
  const x2 = Math.min(f(a[2]), f(b[2])), y2 = Math.min(f(a[3]), f(b[3]));
  const inter = f(Math.max(0, f(x2 - x1)) * Math.max(0, f(y2 - y1)));
  const areaA = f(f(a[2] - a[0]) * f(a[3] - a[1])), areaB = f(f(b[2] - b[0]) * f(b[3] - b[1]));
  return f(inter / Math.max(f(areaA + areaB - inter), 1e-6));
}

/** Greedy NMS; returns kept indices sorted by score. */
export function nms(boxes: Box[], scores: number[], threshold: number, maxOutput = 0): number[] {
  let order = scores.map((_, i) => i).sort((a, b) => scores[b] - scores[a]);
  const keep: number[] = [];
  while (order.length) {
    const i = order[0];
    keep.push(i);
    if (maxOutput && keep.length >= maxOutput) break;
    order = order.slice(1).filter((j) => iou(boxes[i], boxes[j]) <= threshold);
  }
  return keep;
}
