/**
 * ONNX decoders (port of detectors.py).  Inference is injected through a
 * minimal Runner, so the same code runs on onnxruntime-web (WASM / WebGPU),
 * onnxruntime-react-native, or a fake runner in tests.
 */
import { letterbox, type Frame } from './image';
import { getSchema, nms, SCHEMAS, type Box, type Detection, type KeypointSchema, type Point } from './schema';

export interface TensorOut { data: Float32Array; dims: number[] }
export interface Runner {
  inputSize: number;
  /** The model has a free input size, so it can also run on other square sizes (refine.crop_size). */
  dynamic?: boolean;
  run(input: Float32Array, dims: number[]): Promise<TensorOut[]>;
}

export interface FaceDetector {
  /** `size`: a square input other than the model's own (only where runsAt(size)). */
  detect(frame: Frame, size?: number): Promise<Detection[]>;
  readonly size?: number;
  runsAt?(size: number): boolean;
}
export interface BodyGate { detect(frame: Frame): Promise<{ boxes: Box[]; scores: number[] }> }

function chw(canvas: Frame, order: 'rgb' | 'bgr', scale: number): Float32Array {
  const n = canvas.width * canvas.height;
  const out = new Float32Array(3 * n);
  const idx = order === 'rgb' ? [0, 1, 2] : [2, 1, 0];
  for (let i = 0; i < n; i++) {
    out[i] = canvas.data[i * 4 + idx[0]] * scale;
    out[n + i] = canvas.data[i * 4 + idx[1]] * scale;
    out[2 * n + i] = canvas.data[i * 4 + idx[2]] * scale;
  }
  return out;
}

function finish(boxes: Box[], scores: number[], points: Point[][], kp: number[][] | null, ratio: number,
  pad: [number, number], frame: Frame, schema: KeypointSchema, iouThreshold: number, maxFaces: number, source: string) {
  const keep = nms(boxes, scores, iouThreshold, maxFaces);
  const [px, py] = pad;
  return keep.map((i): Detection => ({
    box: [Math.min(Math.max((boxes[i][0] - px) / ratio, 0), frame.width), Math.min(Math.max((boxes[i][1] - py) / ratio, 0), frame.height),
      Math.min(Math.max((boxes[i][2] - px) / ratio, 0), frame.width), Math.min(Math.max((boxes[i][3] - py) / ratio, 0), frame.height)],
    score: scores[i],
    keypoints: points[i].map(([x, y]) => [(x - px) / ratio, (y - py) / ratio] as Point),
    schema, kpScores: kp ? kp[i] : null, source,
  }));
}

/** Ultralytics YOLO-pose export: [1, 4 + nc + K*D, N] (or end-to-end [1, M, 6 + K*D]). */
export class YoloPoseDetector implements FaceDetector {
  schema: KeypointSchema;
  constructor(private runner: Runner, public options: { kpt: [number, number]; schema?: string; numClasses?: number;
    scoreThreshold?: number; nmsIou?: number; maxFaces?: number; kptNames?: string[]; yawScale?: number }) {
    const [k] = options.kpt;
    this.schema = options.schema ? getSchema(options.schema)
      : schemaFromNames(options.kptNames) ?? getSchema(({ 6: 'dogface6', 5: 'anyface5', 3: 'core3' } as Record<number, string>)[k]);
    if (this.schema.names.length !== k) throw new Error(`${k} keypoints do not fit schema ${this.schema.name}`);
    if (options.yawScale !== undefined) {   // per-model yaw calibration from the model sidecar
      if (!(options.yawScale >= 0.2 && options.yawScale <= 5)) throw new Error(`yaw_scale ${options.yawScale} out of range`);
      this.schema = { ...this.schema, yawScale: options.yawScale };
    }
  }

  /** Square network input of the full-frame pass (the refiner crops at this size unless refine.crop_size). */
  get size() { return this.runner.inputSize; }

  /** True if the model can run on a square input of `size` (a dynamic-shape export can). */
  runsAt(size: number) { return size === this.runner.inputSize || !!this.runner.dynamic; }

  async detect(frame: Frame, size: number = this.runner.inputSize): Promise<Detection[]> {
    const { canvas, ratio, pad } = letterbox(frame, size);
    const [out] = await this.runner.run(chw(canvas, 'rgb', 1 / 255), [1, 3, size, size]);
    const [k, d] = this.options.kpt;
    const nc = this.options.numClasses ?? 1;
    const thr = this.options.scoreThreshold ?? 0.35;
    const boxes: Box[] = [], scores: number[] = [], points: Point[][] = [], vis: number[][] = [];
    const [, a, b] = out.dims;
    if (b === 6 + k * d && a <= 1000) {           // end-to-end export: rows are detections
      for (let i = 0; i < a; i++) {
        const row = out.data.subarray(i * b, (i + 1) * b);
        if (row[4] < thr) continue;
        boxes.push([row[0], row[1], row[2], row[3]]); scores.push(row[4]);
        points.push([...Array(k).keys()].map((j) => [row[6 + j * d], row[7 + j * d]] as Point));
        if (d === 3) vis.push([...Array(k).keys()].map((j) => row[8 + j * d]));
      }
    } else {                                        // channels x anchors
      const channels = a, anchors = b;
      const at = (c: number, i: number) => out.data[c * anchors + i];
      for (let i = 0; i < anchors; i++) {
        let score = -Infinity;
        for (let c = 0; c < nc; c++) score = Math.max(score, at(4 + c, i));
        if (score < thr) continue;
        const cx = at(0, i), cy = at(1, i), w = at(2, i), h = at(3, i);
        boxes.push([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]); scores.push(score);
        const base = 4 + nc;
        points.push([...Array(k).keys()].map((j) => [at(base + j * d, i), at(base + j * d + 1, i)] as Point));
        if (d === 3) vis.push([...Array(k).keys()].map((j) => at(base + j * d + 2, i)));
        void channels;
      }
    }
    return finish(boxes, scores, points, d === 3 ? vis : null, ratio, pad, frame, this.schema,
      this.options.nmsIou ?? 0.45, this.options.maxFaces ?? 8, 'yolo_pose');
  }
}

/**
 * The crop landmark model (tools/train_dog_face_landmarks.py; port of detectors.CropLandmarker):
 * one dog face centred on a canonical crop, at any roll.  Output [1, 3K + 6]: keypoints (x, y) * K,
 * face box centre x, y and long side, presence logit, visibility logits * K, cos/sin of the roll.
 */
export class CropLandmarker implements FaceDetector {
  readonly rotationInvariant = true;
  schema: KeypointSchema;
  constructor(private runner: Runner, public options: { schema?: string; share?: number; keypointMaxRoll?: number;
    scoreThreshold?: number; yawScale?: number } = {}) {
    this.schema = getSchema(options.schema || 'dogface6');
    if (options.yawScale !== undefined) this.schema = { ...this.schema, yawScale: options.yawScale };
  }

  get size() { return this.runner.inputSize; }
  get share() { return this.options.share ?? 0.35; }
  get keypointMaxRoll() { return this.options.keypointMaxRoll ?? 45; }
  runsAt(size: number) { return size === this.runner.inputSize; }

  async detect(crop: Frame): Promise<Detection[]> {
    const size = this.runner.inputSize;
    if (crop.width !== size || crop.height !== size) throw new Error(`CropLandmarker takes ${size} px crops`);
    const [out] = await this.runner.run(chw(crop, 'rgb', 1 / 255), [1, 3, size, size]);
    const k = this.schema.names.length, o = out.data;
    if (o.length !== 3 * k + 6) throw new Error(`expected ${3 * k + 6} landmark outputs, got ${o.length}`);
    const sigmoid = (v: number) => 1 / (1 + Math.exp(-v));
    const presence = sigmoid(o[2 * k + 3]);
    const [cx, cy, side] = [o[2 * k], o[2 * k + 1], o[2 * k + 2]];
    if (presence < (this.options.scoreThreshold ?? 0.05) || !(side > 0)) return [];
    const keypoints = [...Array(k).keys()].map((j) => [o[2 * j], o[2 * j + 1]] as Point);
    const kpScores = [...Array(k).keys()].map((j) => sigmoid(o[2 * k + 4 + j]));
    return [{ box: [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2], score: presence, keypoints,
      schema: this.schema, kpScores, source: 'crop_landmarks',
      extra: { roll_deg: Math.atan2(o[3 * k + 5], o[3 * k + 4]) * 180 / Math.PI } }];
  }
}

/** Ultralytics stores kpt_names as "{0: ['left_eye', ...]}"; use it when it names eyes and nose. */
export function schemaFromNames(names?: string[]): KeypointSchema | null {
  if (!names?.length) return null;
  for (const s of Object.values(SCHEMAS)) if (s.names.join() === names.join()) return s;
  if (!['left_eye', 'right_eye', 'nose'].every((n) => names.includes(n))) return null;
  const pairs = names.filter((n) => n.startsWith('left_') && names.includes('right_' + n.slice(5)))
    .map((n) => [n, 'right_' + n.slice(5)] as [string, string]);
  return { name: 'custom_' + names.join('_').slice(0, 40), names, pairs, yawScale: 1 };
}

/** YOLOX COCO detector used as the dog species gate (raw BGR 0..255, top-left letterbox). */
export class YoloxBodyGate implements BodyGate {
  private grid: Float32Array;
  private stride: Float32Array;
  constructor(private runner: Runner, private options: { classId?: number; scoreThreshold?: number } = {}) {
    const size = runner.inputSize, cells: number[] = [], strides: number[] = [];
    for (const s of [8, 16, 32]) {
      const n = Math.floor(size / s);
      for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) { cells.push(x, y); strides.push(s); }
    }
    this.grid = new Float32Array(cells); this.stride = new Float32Array(strides);
  }

  async detect(frame: Frame) {
    const size = this.runner.inputSize;
    const { canvas, ratio } = letterbox(frame, size, false);
    const [out] = await this.runner.run(chw(canvas, 'bgr', 1), [1, 3, size, size]);
    const [, n, width] = out.dims;
    const cls = this.options.classId ?? 16, thr = this.options.scoreThreshold ?? 0.3;
    const boxes: Box[] = [], scores: number[] = [];
    for (let i = 0; i < n; i++) {
      const row = out.data.subarray(i * width, (i + 1) * width);
      const score = row[4] * row[5 + cls];
      if (score < thr) continue;
      let best = 0;
      for (let c = 1; c < width - 5; c++) if (row[5 + c] > row[5 + best]) best = c;
      if (best !== cls) continue; // the dog class must win, not merely pass
      const s = this.stride[i];
      const cx = (row[0] + this.grid[2 * i]) * s, cy = (row[1] + this.grid[2 * i + 1]) * s;
      const w = Math.exp(Math.min(Math.max(row[2], -10), 10)) * s, h = Math.exp(Math.min(Math.max(row[3], -10), 10)) * s;
      boxes.push([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]); scores.push(score);
    }
    const keep = nms(boxes, scores, 0.45, 16);
    return {
      boxes: keep.map((i) => boxes[i].map((v, j) => Math.min(Math.max(v / ratio, 0), j % 2 ? frame.height : frame.width)) as Box),
      scores: keep.map((i) => scores[i]),
    };
  }
}

/** Build a Runner on onnxruntime-web (or any API-compatible `ort` module). */
export async function createOrtRunner(ort: {
  InferenceSession: { create(model: ArrayBuffer | Uint8Array | string, options?: object): Promise<any> };
  Tensor: new (type: string, data: Float32Array, dims: number[]) => any;
}, model: ArrayBuffer | Uint8Array | string, providers: string[] = ['webgpu', 'wasm']): Promise<Runner & {
  providers: string[]; metadata: Record<string, string>; release(): Promise<void>;
}> {
  let session: any = null, used: string[] = [];
  let lastError: unknown = null;
  for (let i = 0; i < providers.length && !session; i++) {
    try {
      session = await ort.InferenceSession.create(model as ArrayBuffer, { executionProviders: providers.slice(i),
        graphOptimizationLevel: 'all' });
      used = providers.slice(i);
    } catch (error) { lastError = error; }
  }
  if (!session) throw new Error(`Could not create an ONNX session: ${String(lastError)}`);
  const inputName = session.inputNames[0];
  const meta = session.inputMetadata?.[0] ?? session.inputMetadata?.[inputName];
  const shape: unknown[] = meta?.shape ?? meta?.dimensions ?? [];
  const dynamic = typeof shape[2] !== 'number' || typeof shape[3] !== 'number';
  const inputSize = dynamic ? 320 : (shape[2] as number);
  return {
    inputSize, dynamic, providers: used, metadata: {},
    async release() { await session.release?.(); },
    async run(input: Float32Array, dims: number[]) {
      const outputs = await session.run({ [inputName]: new ort.Tensor('float32', input, dims) });
      return session.outputNames.map((name: string) => ({ data: outputs[name].data as Float32Array, dims: outputs[name].dims as number[] }));
    },
  };
}
