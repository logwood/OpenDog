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
  run(input: Float32Array, dims: number[]): Promise<TensorOut[]>;
}

export interface FaceDetector { detect(frame: Frame): Promise<Detection[]> }
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
    scoreThreshold?: number; nmsIou?: number; maxFaces?: number; kptNames?: string[] }) {
    const [k] = options.kpt;
    this.schema = options.schema ? getSchema(options.schema)
      : schemaFromNames(options.kptNames) ?? getSchema(({ 6: 'dogface6', 5: 'anyface5', 3: 'core3' } as Record<number, string>)[k]);
    if (this.schema.names.length !== k) throw new Error(`${k} keypoints do not fit schema ${this.schema.name}`);
  }

  async detect(frame: Frame): Promise<Detection[]> {
    const size = this.runner.inputSize;
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
  providers: string[]; metadata: Record<string, string>;
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
  const inputSize = typeof shape[2] === 'number' ? (shape[2] as number) : 320;
  return {
    inputSize, providers: used, metadata: {},
    async run(input: Float32Array, dims: number[]) {
      const outputs = await session.run({ [inputName]: new ort.Tensor('float32', input, dims) });
      return session.outputNames.map((name: string) => ({ data: outputs[name].data as Float32Array, dims: outputs[name].dims as number[] }));
    },
  };
}
