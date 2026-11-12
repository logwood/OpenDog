/** Engine, decoders and protocol client on real reference frames (no browser, no model files). */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { inflateSync } from 'node:zlib';
import { describe, expect, it, vi } from 'vitest';
import {
  buildManifest, CaptureEngine, defaultConfig, encodeBundle, FaceRefiner, fromBGR, getSchema, sha256Hex, SignalRecorder, YoloPoseDetector,
  YoloxBodyGate, type Detection, type Frame, type Runner, type SessionConfig,
} from '../src';

const DIR = resolve(__dirname, '../../../../../dog_capture/protocol/conformance');
const quality = JSON.parse(readFileSync(resolve(DIR, 'quality.json'), 'utf8'));
const unpack = (p: { deflate_b64: string }) => new Uint8Array(inflateSync(Buffer.from(p.deflate_b64, 'base64')));
const frames = Object.fromEntries(quality.frames.map((f: any) => {
  const [h, w] = f.frame_bgr.shape;
  const det: Detection = { box: f.box, score: 0.9, keypoints: f.keypoints, schema: getSchema('anyface5') };
  return [f.name, { frame: fromBGR(unpack(f.frame_bgr), w, h), det }];
})) as Record<string, { frame: Frame; det: Detection }>;

async function run(engine: CaptureEngine, names: string[], start = 0) {
  const results = [];
  for (const [i, name] of names.entries()) {
    const { frame, det } = frames[name];
    results.push(await engine.process(frame, (start + i) / 30, start + i, [{ ...det, keypoints: det.keypoints.map((p) => [...p] as [number, number]) }]));
  }
  return results;
}

describe('CaptureEngine', () => {
  it('captures front and side views from one track, rejects blur/dark', async () => {
    const engine = new CaptureEngine({ pose: { views: [
      { name: 'front', center: 0, lower: -0.13, upper: 0.13, shots: 1 },
      { name: 'right', center: 0.45, lower: 0.25, upper: 0.75, shots: 1 }] } });
    const results = await run(engine, [...Array(8).fill('sharp'), ...Array(8).fill('side')]);
    // right is captured but still inside its improve window and the dog faces it: hold, don't move on
    expect(results.at(-1)!.guidance).toMatchObject({ code: 'hold_still', view: 'right' });
    expect(engine.complete()).toBe(false);
    const later = await run(engine, Array(64).fill('side'), 16);
    expect(later.at(-1)!.guidance.code).toBe('done');
    expect(engine.complete()).toBe(true);
    const bundle = engine.finalize()!;
    expect(bundle.captured()).toEqual(['front', 'right']);
    expect(bundle.best('front')!.frame!.width).toBe(320);
    // the reference accepts the 'dark' frame (brightness 38.6 > 30); halve it to go below the limit
    const darker = { ...frames.dark.frame, data: frames.dark.frame.data.map((v, i) => (i % 4 === 3 ? v : v >> 1)) };
    const dark = new CaptureEngine();
    let last = null;
    for (let i = 0; i < 10; i++) last = await dark.process(darker, i / 30, i, [frames.dark.det]);
    expect(dark.finalize()).toBeNull();
    expect(last!.guidance.code).toBe('too_dark');
  });

  it('drops non-increasing timestamps and keeps memory bounded', async () => {
    const engine = new CaptureEngine();
    const { frame, det } = frames.sharp;
    expect(await engine.process(frame, 1, 0, [det])).not.toBeNull();
    expect(await engine.process(frame, 1, 1, [det])).toBeNull();
    expect(engine.dropped.non_increasing_timestamp).toBe(1);
    for (let i = 0; i < 300; i++) await engine.process(frame, 2 + i / 30, i + 2, i % 50 < 40 ? [det] : []);
    expect(engine.tracker.tracks.length).toBeLessThanOrEqual(2);
    expect(engine.events.length).toBeLessThanOrEqual(1024);
  });

  it('records liveness signals', async () => {
    const engine = new CaptureEngine();
    const recorder = new SignalRecorder();
    for (const r of await run(engine, [...Array(8).fill('sharp'), ...Array(8).fill('side')])) recorder.add(r!);
    const signals = recorder.toJSON(['front', 'right', 'left']);
    expect(signals.single_track).toBe(true);
    expect(signals.view_sequence.map(([v]) => v)).toEqual(['front', 'right']);
    expect(signals.prompt_order_followed).toBe(true);
    expect(signals.yaw_trace.length).toBeGreaterThan(3);
  });
});

it.each([
  { front: true, score: .49, eyes: [.9, .9], offset: 0, skip: 'score' },
  { front: true, score: .9, eyes: [.1, .9], offset: 0, skip: 'eyes' },
  { front: false, score: .9, eyes: [.1, .9], offset: 0, skip: undefined },
  { front: true, score: .9, eyes: [.9, .9], offset: 7, skip: 'agreement' },
  { front: true, score: .5, eyes: [.9, .9], offset: 6, skip: undefined },
])('short-circuits a rejected mirror check: %j', async ({ front, score, eyes, offset, skip }) => {
  const { frame, det } = frames.sharp;
  const own: Detection = { ...det, keypoints: [[40, 40], [60, 40], ...det.keypoints.slice(2)] };
  const checked: Detection = { ...own, score, kpScores: [...eyes, ...own.keypoints.slice(2).map(() => .9)],
    keypoints: own.keypoints.map(([x, y]) => [x + offset, y]) };
  const detect = vi.fn(async () => [checked]);
  const refiner = new FaceRefiner({ detect }, defaultConfig().refine);
  vi.spyOn(refiner, 'crop').mockReturnValue({ image: frame, matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]] });
  const full = await refiner.verify(frame, own);
  expect(detect).toHaveBeenCalledTimes(2);
  detect.mockClear();
  const fast = await refiner.verify(frame, own, front);
  expect(fast.mirror_skipped).toBe(skip);
  expect(detect).toHaveBeenCalledTimes(skip ? 1 : 2);
  for (const key of ['score', 'eyes', 'agreement'] as const) expect(fast[key]).toEqual(full[key]);
  if (!skip) expect(fast).toEqual(full);
});

it.each([null, .25])('reuses the examined base crop when tracking returns %s', async (weakScore) => {
  const { frame, det } = frames.sharp;
  const refiner = new FaceRefiner({ detect: async () => [] }, defaultConfig().refine);
  let baseScore: number | null = .95;
  const examine = vi.spyOn(refiner, 'examine').mockImplementation(async (_frame, box, angle) => {
    const score = angle === 0 ? baseScore : angle === 90 ? .9 : null;
    return score === null ? null : {
      ...det, box, score, extra: { upright_angle: angle, crop_tilt: 0, region: box },
    };
  });
  await refiner.refine(frame, [det], 0);
  examine.mockClear();
  baseScore = weakScore;
  const refined = await refiner.refine(frame, [det], 1 / 30);
  expect(examine.mock.calls.map(([, , angle]) => angle)).toEqual([0, 90, 180, 270]);
  expect(refined[0].extra?.upright_angle).toBe(90);
});

it('waits before repeating an empty orientation search at the same place', async () => {
  const { frame, det } = frames.sharp;
  const config = defaultConfig().refine;
  const refiner = new FaceRefiner({ detect: async () => [] }, config);
  const examine = vi.spyOn(refiner, 'examine').mockImplementation(async () => null);   // nothing anywhere
  const counts: number[] = [];
  for (const t of [0, 1 / 30, 2 / 30, 0.3]) {
    examine.mockClear();
    await refiner.refine(frame, [det], t);
    counts.push(examine.mock.calls.length);
  }
  expect(counts).toEqual([4, 1, 1, 4]);
  config.empty_search_period_s = 0;   // off
  examine.mockClear();
  await refiner.refine(frame, [det], 0.31);
  expect(examine.mock.calls.length).toBe(4);
});

it('runs refinement crops at refine.crop_size only on a model that can', async () => {
  const config = { ...defaultConfig().refine, crop_size: 224 };
  expect(new FaceRefiner({ detect: async () => [], size: 320 }, config).size).toBe(320);
  const sizes: (number | undefined)[] = [];
  const dynamic = { size: 320, runsAt: () => true, detect: async (_f: unknown, size?: number) => { sizes.push(size); return []; } };
  const refiner = new FaceRefiner(dynamic, config);
  expect(refiner.size).toBe(224);
  await refiner.refine(frames.sharp.frame, [frames.sharp.det], 0, true);
  expect(sizes.length).toBeGreaterThan(0);
  expect(sizes.every((s) => s === 224)).toBe(true);
  // a dynamic-shape ONNX model reports it through the runner
  const detector = new YoloPoseDetector({ inputSize: 320, dynamic: true, run: async () => [] }, { kpt: [6, 3] });
  expect(detector.runsAt(224)).toBe(true);
  expect(new YoloPoseDetector({ inputSize: 320, run: async () => [] }, { kpt: [6, 3] }).runsAt(224)).toBe(false);
});

/** A canonical anyface5 face centred on a size x size crop (eyes level). */
function centredFace(size: number, score: number, rollDeg?: number): Detection {
  const u = size * 0.35 / 2.1;   // the face long side is share * size; eye distance ~ half of it
  const c = size / 2;
  return { box: [c - 1.05 * u, c - 0.8 * u, c + 1.05 * u, c + 1.3 * u], score, schema: getSchema('anyface5'),
    keypoints: [[c - 0.5 * u, c], [c + 0.5 * u, c], [c, c + 0.5 * u], [c - 0.46 * u, c + 0.63 * u], [c + 0.46 * u, c + 0.63 * u]],
    kpScores: [0.95, 0.95, 0.95, 0.9, 0.9], extra: rollDeg === undefined ? {} : { roll_deg: rollDeg } };
}

it('tracks with a crop landmarker: no four-way search, the detector only for the precise pass', async () => {
  const { frame, det } = frames.sharp;
  let roll = 0, lmCalls = 0, detCalls = 0;
  const landmarker = { size: 128, share: 0.35, keypointMaxRoll: 55, rotationInvariant: true,
    detect: async () => { lmCalls++; return [centredFace(128, 0.95, roll)]; } };
  const detector = { size: 320, detect: async () => { detCalls++; return [centredFace(320, 0.8)]; } };
  const refiner = new FaceRefiner(detector, defaultConfig().refine, landmarker);
  const [tracked] = await refiner.refine(frame, [det], 0, true);
  expect(tracked.extra?.refined).toBe(true);
  expect([lmCalls, detCalls]).toEqual([1, 0]);            // upright: one tracker run, no search
  const sharp = await refiner.precise(frame, tracked);
  expect(detCalls).toBe(1);
  expect(sharp?.extra?.tracker_score).toBe(0.95);
  expect(sharp?.source).toBe('detector+refined');   // the precise model's keypoints
  roll = 120;                                             // beyond the keypoint range: the roll head turns the crop
  lmCalls = 0;
  const [turned] = await refiner.refine(frame, [det], 0, true);
  expect(lmCalls).toBe(2);
  expect(Math.abs(Number(turned.extra?.crop_angle) - (-120))).toBeLessThan(1e-6);
});

function fakeRunner(size: number, output: Float32Array, dims: number[]): Runner {
  return { inputSize: size, run: async () => [{ data: output, dims }] };
}

describe('decoders', () => {
  const frame: Frame = { width: 640, height: 480, data: new Uint8ClampedArray(640 * 480 * 4) }; // ratio .5, pad y 40

  it('decodes Ultralytics YOLO-pose channel-major output', async () => {
    const k = 6, n = 10, channels = 4 + 1 + 3 * k;
    const out = new Float32Array(channels * n);
    const set = (c: number, i: number, v: number) => { out[c * n + i] = v; };
    [160, 160, 100, 80, 0.9].forEach((v, c) => set(c, 3, v));
    const kpts = [[140, 150], [180, 150], [160, 175], [160, 195], [125, 125], [195, 125]];
    kpts.forEach(([x, y], j) => { set(5 + 3 * j, 3, x); set(6 + 3 * j, 3, y); set(7 + 3 * j, 3, 0.8); });
    [162, 160, 98, 80, 0.8].forEach((v, c) => set(c, 5, v));
    const det = new YoloPoseDetector(fakeRunner(320, out, [1, channels, n]), { kpt: [6, 3] });
    const found = await det.detect(frame);
    expect(found).toHaveLength(1);
    expect(found[0].schema.name).toBe('dogface6');
    found[0].box.forEach((v, i) => expect(v).toBeCloseTo([220, 160, 420, 320][i], 3));
    expect(found[0].keypoints[0][0]).toBeCloseTo(280, 3);
    expect(found[0].keypoints[0][1]).toBeCloseTo((150 - 40) / 0.5, 3);
  });

  it('decodes YOLOX and keeps only the winning dog class', async () => {
    const width = 85, n = 3549, out = new Float32Array(n * width);
    const cell = 12 * 52 + 10;
    out.set([0.5, 0.5, Math.log(100 / 8), Math.log(60 / 8), 0.9], cell * width);
    out[cell * width + 5 + 16] = 0.95;
    const other = 13 * 52 + 30;
    out.set([0.5, 0.5, Math.log(10), Math.log(10), 0.9], other * width);
    out[other * width + 5 + 16] = 0.5; out[other * width + 5] = 0.9;
    const gate = new YoloxBodyGate(fakeRunner(416, out, [1, n, width]));
    const { boxes, scores } = await gate.detect({ width: 832, height: 416, data: new Uint8ClampedArray(832 * 416 * 4) });
    expect(boxes).toHaveLength(1);
    expect(scores[0]).toBeCloseTo(0.855, 5);
    const cx = 10.5 * 8 / 0.5, cy = 12.5 * 8 / 0.5;
    boxes[0].forEach((v, i) => expect(v).toBeCloseTo([cx - 100, cy - 60, cx + 100, cy + 60][i], 3));
  });
});

describe('protocol client', () => {
  it('encodes the original frames and builds a manifest the server can verify', async () => {
    const engine = new CaptureEngine({ pose: { views: [{ name: 'front', center: 0, lower: -0.13, upper: 0.13, shots: 2 }] },
      selection: { mode: 'best', min_gap_s: 0.1 } });
    await run(engine, Array(20).fill('sharp'));
    const bundle = engine.finalize()!;
    const config = { session_id: 'cap_x', upload: { jpeg_quality: 0.95, max_long_side: 4096, max_bytes_per_image: 1 << 20,
      max_shots: 8, image_format: 'jpeg' } } as unknown as SessionConfig;
    const shots = await encodeBundle(bundle, config, async (f) => new Uint8Array(f.data.slice(0, 1000)));
    expect(shots.map((s) => s.id)).toEqual(['front-1', 'front-2']);
    const manifest = await buildManifest(config, 'n0nce', shots, { signals: { single_track: true } });
    expect(manifest.protocol).toBe('dog-capture/1');
    expect(manifest.shots[0].sha256).toBe(await sha256Hex(shots[0].jpeg));
    expect(manifest.shots[0].detection!.box).toHaveLength(4);
    expect(manifest.shots[0].file).toBe('front-1.jpg');
  });
});
