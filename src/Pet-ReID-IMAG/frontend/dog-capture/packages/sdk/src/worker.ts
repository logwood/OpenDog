/**
 * Web Worker host for the capture engine: inference and quality never block
 * the UI thread.  Messages (main -> worker):
 *   {type:'init', config, models:{face, gate}, wasmPaths?, backend?}    -> {type:'ready', providers}
 *     backend 'wasm' | 'webgpu' | 'auto' (default): auto uses WebGPU only with a hardware adapter and
 *     only if it measures faster than WASM on this device (software adapters such as SwiftShader
 *     are 20-30x slower than WASM and are skipped).
 *   {type:'frame', bitmap: ImageBitmap, t, index}                    -> {type:'result', result}
 *   {type:'finish'}                                                 -> {type:'shots', shots, complete}
 * The main thread sends the next frame only after a result arrives (latest-frame semantics).
 */
import { CaptureEngine, type FrameResult } from './engine';
import { alignFace } from './alignment';
import type { Frame } from './image';
import { detectionToJSON } from './schema';
import { createOrtRunner } from './detectors';
import { createEngine, encodeJpegBrowser, sha256Hex, SignalRecorder, type SessionConfig } from './session';

interface GpuInfo { vendor?: string; architecture?: string; description?: string }

/** A hardware WebGPU adapter, or null (none, or a software rasteriser). */
async function hardwareAdapter(): Promise<GpuInfo | null> {
  const gpu = (self.navigator as unknown as { gpu?: { requestAdapter(): Promise<any> } }).gpu;
  if (!gpu) return null;
  const adapter = await gpu.requestAdapter().catch(() => null);
  if (!adapter || adapter.isFallbackAdapter) return null;
  const info: GpuInfo = adapter.info ?? {};
  return /swiftshader|llvmpipe|lavapipe|software|microsoft basic/i.test(`${info.vendor} ${info.architecture} ${info.description}`)
    ? null : info;
}

/** Time the face model on each backend (after a warm-up that compiles the GPU shaders); fastest first. */
async function rankBackends(ort: Parameters<typeof createOrtRunner>[0], face: ArrayBuffer, size: number) {
  const timing: Record<string, number> = {};
  for (const ep of ['webgpu', 'wasm']) {
    try {
      const runner = await createOrtRunner(ort, new Uint8Array(face), [ep]);
      const input = new Float32Array(3 * size * size), dims = [1, 3, size, size];
      await runner.run(input, dims); await runner.run(input, dims);
      const started = performance.now();
      for (let i = 0; i < 3; i++) await runner.run(input, dims);
      timing[ep] = (performance.now() - started) / 3;
      await runner.release();
    } catch { /* backend unavailable on this device */ }
  }
  const order = Object.entries(timing).sort((a, b) => a[1] - b[1]).map(([ep]) => ep);
  return { order: order.length ? order : ['wasm'], timing };
}

let engine: CaptureEngine | null = null;
let recorder = new SignalRecorder();
let sessionConfig: SessionConfig | null = null;
let providers: unknown = null;
let canvas: OffscreenCanvas | null = null;

export function serialiseResult(r: FrameResult) {
  return {
    frameIndex: r.frameIndex, timeS: r.timeS, rotationDeg: r.rotationDeg, primaryTrack: r.primaryTrack,
    guidance: r.guidance, capturedViews: r.capturedViews, missingViews: r.missingViews, events: r.events,
    rejected: r.rejected, timingsMs: r.timingsMs,
    faces: r.faces.map((f) => ({
      trackId: f.trackId, box: f.detection.box, keypoints: f.detection.keypoints, score: f.detection.score,
      view: f.viewState.view, raw: f.viewState.raw, stable: f.viewState.stable, yaw: f.geometry.yaw,
      approxDeg: f.geometry.approx_deg, valid: f.geometry.valid,
      quality: { score: f.quality.score, level: f.quality.level, passed: f.quality.passed, reasons: f.quality.reasons },
      accepted: f.accepted,
    })),
  };
}

function frameFrom(bitmap: ImageBitmap): Frame {
  if (!canvas || canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
    canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  }
  const context = canvas.getContext('2d', { willReadFrequently: true })!;
  context.drawImage(bitmap, 0, 0);
  bitmap.close();
  const image = context.getImageData(0, 0, canvas.width, canvas.height);
  return { width: image.width, height: image.height, data: image.data };
}

async function handle(message: MessageEvent) {
  const data = message.data;
  const reply = (value: object) => postMessage({ ...value, requestId: data.requestId });
  try {
    if (data.type === 'init') {
      const backend: string = data.backend ?? (data.webgpu ? 'webgpu' : 'auto');
      const adapter = backend === 'wasm' ? null : await hardwareAdapter();
      const gpu = backend === 'webgpu' || adapter !== null;
      // The WASM-only build is the smaller download; the WebGPU build (asyncify runtime) only when it can help.
      const ort = gpu ? await import('onnxruntime-web/webgpu') : await import('onnxruntime-web/wasm');
      if (data.wasmPaths) ort.env.wasm.wasmPaths = data.wasmPaths;
      ort.env.wasm.numThreads = data.threads ?? Math.min(4, (self.navigator?.hardwareConcurrency ?? 2));
      let chosen: string[] = data.providers ?? ['wasm'], ranking = null;
      if (gpu && !data.providers) {
        const size = (data.config as SessionConfig).models.face?.input_size ?? 320;
        ranking = backend === 'webgpu' ? { order: ['webgpu', 'wasm'], timing: {} } : await rankBackends(ort, data.models.face, size);
        chosen = ranking.order[0] === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'];
      }
      const created = await createEngine(data.config as SessionConfig, data.models, ort, chosen);
      engine = created.engine;
      sessionConfig = data.config as SessionConfig;
      recorder = new SignalRecorder();
      providers = { ...created.providers, backend: created.providers.face[0] ?? 'wasm', adapter, timing: ranking?.timing ?? null };
      reply({ type: 'ready', providers });
    } else if (data.type === 'frame') {
      if (!engine) throw new Error('engine not initialised');
      const result = await engine.process(frameFrom(data.bitmap), data.t, data.index);
      if (result) recorder.add(result);
      const primary = engine.primaryId !== null ? engine.contexts.get(engine.primaryId) : undefined;
      const counts = primary ? Object.fromEntries(engine.viewNames.map((v) => [v, primary.shots.slots[v].length])) : null;
      reply({ type: 'result', result: result ? serialiseResult(result) : null, complete: engine.complete(), counts });
    } else if (data.type === 'retake') {
      engine?.retake(data.views);
      reply({ type: 'retaken', views: data.views });
    } else if (data.type === 'finish') {
      if (!engine) throw new Error('engine not initialised');
      const bundle = engine.finalize();
      const shots = [];
      if (bundle) {
        for (const view of bundle.viewNames) {
          for (const [rank, shot] of bundle.slots[view].entries()) {
            if (!shot.frame || !shot.geometry) continue;
            const thumb = alignFace(shot.frame, shot.geometry.points, 224).image;
            const id = `${view}-${rank + 1}`;
            const jpeg = await encodeJpegBrowser(shot.frame, sessionConfig?.upload.jpeg_quality ?? 0.95);
            shots.push({
              id, view, rank: rank + 1, jpeg, thumbnail: await encodeJpegBrowser(thumb, 0.9),
              level: shot.quality?.level ?? 'low', score: shot.quality?.score ?? 0,
              // manifest entry (dog-capture/1 capture-package shot): the upload is the ORIGINAL frame
              entry: { id, view, rank: rank + 1, file: `${id}.jpg`, sha256: await sha256Hex(jpeg),
                width: shot.frame.width, height: shot.frame.height, time_s: shot.timeS, frame_index: shot.frameIndex,
                track_id: shot.trackId, detection: shot.detection ? detectionToJSON(shot.detection) : null,
                pose: { yaw: shot.geometry.yaw, pitch: shot.geometry.pitch, roll_deg: shot.geometry.roll_deg,
                  iod: shot.geometry.iod, approx_deg: shot.geometry.approx_deg },
                quality: shot.quality ? { score: shot.quality.score, level: shot.quality.level, factors: shot.quality.factors } : null },
            });
          }
        }
      }
      reply({ type: 'shots', shots, complete: engine.complete(), providers,
        signals: recorder.toJSON(sessionConfig?.policy.view_order ?? []), frames: recorder.frames,
        views: bundle ? bundle.viewNames : [], captured: bundle ? bundle.captured() : [] });
    } else if (data.type === 'reset') {
      engine = null;
    }
  } catch (error) {
    reply({ type: 'error', message: error instanceof Error ? error.message : String(error) });
  }
}

if (typeof self !== 'undefined' && typeof (self as unknown as { importScripts?: unknown }).importScripts === 'function') {
  self.onmessage = handle;
}
