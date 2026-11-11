/**
 * Web Worker host for the capture engine: inference and quality never block
 * the UI thread.  Messages (main -> worker):
 *   {type:'init', config, models:{face, gate}, wasmPaths?, providers?}  -> {type:'ready', providers}
 *   {type:'frame', bitmap: ImageBitmap, t, index}                    -> {type:'result', result}
 *   {type:'finish'}                                                 -> {type:'shots', shots, complete}
 * The main thread sends the next frame only after a result arrives (latest-frame semantics).
 */
import { CaptureEngine, type FrameResult } from './engine';
import { alignFace } from './alignment';
import type { Frame } from './image';
import { createEngine, encodeJpegBrowser, type SessionConfig } from './session';

let engine: CaptureEngine | null = null;
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
  try {
    if (data.type === 'init') {
      const ort = await import('onnxruntime-web');
      if (data.wasmPaths) ort.env.wasm.wasmPaths = data.wasmPaths;
      ort.env.wasm.numThreads = data.threads ?? Math.min(4, (self.navigator?.hardwareConcurrency ?? 2));
      const created = await createEngine(data.config as SessionConfig, data.models, ort, data.providers);
      engine = created.engine;
      postMessage({ type: 'ready', providers: created.providers });
    } else if (data.type === 'frame') {
      if (!engine) throw new Error('engine not initialised');
      const result = await engine.process(frameFrom(data.bitmap), data.t, data.index);
      postMessage({ type: 'result', result: result ? serialiseResult(result) : null, complete: engine.complete() });
    } else if (data.type === 'finish') {
      if (!engine) throw new Error('engine not initialised');
      const bundle = engine.finalize();
      const shots = [];
      if (bundle) {
        for (const view of bundle.viewNames) {
          for (const [rank, shot] of bundle.slots[view].entries()) {
            if (!shot.frame || !shot.geometry) continue;
            const thumb = alignFace(shot.frame, shot.geometry.points, 224).image;
            shots.push({
              id: `${view}-${rank + 1}`, view, rank: rank + 1, timeS: shot.timeS, frameIndex: shot.frameIndex,
              trackId: shot.trackId, width: shot.frame.width, height: shot.frame.height,
              jpeg: await encodeJpegBrowser(shot.frame, data.jpegQuality ?? 0.95),
              thumbnail: await encodeJpegBrowser(thumb, 0.9),
              detection: shot.detection, geometry: { yaw: shot.geometry.yaw, pitch: shot.geometry.pitch,
                roll_deg: shot.geometry.roll_deg, iod: shot.geometry.iod, approx_deg: shot.geometry.approx_deg },
              quality: shot.quality ? { score: shot.quality.score, level: shot.quality.level, factors: shot.quality.factors } : null,
            });
          }
        }
      }
      postMessage({ type: 'shots', shots, complete: engine.complete(),
        views: bundle ? bundle.viewNames : [], captured: bundle ? bundle.captured() : [] });
    } else if (data.type === 'reset') {
      engine = null;
    }
  } catch (error) {
    postMessage({ type: 'error', message: error instanceof Error ? error.message : String(error) });
  }
}

if (typeof self !== 'undefined' && typeof (self as unknown as { importScripts?: unknown }).importScripts === 'function') {
  self.onmessage = handle;
}
