/** dog-capture/1 protocol client: session config, models, signals, package upload. */
import type { CaptureConfig } from './config';
import { createOrtRunner, YoloPoseDetector, YoloxBodyGate } from './detectors';
import { CaptureEngine, type FrameResult } from './engine';
import type { Frame } from './image';
import { detectionToJSON } from './schema';
import type { Shot, TrackShots } from './selection';

export const PROTOCOL = 'dog-capture/1';
export const SDK_VERSION = 'dog-capture-sdk/1.0.0';

export interface ModelInfo { url: string; sha256: string; bytes?: number; kind?: string; schema?: string; input_size?: number; kpt_shape?: number[]; yaw_scale?: number }
export interface SessionConfig {
  protocol: string; session_id: string; purpose: 'enroll' | 'identify';
  policy: { name: string; views: CaptureConfig['pose']['views']; view_order: string[]; max_duration_s: number;
    max_attempts?: number; selection?: object };
  engine: CaptureConfig; models: { face?: ModelInfo; gate?: ModelInfo };
  upload: { image_format: string; jpeg_quality: number; max_long_side: number; max_bytes_per_image: number; max_shots: number };
  expires_at: number; client_commit: boolean; state: string;
}

export interface VerificationResult {
  protocol: string; session_id: string; status: 'accepted' | 'retake';
  shots: { id: string; view: string; accepted: boolean; reasons: string[]; server?: Record<string, unknown> }[];
  missing_views: string[]; retake_views: string[]; recognition?: Record<string, unknown>;
  capture_token?: string; token_expires_at?: number; elapsed_ms?: number;
}

export class ProtocolClientError extends Error {
  constructor(public code: string, message: string, public status = 0) { super(message); }
}

async function json(response: Response) {
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = (body as { detail?: { code?: string; message?: string } }).detail ?? body;
    throw new ProtocolClientError((detail as { code?: string }).code ?? 'http_error',
      (detail as { message?: string }).message ?? `HTTP ${response.status}`, response.status);
  }
  return body;
}

export async function fetchSessionConfig(apiBase: string, sessionId: string, nonce: string): Promise<SessionConfig> {
  const url = `${apiBase}/v1/capture/sessions/${encodeURIComponent(sessionId)}/config?nonce=${encodeURIComponent(nonce)}`;
  const config = (await json(await fetch(url))) as SessionConfig;
  if (config.protocol !== PROTOCOL) throw new ProtocolClientError('protocol_mismatch', `server speaks ${config.protocol}`);
  return config;
}

export async function sha256Hex(data: ArrayBuffer | Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', data as BufferSource);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

/** Download a model and verify its SHA-256 before it is ever executed. */
export async function fetchModel(apiBase: string, info: ModelInfo): Promise<ArrayBuffer> {
  const response = await fetch(info.url.startsWith('http') ? info.url : apiBase + info.url);
  if (!response.ok) throw new ProtocolClientError('model_download_failed', `${info.url}: HTTP ${response.status}`);
  const buffer = await response.arrayBuffer();
  if ((await sha256Hex(buffer)) !== info.sha256) throw new ProtocolClientError('model_hash_mismatch', info.url);
  return buffer;
}

/** Engine wired to onnxruntime-web sessions for the session's models. */
export async function createEngine(config: SessionConfig, models: { face: ArrayBuffer; gate?: ArrayBuffer | null },
  ort: Parameters<typeof createOrtRunner>[0], providers: string[] = ['webgpu', 'wasm']) {
  const faceInfo = config.models.face!;
  const faceRunner = await createOrtRunner(ort, models.face, providers);
  if (faceInfo.input_size) faceRunner.inputSize = faceInfo.input_size;
  const kpt = (faceInfo.kpt_shape ?? [6, 3]) as [number, number];
  const detector = new YoloPoseDetector(faceRunner, { kpt, schema: faceInfo.schema,
    scoreThreshold: config.engine.detector.score_threshold, nmsIou: config.engine.detector.nms_iou,
    maxFaces: config.engine.detector.max_faces,
    yawScale: typeof faceInfo.yaw_scale === 'number' ? faceInfo.yaw_scale : undefined });
  let gate = null, gateProviders: string[] = [];
  if (models.gate && config.models.gate) {
    const gateRunner = await createOrtRunner(ort, models.gate, providers);
    gate = new YoloxBodyGate(gateRunner, { classId: config.engine.body_gate.class_id,
      scoreThreshold: config.engine.body_gate.score_threshold });
    gateProviders = gateRunner.providers;
  }
  const engine = new CaptureEngine(config.engine, detector, gate);
  return { engine, providers: { face: faceRunner.providers, gate: gateProviders } };
}

/** Reserved liveness / anti-replay evidence (recorded, not judged, in protocol v1). */
export class SignalRecorder {
  private trackIds = new Set<number>();
  private visits: [string, number][] = [];
  private yaw: [number, number][] = [];
  private guidanceCounts: Record<string, number> = {};
  private lastYawT = -Infinity;
  private lastVisit: string | null = null;
  frames = 0;
  started = Date.now();

  add(result: FrameResult) {
    this.frames += 1;
    this.guidanceCounts[result.guidance.code] = (this.guidanceCounts[result.guidance.code] ?? 0) + 1;
    const face = result.faces.find((f) => f.trackId === result.primaryTrack);
    if (!face) return;
    this.trackIds.add(face.trackId);
    if (face.geometry.valid && result.timeS - this.lastYawT >= 0.1) {
      this.yaw.push([Math.round(result.timeS * 1000) / 1000, Math.round(face.geometry.yaw * 1e4) / 1e4]);
      this.lastYawT = result.timeS;
      if (this.yaw.length > 600) this.yaw.splice(0, this.yaw.length - 600);
    }
    const view = face.viewState.stable ? face.viewState.view : null;
    if (view && view !== this.lastVisit) { this.visits.push([view, Math.round(result.timeS * 1000) / 1000]); this.lastVisit = view; }
  }

  toJSON(order: string[]) {
    const seen = this.visits.map(([v]) => v).filter((v, i, all) => all.indexOf(v) === i);
    const expected = order.filter((v) => seen.includes(v));
    return { track_ids: [...this.trackIds], single_track: this.trackIds.size <= 1, view_sequence: this.visits,
      yaw_trace: this.yaw, guidance_counts: this.guidanceCounts,
      prompt_order_followed: seen.length > 0 && seen.join() === expected.join() };
  }
}

export interface EncodedShot { shot: Shot; id: string; rank: number; jpeg: Uint8Array; width: number; height: number }

/** Encode the original full frames of a bundle (the recognizer's input), respecting upload limits. */
export async function encodeBundle(bundle: TrackShots, config: SessionConfig,
  encodeJpeg: (frame: Frame, quality: number) => Promise<Uint8Array>): Promise<EncodedShot[]> {
  const out: EncodedShot[] = [];
  for (const view of bundle.viewNames) {
    for (const [rank, shot] of bundle.slots[view].entries()) {
      if (!shot.frame) continue;
      if (Math.max(shot.frame.width, shot.frame.height) > config.upload.max_long_side) continue;
      const jpeg = await encodeJpeg(shot.frame, config.upload.jpeg_quality);
      if (jpeg.byteLength > config.upload.max_bytes_per_image) continue;
      out.push({ shot, id: `${view}-${rank + 1}`, rank: rank + 1, jpeg, width: shot.frame.width, height: shot.frame.height });
      if (out.length >= config.upload.max_shots) return out;
    }
  }
  return out;
}

export async function buildManifest(config: SessionConfig, nonce: string, shots: EncodedShot[], extra: {
  signals?: object; client?: object; timing?: object; models_used?: object } = {}) {
  const entries = [];
  for (const s of shots) {
    entries.push({ id: s.id, view: s.shot.view, rank: s.rank, file: `${s.id}.jpg`, sha256: await sha256Hex(s.jpeg),
      width: s.width, height: s.height, time_s: s.shot.timeS, frame_index: s.shot.frameIndex, track_id: s.shot.trackId,
      detection: s.shot.detection ? detectionToJSON(s.shot.detection) : null,
      pose: s.shot.geometry ? { yaw: s.shot.geometry.yaw, pitch: s.shot.geometry.pitch, roll_deg: s.shot.geometry.roll_deg,
        iod: s.shot.geometry.iod } : null,
      quality: s.shot.quality ? { score: s.shot.quality.score, level: s.shot.quality.level, factors: s.shot.quality.factors } : null });
  }
  return { protocol: PROTOCOL, session_id: config.session_id, nonce, client: { sdk: SDK_VERSION, ...extra.client },
    models_used: extra.models_used ?? {}, timing: extra.timing ?? {}, shots: entries, signals: extra.signals ?? {} };
}

export async function submitPackage(apiBase: string, sessionId: string, manifest: object, shots: EncodedShot[]):
  Promise<VerificationResult> {
  const form = new FormData();
  form.append('manifest', JSON.stringify(manifest));
  for (const s of shots) form.append('files', new Blob([s.jpeg as BlobPart], { type: 'image/jpeg' }), `${s.id}.jpg`);
  const url = `${apiBase}/v1/capture/sessions/${encodeURIComponent(sessionId)}/package`;
  return (await json(await fetch(url, { method: 'POST', body: form }))) as VerificationResult;
}

/** Dev / LAN mode only (client_commit): normally the business backend commits with its API key. */
export async function commitSession(apiBase: string, sessionId: string, token: string, body: object = {}) {
  const url = `${apiBase}/v1/capture/sessions/${encodeURIComponent(sessionId)}/commit`;
  return json(await fetch(url, { method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ capture_token: token, ...body }) }));
}

/** Browser JPEG encoder (OffscreenCanvas); works in Web Workers. */
export async function encodeJpegBrowser(frame: Frame, quality: number): Promise<Uint8Array> {
  const canvas = new OffscreenCanvas(frame.width, frame.height);
  const context = canvas.getContext('2d')!;
  context.putImageData(new ImageData(new Uint8ClampedArray(frame.data), frame.width, frame.height), 0, 0);
  const blob = await canvas.convertToBlob({ type: 'image/jpeg', quality });
  return new Uint8Array(await blob.arrayBuffer());
}
