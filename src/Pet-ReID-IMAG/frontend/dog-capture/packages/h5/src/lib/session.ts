/**
 * Capture-session controller: config + model download (hash-checked), the
 * inference worker loop with latest-frame semantics, retakes, package upload.
 * UI components only observe `state` and call the methods.
 */
import { reactive, shallowRef } from 'vue';
import { WorkerRPC } from './worker-rpc';
import {
  fetchModel, fetchSessionConfig, PROTOCOL, SDK_VERSION, commitSession,
  type Guidance, type SessionConfig, type VerificationResult,
} from '@opendog/dog-capture-sdk';

export interface FaceView {
  trackId: number; box: number[]; keypoints: number[][]; view: string | null; stable: boolean;
  quality: { score: number; level: string; passed: boolean; reasons: string[] }; approxDeg: number;
}
export interface LiveResult {
  frameIndex: number; timeS: number; primaryTrack: number | null; guidance: Guidance; capturedViews: string[];
  missingViews: string[]; faces: FaceView[]; events: { event: string; view: string }[]; timingsMs: Record<string, number>;
}
export interface CapturedShot {
  id: string; view: string; rank: number; jpeg: Uint8Array; thumbnail: Uint8Array; level: string; score: number;
  entry: Record<string, unknown>; thumbUrl?: string;
}

export function createCaptureSession(options: { apiBase: string; sessionId: string; nonce: string }) {
  const state = reactive({
    phase: 'idle' as 'idle' | 'loading' | 'ready' | 'capturing' | 'finishing' | 'review' | 'uploading' | 'done' | 'error',
    progress: 0, message: '', error: '', providers: null as unknown,
    live: null as LiveResult | null, complete: false, fps: 0, frames: 0, startedAt: 0,
    counts: {} as Record<string, number>,
  });
  const config = shallowRef<SessionConfig | null>(null);
  const shots = shallowRef<CapturedShot[]>([]);
  const result = shallowRef<VerificationResult | null>(null);
  let worker: Worker | null = null;
  let rpc: WorkerRPC | null = null;
  let busy = false, running = false, index = 0, t0 = 0, lastFps = 0, frameCount = 0;
  let signals: unknown = null, framesProcessed = 0;
  let video: HTMLVideoElement | null = null;
  const listeners = new Set<(r: LiveResult) => void>();

  function call<T>(message: object, transfer: Transferable[] = [], expect?: string, timeoutMs?: number): Promise<T> {
    if (!rpc) return Promise.reject(new Error('Capture worker is not ready'));
    return rpc.call<T>(message, transfer, expect, timeoutMs);
  }

  async function load() {
    state.phase = 'loading';
    state.message = '正在获取采集配置…';
    try {
      const cfg = await fetchSessionConfig(options.apiBase, options.sessionId, options.nonce);
      if (!cfg.models.face) throw new Error('Server has no capture face model configured');
      config.value = cfg;
      state.counts = Object.fromEntries(cfg.policy.views.map((v) => [v.name, 0]));
      state.message = '正在下载识别模型（首次约 15 MB，之后会缓存）…';
      const face = await fetchModel(options.apiBase, cfg.models.face!);
      state.progress = 0.6;
      const gate = cfg.models.gate ? await fetchModel(options.apiBase, cfg.models.gate) : null;
      state.progress = 0.8;
      state.message = '正在初始化推理引擎…';
      worker = new Worker(new URL('../../../sdk/src/worker.ts', import.meta.url), { type: 'module' });
      rpc = new WorkerRPC(worker);
      const ready = await call<{ providers: unknown }>({ type: 'init', config: cfg, models: { face, gate },
        wasmPaths: `${import.meta.env.BASE_URL}ort/`, providers: ['wasm'] },
      [face, ...(gate ? [gate] : [])], 'ready', 120_000);
      state.providers = ready.providers;
      state.progress = 1;
      state.phase = 'ready';
    } catch (error) {
      state.phase = 'error';
      state.error = error instanceof Error ? error.message : String(error);
    }
  }

  async function pump() {
    if (!running || !video || !worker) return;
    if (!busy && video.readyState >= 2 && video.videoWidth) {
      busy = true;
      try {
        const bitmap = await createImageBitmap(video);
        const t = performance.now() / 1000 - t0;
        const reply = await call<{ result: LiveResult | null; complete: boolean; counts: Record<string, number> | null }>(
          { type: 'frame', bitmap, t, index: index++ }, [bitmap], 'result');
        if (reply.result) {
          state.live = reply.result;
          frameCount += 1; state.frames += 1;
          if (reply.result.events.some((e) => e.event === 'view_captured')) navigator.vibrate?.(40);
          listeners.forEach((fn) => fn(reply.result!));
        }
        if (reply.counts) state.counts = reply.counts;
        state.complete = reply.complete;
        const now = performance.now();
        if (now - lastFps > 1000) { state.fps = (frameCount * 1000) / (now - lastFps); frameCount = 0; lastFps = now; }
      } catch (error) {
        running = false;
        state.phase = 'error';
        state.error = error instanceof Error ? error.message : String(error);
      } finally {
        busy = false;
      }
    }
    if (running) requestAnimationFrame(() => void pump());
  }

  function start(el: HTMLVideoElement) {
    if (running) return;
    video = el;
    running = true;
    if (!t0) { t0 = performance.now() / 1000; state.startedAt = Date.now(); lastFps = performance.now(); }
    state.startedAt = Date.now();
    state.phase = 'capturing';
    void pump();
  }

  function pause() { running = false; }

  async function finish() {
    running = false;
    while (busy) await new Promise((r) => setTimeout(r, 20));
    state.phase = 'finishing';
    const reply = await call<{ shots: CapturedShot[]; signals: unknown; frames: number }>({ type: 'finish' }, [], 'shots');
    for (const s of shots.value) if (s.thumbUrl) URL.revokeObjectURL(s.thumbUrl);
    shots.value = reply.shots.map((s) => ({ ...s, thumbUrl: URL.createObjectURL(new Blob([s.thumbnail as BlobPart], { type: 'image/jpeg' })) }));
    signals = reply.signals;
    framesProcessed = reply.frames;
    state.phase = 'review';
  }

  async function retake(views: string[]) {
    await call({ type: 'retake', views }, [], 'retaken');
    for (const v of views) state.counts[v] = 0;
    state.complete = false;
    // CameraStage remounts and starts with a fresh video element after the retake is ready.
  }

  async function upload() {
    const cfg = config.value!;
    state.phase = 'uploading';
    const manifest = {
      protocol: PROTOCOL, session_id: cfg.session_id, nonce: options.nonce,
      client: { sdk: SDK_VERSION, user_agent: navigator.userAgent, platform: navigator.platform,
        camera: video ? { width: video.videoWidth, height: video.videoHeight, facing: 'environment' } : null },
      models_used: { face_sha256: cfg.models.face?.sha256, gate_sha256: cfg.models.gate?.sha256, providers: state.providers },
      timing: { started_at_ms: state.startedAt, finished_at_ms: Date.now(), frames_processed: framesProcessed, fps: state.fps },
      shots: shots.value.map((s) => s.entry), signals,
    };
    const form = new FormData();
    form.append('manifest', JSON.stringify(manifest));
    for (const s of shots.value) form.append('files', new Blob([s.jpeg as BlobPart], { type: 'image/jpeg' }), `${s.id}.jpg`);
    try {
      const response = await fetch(`${options.apiBase}/v1/capture/sessions/${encodeURIComponent(cfg.session_id)}/package`,
        { method: 'POST', body: form });
      const body = await response.json();
      if (!response.ok) throw new Error(body.message ?? body.detail?.message ?? `HTTP ${response.status}`);
      result.value = body as VerificationResult;
      state.phase = 'done';
    } catch (error) {
      state.phase = 'error';
      state.error = error instanceof Error ? error.message : String(error);
    }
  }

  /** Dev / LAN mode only: commit from the page (production backends commit with their API key). */
  async function commit() {
    const token = result.value?.capture_token;
    if (!token) throw new Error('no capture token: the business backend commits this session');
    return commitSession(options.apiBase, options.sessionId, token);
  }

  function onResult(fn: (r: LiveResult) => void) { listeners.add(fn); return () => listeners.delete(fn); }

  function dispose() {
    running = false;
    rpc?.close();
    rpc = null;
    worker = null;
    for (const s of shots.value) if (s.thumbUrl) URL.revokeObjectURL(s.thumbUrl);
  }

  return { state, config, shots, result, load, start, pause, finish, retake, upload, commit, onResult, dispose };
}

export type CaptureSession = ReturnType<typeof createCaptureSession>;
