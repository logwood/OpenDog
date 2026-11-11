/**
 * CaptureEngine (video mode) — port of dog_capture/engine/engine.py:
 * detection -> ROI zoom fallback -> ByteTrack -> per-track species gate ->
 * pose -> debounced view -> aligned-crop quality -> best shot -> guidance.
 */
import { alignFace } from './alignment';
import { defaultConfig, mergeConfig, type CaptureConfig, type ViewSpec } from './config';
import type { BodyGate, FaceDetector } from './detectors';
import { guidance, type Guidance } from './guidance';
import { crop, mirror, rotate, type Frame } from './image';
import { faceGeometry, ViewClassifier, type FaceGeometry, type ViewState } from './pose';
import { QualityAssessor, type QualityReport } from './quality';
import { detectionReason, height, iou, width, type Box, type Detection } from './schema';
import { TrackShots, type CaptureEvent, type Shot } from './selection';
import { ByteTracker, type Track } from './tracking';

const ROTATION_SEARCH = [0, 90, 270, 180];

export class OrientationLock {
  fixed: number | null;
  rotation: number;
  locked: boolean;
  private frame = 0;
  private probeIndex = 0;
  private start: number | null = null;
  private lastFace: number | null = null;
  private hits = 0;
  private sideways = 0;
  private probeHits = new Map<number, number>();

  constructor(mode: string, private probeEvery = 10, private probeAfter = 1.0, private relockAfter = 3.0) {
    this.fixed = mode === 'auto' ? null : Number(mode);
    this.rotation = this.fixed ?? 0;
    this.locked = this.fixed !== null;
  }

  next(t: number): [number, boolean] {
    this.frame += 1;
    if (this.fixed !== null) return [this.rotation, false];
    if (this.start === null) this.start = t;
    const since = t - (this.lastFace ?? this.start);
    if (this.locked && since <= this.relockAfter) return [this.rotation, false];
    if (since >= this.probeAfter && this.frame % this.probeEvery === 0) {
      const others = ROTATION_SEARCH.filter((r) => r !== this.rotation);
      return [others[this.probeIndex++ % others.length], true];
    }
    return [this.rotation, false];
  }

  report(rotation: number, probe: boolean, found: boolean, t: number, roll: number | null = null): boolean {
    if (this.fixed !== null) return false;
    if (probe) {
      if (found) {
        const hits = (this.probeHits.get(rotation) ?? 0) + 1;
        this.probeHits.set(rotation, hits);
        if (hits >= 2) {
          this.rotation = rotation; this.locked = false; this.hits = 0; this.probeHits.clear(); this.lastFace = t;
          return true;
        }
      }
      return false;
    }
    if (!found) {
      this.sideways = 0;
      if (this.locked && this.lastFace !== null && t - this.lastFace > this.relockAfter) { this.locked = false; this.hits = 0; }
      return false;
    }
    this.lastFace = t;
    const quarter = roll !== null && Math.abs(roll) > 65 ? ((Math.round(roll / 90) % 4) + 4) % 4 : 0;
    if (quarter && !this.locked) {
      this.sideways += 1;
      if (this.sideways >= 2) {
        this.rotation = (((this.rotation - 90 * quarter) % 360) + 360) % 360;
        this.hits = 0; this.sideways = 0;
        return true;
      }
      return false;
    }
    this.sideways = 0; this.probeHits.clear();
    if (!this.locked) { this.hits += 1; this.locked = this.hits >= 2; }
    return false;
  }
}

export interface FaceResult {
  trackId: number;
  detection: Detection;
  geometry: FaceGeometry;
  viewState: ViewState;
  quality: QualityReport;
  accepted: boolean;
  speedIodPerS: number;
}

export interface FrameResult {
  frameIndex: number;
  timeS: number;
  rotationDeg: number;
  faces: FaceResult[];
  primaryTrack: number | null;
  guidance: Guidance;
  capturedViews: string[];
  missingViews: string[];
  events: CaptureEvent[];
  rejected: Record<string, number>;
  timingsMs: Record<string, number>;
}

const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());

interface Context { classifier: ViewClassifier; shots: TrackShots }

export class CaptureEngine {
  readonly config: CaptureConfig;
  readonly views: Map<string, ViewSpec>;
  readonly viewNames: string[];
  readonly requiredViews: ReadonlySet<string>;
  readonly assessor: QualityAssessor;
  readonly tracker: ByteTracker;
  readonly orientation: OrientationLock;
  contexts = new Map<number, Context>();
  finished: TrackShots[] = [];
  events: CaptureEvent[] = [];
  primaryId: number | null = null;
  frames = 0;
  dropped = { non_increasing_timestamp: 0 };
  private previousTime: number | null = null;
  private bodyBoxes: Box[] = [];
  private bodyScores: number[] = [];
  private bodyTime = -Infinity;
  private bodyChecked = -Infinity;
  private roiChecked = -Infinity;
  private history: [number, Guidance][] = [];
  private shownCode: string | null = null;

  constructor(config: Partial<CaptureConfig> | Record<string, unknown> = {}, public detector: FaceDetector | null = null,
    public bodyGate: BodyGate | null = null, mask?: Uint8Array) {
    this.config = mergeConfig(defaultConfig(), config);
    this.views = new Map(this.config.pose.views.map((v) => [v.name, v]));
    this.viewNames = this.config.pose.views.map((v) => v.name);
    this.requiredViews = new Set(this.config.pose.views.filter((v) => v.required).map((v) => v.name));
    this.assessor = new QualityAssessor(this.config.quality, mask);
    this.tracker = new ByteTracker(this.config.tracker);
    this.orientation = new OrientationLock(this.config.runtime.rotate);
  }

  private context(id: number): Context {
    let ctx = this.contexts.get(id);
    if (!ctx) {
      ctx = { classifier: new ViewClassifier(this.config.pose),
        shots: new TrackShots(id, this.viewNames, this.config.selection,
          Object.fromEntries(this.config.pose.views.map((v) => [v.name, v.shots]))) };
      this.contexts.set(id, ctx);
    }
    return ctx;
  }

  private archive(id: number, t: number) {
    const ctx = this.contexts.get(id);
    if (!ctx) return;
    this.contexts.delete(id);
    ctx.shots.finalize();
    if (ctx.shots.captured().length) {
      this.finished.push(ctx.shots);
      this.finished.sort((a, b) => compareRank(b.rank(), a.rank()));
      this.finished.length = Math.min(this.finished.length, 4);
      this.pushEvents([{ event: 'track_finished', view: '', track_id: id, time_s: t }]);
    }
  }

  private pushEvents(events: CaptureEvent[]) {
    this.events.push(...events);
    if (this.events.length > 1024) this.events.splice(0, this.events.length - 1024);
  }

  private async refreshBodies(frame: Frame, t: number, timings: Record<string, number>) {
    if (!this.bodyGate) return;
    const g = this.config.body_gate;
    if (t - this.bodyChecked >= g.period_s || t < this.bodyChecked) {
      const started = now();
      const { boxes, scores } = await this.bodyGate.detect(frame);
      timings.body_gate = (timings.body_gate ?? 0) + now() - started;
      this.bodyChecked = t;
      if (boxes.length) { this.bodyBoxes = boxes; this.bodyScores = scores; this.bodyTime = t; }
    }
  }

  private async checkDog(frame: Frame, det: Detection, t: number, timings: Record<string, number>) {
    const g = this.config.body_gate;
    await this.refreshBodies(frame, t, timings);
    const cx = (det.box[0] + det.box[2]) / 2, cy = (det.box[1] + det.box[3]) / 2;
    if (t - this.bodyTime <= g.hold_s) {
      for (let i = 0; i < this.bodyBoxes.length; i++) {
        const b = this.bodyBoxes[i], mx = (b[2] - b[0]) * g.margin, my = (b[3] - b[1]) * g.margin;
        if (b[0] - mx <= cx && cx <= b[2] + mx && b[1] - my <= cy && cy <= b[3] + my) {
          det.bodyBox = b; det.bodyScore = this.bodyScores[i];
          return true;
        }
      }
    }
    const size = Math.max(width(det), height(det)) * 1.5;
    const x1 = Math.floor(Math.max(0, cx - size)), y1 = Math.floor(Math.max(0, cy - size));
    const x2 = Math.floor(Math.min(frame.width, cx + size)), y2 = Math.floor(Math.min(frame.height, cy + size));
    if (x2 - x1 < 32 || y2 - y1 < 32 || !this.bodyGate) return false;
    const started = now();
    const { boxes, scores } = await this.bodyGate.detect(crop(frame, x1, y1, x2, y2));
    timings.head_gate = (timings.head_gate ?? 0) + now() - started;
    for (let i = 0; i < boxes.length; i++) {
      const b = boxes[i];
      if (scores[i] >= g.head_score_threshold && b[0] <= cx - x1 && cx - x1 <= b[2] && b[1] <= cy - y1 && cy - y1 <= b[3]) {
        det.bodyBox = [b[0] + x1, b[1] + y1, b[2] + x1, b[3] + y1]; det.bodyScore = scores[i];
        return true;
      }
    }
    return false;
  }

  private async trackIsDog(frame: Frame, track: Track, t: number, timings: Record<string, number>) {
    if (!this.bodyGate || track.isDog) return true;
    if (t < track.dogRetry) return false;
    track.isDog = await this.checkDog(frame, track.detection, t, timings);
    if (!track.isDog) track.dogRetry = t + this.config.body_gate.period_s;
    return track.isDog;
  }

  private async roiZoom(frame: Frame, t: number, timings: Record<string, number>): Promise<Detection[]> {
    const d = this.config.detector;
    if (!d.roi_zoom || !this.bodyGate || !this.detector) return [];
    if (t - this.roiChecked < d.roi_period_s && t >= this.roiChecked) return [];
    this.roiChecked = t;
    await this.refreshBodies(frame, t, timings);
    if (t - this.bodyTime > this.config.body_gate.hold_s || !this.bodyBoxes.length) return [];
    const [x1, y1, x2, y2] = this.bodyBoxes.reduce((a, b) => ((b[2] - b[0]) * (b[3] - b[1]) > (a[2] - a[0]) * (a[3] - a[1]) ? b : a));
    const side = Math.max(x2 - x1, y2 - y1) * 1.15;
    if (Math.max(frame.width, frame.height) / Math.max(side, 1) < d.roi_min_zoom) return [];
    const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
    const left = Math.floor(Math.max(0, cx - side / 2)), top = Math.floor(Math.max(0, cy - side / 2));
    const right = Math.floor(Math.min(frame.width, cx + side / 2)), bottom = Math.floor(Math.min(frame.height, cy + side / 2));
    if (right - left < 32 || bottom - top < 32) return [];
    const started = now();
    const found = await this.detector.detect(crop(frame, left, top, right, bottom));
    timings.roi_detector = now() - started;
    return found.map((det) => ({ ...det, source: 'roi_zoom',
      box: [det.box[0] + left, det.box[1] + top, det.box[2] + left, det.box[3] + top] as Box,
      keypoints: det.keypoints.map(([x, y]) => [x + left, y + top] as [number, number]) }))
      .filter((det) => detectionReason(det) === null);
  }

  private choosePrimary(tracks: Track[]) {
    const size = (t: Track) => width(t.detection) * height(t.detection) * Math.max(t.score, 0.05);
    const current = tracks.find((t) => t.id === this.primaryId) ?? null;
    const best = tracks.reduce<Track | null>((a, t) => (a === null || size(t) > size(a) ? t : a), null);
    if (!current && this.primaryId !== null) {
      const lost = this.tracker.tracks.find((t) => t.id === this.primaryId);
      if (lost && lost.state === 'lost' && (this.previousTime ?? 0) - lost.lastTime < 0.5) return;
    }
    if (!best) { this.primaryId = null; return; }
    if (!current || size(best) > 1.5 * size(current)) this.primaryId = best.id;
  }

  private uprightEvidence(dets: Detection[]): [boolean, number | null] {
    const confident = dets.filter((d) => d.score >= this.config.tracker.new_track_score);
    if (!confident.length) return [false, null];
    const best = confident.reduce((a, d) => (width(d) * height(d) > width(a) * height(a) ? d : a));
    return [true, faceGeometry(best, this.config.pose).roll_deg];
  }

  /** Analyse one RGBA frame captured at `t` seconds; null if the frame was dropped. */
  async process(frame: Frame, t: number, frameIndex?: number, detections?: Detection[]): Promise<FrameResult | null> {
    const started = now();
    if (!Number.isFinite(t) || (this.previousTime !== null && t <= this.previousTime)) {
      if (this.config.runtime.timestamps === 'raise') throw new Error('Capture timestamps must be finite and strictly increasing');
      this.dropped.non_increasing_timestamp += 1;
      return null;
    }
    const index = frameIndex ?? this.frames;
    this.previousTime = t;
    this.frames += 1;
    const timings: Record<string, number> = {}, rejected: Record<string, number> = {};
    const external = detections !== undefined;
    const [rotation, probe] = external ? [0, false] : this.orientation.next(t);
    let work = rotate(frame, rotation);
    if (this.config.runtime.mirror && !external) work = mirror(work);
    let found: Detection[];
    const t0 = now();
    if (external) found = detections!;
    else if (!this.detector) throw new Error('No detector configured; pass detections for upstream observations');
    else { found = await this.detector.detect(work); timings.detector = now() - t0; }
    let valid: Detection[] = [];
    for (const det of found) {
      const reason = detectionReason(det);
      if (reason) rejected[reason] = (rejected[reason] ?? 0) + 1; else valid.push(det);
    }
    if (probe) {
      const [ok, roll] = this.uprightEvidence(valid);
      if (this.orientation.report(rotation, true, ok && roll !== null && Math.abs(roll) <= 45, t, roll)) this.resetForRotation(t);
      timings.orientation_probe = 1;
      return this.result([], [], index, t, this.orientation.rotation, rejected, timings, started);
    }
    if (!external && !valid.some((d) => d.score >= this.config.tracker.high_score)) {
      valid = valid.concat(await this.roiZoom(work, t, timings));
    }
    const [ok, roll] = this.uprightEvidence(valid);
    if (!external && this.orientation.report(rotation, false, ok, t, roll)) this.resetForRotation(t);
    const t1 = now();
    let tracks = this.tracker.update(valid, t);
    for (const track of this.tracker.removed) this.archive(track.id, t);
    timings.tracking = now() - t1;
    const dogs: Track[] = [];
    for (const track of tracks) if (await this.trackIsDog(work, track, t, timings)) dogs.push(track);
    if (dogs.length < tracks.length) rejected.not_dog = tracks.length - dogs.length;
    tracks = dogs;
    if (!tracks.length && this.bodyGate) await this.refreshBodies(work, t, timings);
    tracks = tracks.sort((a, b) => width(b.detection) * height(b.detection) - width(a.detection) * height(a.detection))
      .slice(0, this.config.detector.max_faces);
    this.choosePrimary(tracks);
    const t2 = now();
    const faces: FaceResult[] = [], events: CaptureEvent[] = [];
    for (const track of tracks) {
      const [face, newEvents] = this.analyse(work, track, t, index, rotation);
      faces.push(face); events.push(...newEvents);
    }
    timings.analysis = now() - t2;
    return this.result(faces, events, index, t, rotation, rejected, timings, started);
  }

  private resetForRotation(t: number) {
    for (const track of this.tracker.tracks) this.archive(track.id, t);
    this.tracker.reset();
    this.bodyTime = this.bodyChecked = -Infinity;
  }

  private analyse(frame: Frame, track: Track, t: number, index: number, rotation: number): [FaceResult, CaptureEvent[]] {
    const ctx = this.context(track.id);
    const det = track.detection;
    const geometry = faceGeometry(det, this.config.pose);
    const state = ctx.classifier.update(t, geometry);
    const events = ctx.shots.observe(state, t);
    const target = state.raw ? this.views.get(state.raw) ?? null : null;
    const [vx, vy] = track.velocity(this.config.tracker.nominal_fps);
    const speed = Math.hypot(vx, vy);
    const inBand = state.view !== null && state.raw === state.view && !ctx.shots.locked.has(state.view);
    const full = inBand || track.id === this.primaryId;
    const { report, aligned } = this.assessor.assess(frame, det, geometry, target, speed, full);
    const face: FaceResult = { trackId: track.id, detection: det, geometry, viewState: state, quality: report,
      accepted: false, speedIodPerS: speed / Math.max(geometry.iod, 1e-6) };
    const eligible = inBand && report.passed && ctx.shots.eligibleLevel(report.level);
    if (eligible && aligned && ctx.shots.wants(state.view!, report.score)) {
      const shot: Shot = { view: state.view!, trackId: track.id, frameIndex: index, timeS: t, score: report.score,
        quality: report, geometry, detection: det, frame: this.config.output.keep_frames ? copyFrame(frame) : null,
        rotationDeg: rotation, confirmedAt: null };
      const [newEvents, kept] = ctx.shots.offer(shot, state, t);
      face.accepted = kept;
      events.push(...newEvents);
    }
    this.pushEvents(events);
    return [face, events];
  }

  private debounce(advice: Guidance, t: number, windowS = 0.6): Guidance {
    this.history.push([t, advice]);
    while (this.history.length && t - this.history[0][0] > windowS) this.history.shift();
    if (advice.code === 'done' || advice.code === 'hold_still') { this.shownCode = advice.code; return advice; }
    const counts = new Map<string, number>();
    for (const [, item] of this.history) counts.set(item.code, (counts.get(item.code) ?? 0) + 1);
    let code = advice.code, best = -1;
    for (const [k, n] of counts) {
      const key = n * 4 + (k === this.shownCode ? 2 : 0) + (k === advice.code ? 1 : 0);
      if (key > best) { best = key; code = k; }
    }
    this.shownCode = code;
    for (let i = this.history.length - 1; i >= 0; i--) if (this.history[i][1].code === code) return this.history[i][1];
    return advice;
  }

  private dogSeen(t: number) { return !!this.bodyGate && this.bodyBoxes.length > 0 && t - this.bodyTime <= this.config.body_gate.hold_s; }

  private result(faces: FaceResult[], events: CaptureEvent[], index: number, t: number, rotation: number,
    rejected: Record<string, number>, timings: Record<string, number>, started: number): FrameResult {
    const primary = faces.find((f) => f.trackId === this.primaryId) ?? null;
    const ctx = this.primaryId !== null ? this.contexts.get(this.primaryId) : undefined;
    const captured = ctx ? ctx.shots.captured() : [];
    const missing = this.viewNames.filter((v) => !captured.includes(v));
    // Guide towards views that can still change (a captured view keeps improving until it locks),
    // required ones first; "done" only once nothing is outstanding.
    const targets = ctx ? ctx.shots.outstanding(this.requiredViews)
      : [...this.viewNames].sort((a, b) => +!this.requiredViews.has(a) - +!this.requiredViews.has(b));
    let advice = guidance(primary, targets, { rejectedSpecies: !!rejected.not_dog, speedIod: primary?.speedIodPerS ?? 0,
      motionRef: this.config.quality.motion_ref_iod_per_s, dogSeen: !primary && this.dogSeen(t) });
    advice = this.debounce(advice, t);
    timings.total = now() - started;
    return { frameIndex: index, timeS: t, rotationDeg: rotation, faces, primaryTrack: this.primaryId, guidance: advice,
      capturedViews: captured, missingViews: missing, events, rejected, timingsMs: timings };
  }

  /** Primary dog has every required view and no view can still improve. */
  complete(): boolean {
    const ctx = this.primaryId !== null ? this.contexts.get(this.primaryId) : undefined;
    if (!ctx) return false;
    const required = this.config.pose.views.filter((v) => v.required).map((v) => v.name);
    if (!required.every((v) => ctx.shots.full(v))) return false;
    return this.config.selection.mode === 'best' || required.every((v) => ctx.shots.locked.has(v));
  }

  bundles(): TrackShots[] {
    const items = new Set<TrackShots>([...[...this.contexts.values()].map((c) => c.shots).filter((s) => s.captured().length),
      ...this.finished]);
    return [...items].sort((a, b) => compareRank(b.rank(), a.rank()));
  }

  /** The bundle that will be submitted: the primary dog's, else the best-ranked one. */
  bestBundle(): TrackShots | null {
    const primary = this.primaryId !== null ? this.contexts.get(this.primaryId)?.shots : undefined;
    if (primary && primary.captured().length) return primary;
    return this.bundles()[0] ?? null;
  }

  /** Clear views on every track (operator asked to retake them, or the server rejected them). */
  retake(views: string[]) {
    for (const ctx of this.contexts.values()) for (const v of views) ctx.shots.clear(v);
    for (const shots of this.finished) for (const v of views) shots.clear(v);
  }

  finalize(): TrackShots | null {
    for (const ctx of this.contexts.values()) ctx.shots.finalize();
    return this.bestBundle();
  }

  /** Aligned 224 crop of a shot for previews (the upload is the original frame). */
  static thumbnail(shot: Shot, size = 224): Frame | null {
    if (!shot.frame || !shot.geometry) return null;
    return alignFace(shot.frame, shot.geometry.points, size).image;
  }
}

function copyFrame(frame: Frame): Frame {
  return { width: frame.width, height: frame.height, data: new Uint8ClampedArray(frame.data) };
}

function compareRank(a: number[], b: number[]) {
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return a[i] - b[i];
  return 0;
}

export { iou };
