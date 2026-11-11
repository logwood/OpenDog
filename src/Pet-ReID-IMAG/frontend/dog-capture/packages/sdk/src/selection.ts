/** Per-track best-shot selection: fast / best / hybrid (port of selection.py). */
import type { CaptureConfig } from './config';
import type { Frame } from './image';
import type { FaceGeometry, ViewState } from './pose';
import type { QualityReport } from './quality';
import type { Detection } from './schema';

export const LEVEL_ORDER = { reject: 0, low: 1, medium: 2, high: 3 } as const;

export interface Shot {
  view: string;
  trackId: number;
  frameIndex: number;
  timeS: number;
  score: number;
  quality: QualityReport | null;
  geometry: FaceGeometry | null;
  detection: Detection | null;
  frame: Frame | null;
  rotationDeg: number;
  confirmedAt: number | null;
}

export interface CaptureEvent { event: string; view: string; track_id: number; time_s: number; frame_index?: number; score?: number }

type Selection = CaptureConfig['selection'];
interface StateLike { view: string | null; stable: boolean }

export class TrackShots {
  slots: Record<string, Shot[]> = {};
  pending: Record<string, Shot[]> = {};
  firstGood: Record<string, number | null> = {};
  locked = new Set<string>();
  k: Record<string, number> = {};
  visit: string | null = null;
  firstTime: number | null = null;
  lastTime: number | null = null;

  constructor(public trackId: number, public viewNames: string[], private config: Selection,
    shots: Record<string, number> = {}) {
    for (const v of viewNames) {
      this.slots[v] = []; this.pending[v] = []; this.firstGood[v] = null;
      this.k[v] = shots[v] || config.top_k;
    }
  }

  static levelOk(level: QualityReport['level'], minimum: Selection['min_level']) {
    return LEVEL_ORDER[level] >= LEVEL_ORDER[minimum];
  }

  eligibleLevel(level: QualityReport['level']) { return TrackShots.levelOk(level, this.config.min_level); }

  wants(view: string, score: number) {
    if (!(view in this.slots) || this.locked.has(view)) return false;
    for (const bucket of [this.slots[view], this.pending[view]]) {
      if (bucket.length >= this.k[view] && score <= Math.min(...bucket.map((s) => s.score))) return false;
    }
    return true;
  }

  captured() { return this.viewNames.filter((v) => this.slots[v].length > 0); }
  full(view: string) { return this.slots[view].length >= this.k[view]; }
  best(view: string): Shot | null { return this.slots[view]?.[0] ?? null; }

  rank(): [number, number, number] {
    const views = this.captured();
    return [views.length, views.length && views.includes(this.viewNames[0]) ? 1 : 0,
      views.reduce((s, v) => s + this.slots[v][0].score, 0)];
  }

  observe(state: StateLike | ViewState, t: number): CaptureEvent[] {
    if (this.firstTime === null) this.firstTime = t;
    this.lastTime = t;
    if (state.view !== this.visit) {
      if (this.visit !== null) this.pending[this.visit] = [];
      this.visit = state.view;
    }
    const events: CaptureEvent[] = [];
    if (this.visit !== null && state.stable && this.pending[this.visit].length) {
      for (const shot of this.pending[this.visit]) { shot.confirmedAt = t; events.push(...this.insert(shot, t)); }
      this.pending[this.visit] = [];
    }
    this.updateLocks(t);
    return events;
  }

  offer(shot: Shot, state: StateLike | ViewState, t: number): [CaptureEvent[], boolean] {
    if (this.locked.has(shot.view) || state.view !== shot.view) return [[], false];
    if (!state.stable) {
      const bucket = this.pending[shot.view];
      bucket.push(shot);
      bucket.sort((a, b) => b.score - a.score);
      bucket.length = Math.min(bucket.length, this.k[shot.view]);
      return [[], bucket.includes(shot)];
    }
    shot.confirmedAt = t;
    const events = this.insert(shot, t);
    this.updateLocks(t);
    return [events, this.slots[shot.view].includes(shot)];
  }

  private insert(shot: Shot, t: number): CaptureEvent[] {
    const c = this.config;
    if (this.locked.has(shot.view)) return [];
    const slot = this.slots[shot.view];
    const wasEmpty = slot.length === 0;
    const previousBest = slot.length ? slot[0].score : null;
    const k = this.k[shot.view];
    if (k > 1) {
      const near = slot.filter((s) => Math.abs(s.timeS - shot.timeS) < c.min_gap_s);
      if (near.length) {
        if (shot.score <= Math.max(...near.map((s) => s.score))) return [];
        for (const s of near) slot.splice(slot.indexOf(s), 1);
      }
    }
    slot.push(shot);
    slot.sort((a, b) => b.score - a.score);
    slot.length = Math.min(slot.length, k);
    if (!slot.includes(shot)) return [];
    const events: CaptureEvent[] = [];
    const good = shot.score >= c.good_enough;
    if (good && this.firstGood[shot.view] === null) this.firstGood[shot.view] = t;
    const base = { view: shot.view, track_id: this.trackId, frame_index: shot.frameIndex, time_s: shot.timeS, score: shot.score };
    if (wasEmpty) events.push({ event: 'view_captured', ...base });
    else if (previousBest !== null && shot.score > previousBest + 0.01) events.push({ event: 'view_improved', ...base });
    if (c.mode === 'fast' && slot.length >= k && Math.min(...slot.map((s) => s.score)) >= c.good_enough) {
      this.locked.add(shot.view);
      events.push({ event: 'view_locked', view: shot.view, track_id: this.trackId, time_s: t });
    }
    return events;
  }

  private updateLocks(t: number) {
    if (this.config.mode !== 'hybrid') return;
    for (const [view, first] of Object.entries(this.firstGood)) {
      if (first !== null && !this.locked.has(view) && this.full(view) && t - first >= this.config.improve_window_s) {
        this.locked.add(view);
      }
    }
  }

  finalize() { for (const v of this.viewNames) this.pending[v] = []; }

  /** Operator retake: forget a view so it is captured again. */
  clear(view: string) {
    if (!(view in this.slots)) return;
    this.slots[view] = []; this.pending[view] = []; this.firstGood[view] = null; this.locked.delete(view);
  }
}
