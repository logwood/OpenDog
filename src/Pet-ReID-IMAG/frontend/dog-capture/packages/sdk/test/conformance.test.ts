/** The Python engine is the reference: reproduce tools/export_capture_conformance.py vectors. */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { inflateSync } from 'node:zlib';
import { describe, expect, it } from 'vitest';
import {
  ByteTracker, SETTLE_FACTOR, blurEffect, coreMask, defaultConfig, eyePatches, faceGeometry, fromBGR, getSchema, guidance,
  laplacianEnergy, mergeConfig, OneEuroFilter, QualityAssessor, similarity, toGray, TrackShots, ViewClassifier,
  type Detection, type Point,
} from '../src';

const DIR = resolve(__dirname, '../../../../../dog_capture/protocol/conformance');
const load = (name: string) => JSON.parse(readFileSync(resolve(DIR, name), 'utf8'));
const unpack = (packed: { shape: number[]; deflate_b64: string }) =>
  new Uint8Array(inflateSync(Buffer.from(packed.deflate_b64, 'base64')));
const close = (a: number, b: number, tol: number) => expect(Math.abs(a - b)).toBeLessThanOrEqual(tol);
const config = defaultConfig();

describe('configuration', () => {
  it('has the reference defaults', () => {
    const ref = load('config_default.json');
    for (const section of Object.keys(ref)) {
      for (const [key, value] of Object.entries(ref[section] as Record<string, unknown>)) {
        expect((config as unknown as Record<string, Record<string, unknown>>)[section][key], `${section}.${key}`).toEqual(value);
      }
    }
  });
  it('has the reference keypoint schemas', () => {
    for (const [name, ref] of Object.entries(load('schemas.json') as Record<string, { names: string[]; yaw_scale: number }>)) {
      expect(getSchema(name).names).toEqual(ref.names);
      expect(getSchema(name).yawScale).toBeCloseTo(ref.yaw_scale, 9);
    }
  });
});

describe('geometry', () => {
  it('matches every case', () => {
    for (const c of load('geometry.json')) {
      const det: Detection = { box: c.box, score: 0.9, keypoints: c.keypoints, schema: getSchema(c.schema), kpScores: c.kp_scores };
      const g = faceGeometry(det, config.pose);
      const e = c.expected;
      expect(g.valid, JSON.stringify(c.keypoints)).toBe(e.valid);
      expect(g.reason).toBe(e.reason);
      expect(g.swapped).toBe(e.swapped);
      for (const k of ['yaw', 'pitch', 'roll_deg', 'iod', 'approx_deg'] as const) close(g[k], e[k], 1e-4);
      if (e.lower_yaw === null) expect(g.lower_yaw).toBeNull(); else close(g.lower_yaw!, e.lower_yaw, 1e-4);
      for (const [name, p] of Object.entries(e.points as Record<string, Point>)) {
        close(g.points[name][0], p[0], 1e-3); close(g.points[name][1], p[1], 1e-3);
      }
    }
  });
});

describe('similarity alignment', () => {
  it('matches Umeyama', () => {
    for (const c of load('similarity.json')) {
      const m = similarity(c.src, c.dst);
      for (let i = 0; i < 2; i++) for (let j = 0; j < 3; j++) close(m[i][j], c.matrix[i][j], 1e-5); // inputs rounded to 1e-6
    }
  });
});

describe('view classifier and One-Euro', () => {
  it('matches the debounced view sequences', () => {
    for (const [name, c] of Object.entries(load('views.json') as Record<string, { steps: [number, number, number, boolean][]; expected: any[] }>)) {
      const classifier = new ViewClassifier(config.pose);
      c.steps.forEach(([t, yaw, roll, valid], i) => {
        const s = classifier.update(t, { yaw, roll_deg: roll, valid } as never);
        const e = c.expected[i];
        expect([s.raw, s.view, s.stable, s.observations], `${name}[${i}]`).toEqual([e.raw, e.view, e.stable, e.observations]);
        if (e.smoothed_yaw === null) expect(s.smoothed_yaw).toBeNull(); else close(s.smoothed_yaw!, e.smoothed_yaw, 1e-6);
      });
    }
  });
  it('matches the One-Euro filter', () => {
    const c = load('one_euro.json');
    const f = new OneEuroFilter(c.min_cutoff, c.beta);
    c.steps.forEach(([t, x]: [number, number], i: number) => close(f.filter(t, x), c.expected[i], 1e-5));
  });
});

describe('tracking', () => {
  it('produces the same track ids and boxes', () => {
    const c = load('tracking.json');
    const tracker = new ByteTracker(config.tracker);
    c.frames.forEach((frame: { t: number; detections: number[][] }, i: number) => {
      const dets: Detection[] = frame.detections.map((d) => ({ box: [d[0], d[1], d[2], d[3]], score: d[4],
        keypoints: [[0, 0], [1, 0], [0.5, 1], [0, 1], [1, 1]], schema: getSchema('anyface5') }));
      const active = tracker.update(dets, frame.t).sort((a, b) => a.id - b.id);
      expect(active.map((a) => a.id), `frame ${i}`).toEqual(c.expected[i].map((e: { id: number }) => e.id));
      active.forEach((a, j) => a.box.forEach((v, k) => close(v, c.expected[i][j].box[k], 1e-2)));
    });
  });
});

describe('selection', () => {
  const c = load('selection.json');
  const run = (script: any[], cases: Record<string, any>) => {
    for (const [name, spec] of Object.entries(cases)) {
      const selection = { ...config.selection, mode: spec.mode, good_enough: c.good_enough,
        improve_window_s: c.improve_window_s, min_gap_s: c.min_gap_s };
      const ts = new TrackShots(1, ['front', 'left'], selection, { front: spec.front_shots });
      const events: [string, string, number][] = [];
      const outstanding: string[][] = [];
      for (const [action, view, stable, t, score] of script) {
        const state = { view, stable };
        if (action === 'observe') ts.observe(state, t).forEach((e) => events.push([e.event, e.view, e.time_s]));
        else ts.offer({ view, trackId: 1, frameIndex: Math.trunc(t * 30), timeS: t, score, quality: null, geometry: null,
          detection: null, frame: null, rotationDeg: 0, confirmedAt: null }, state, t)[0]
          .forEach((e) => events.push([e.event, e.view, e.time_s]));
        outstanding.push(ts.outstanding(new Set(c.required)));
      }
      for (const view of ['front', 'left']) {
        expect(ts.slots[view].map((s) => [s.timeS, s.score]), `${name}.${view}`).toEqual(spec.expected_slots[view]);
      }
      expect([...ts.locked].sort(), name).toEqual(spec.expected_locked);
      expect(events, name).toEqual(spec.expected_events);
      expect(outstanding, `${name} outstanding`).toEqual(spec.expected_outstanding);
    }
  };
  it('matches fast / best / hybrid with one and two front shots', () => run(c.script, c.cases));
  it('settles full views that never reach good_enough', () => {
    expect(SETTLE_FACTOR).toBe(c.settle_factor);
    run(c.settle_script, c.settle_cases);
  });
});

describe('quality', () => {
  const q = load('quality.json');
  const refMask = unpack(q.mask);
  it('has the same eye patches and nearly the same core mask', () => {
    expect(eyePatches()).toEqual(q.eye_patches);
    const mask = coreMask();
    let diff = 0, inside = 0;
    for (let i = 0; i < mask.length; i++) { diff += Number(mask[i] !== refMask[i]); inside += refMask[i]; }
    expect(diff / inside).toBeLessThan(0.02);
  });
  it('reproduces the pixel metrics on the reference aligned crops', () => {
    const assessor = new QualityAssessor(config.quality, refMask);
    expect(assessor.maskRect).toEqual(q.mask_rect);
    for (const f of q.frames) {
      const gray = unpack(f.aligned_gray);
      const e = f.expected_metrics_on_aligned_gray;
      const lap = laplacianEnergy(gray, 224, 224, refMask);
      expect(Math.abs(lap - e.laplacian) / Math.max(e.laplacian, 1), `${f.name} laplacian`).toBeLessThan(0.02);
      close(blurEffect(gray, 224, q.mask_rect), e.blur_effect, 0.01);
      assessor.eyes.forEach((rect, i) => {
        const v = laplacianEnergy(gray, 224, 224, null, rect);
        expect(Math.abs(v - e.eye_laplacian[i]) / Math.max(e.eye_laplacian[i], 1), `${f.name} eye ${i}`).toBeLessThan(0.02);
      });
    }
  });
  it('reproduces the full report from the original frame (warp included)', () => {
    const assessor = new QualityAssessor(config.quality);
    for (const f of q.frames) {
      const [h, w] = f.frame_bgr.shape;
      const frame = fromBGR(unpack(f.frame_bgr), w, h);
      const det: Detection = { box: f.box, score: 0.9, keypoints: f.keypoints, schema: getSchema('anyface5') };
      const g = faceGeometry(det, config.pose);
      const target = config.pose.views.find((v) => v.name === f.target)!;
      const { report, aligned } = assessor.assess(frame, det, g, target, f.speed_px_s);
      const e = f.expected_report;
      for (let i = 0; i < 2; i++) for (let j = 0; j < 3; j++) close(aligned!.matrix[i][j], f.expected_matrix[i][j], 1e-6);
      expect(report.passed, f.name).toBe(e.passed);
      expect(report.reasons, f.name).toEqual(e.reasons);
      expect(report.level, f.name).toBe(e.level);
      close(report.score, e.score, 0.03);
      for (const [k, v] of Object.entries(e.factors as Record<string, number>)) close(report.factors[k], v, 0.05);
      expect(toGray(aligned!.image).length).toBe(224 * 224);
    }
  });
});

describe('guidance', () => {
  it('maps the same situations to the same codes and messages', () => {
    for (const c of load('guidance.json')) {
      const i = c.input;
      const face = i.face === null ? null : { quality: { reasons: i.face.reasons },
        viewState: { view: i.face.view ?? null, stable: i.face.stable ?? false }, geometry: { valid: true } };
      const result = guidance(face, i.missing, { rejectedSpecies: i.rejected_species, speedIod: i.speed_iod,
        dogSeen: i.dog_seen });
      expect(result).toEqual(c.expected);
    }
  });
});

describe('mergeConfig', () => {
  it('applies a session engine config and rejects overlapping views', () => {
    const merged = mergeConfig(defaultConfig(), { selection: { mode: 'fast' }, pose: { views: [
      { name: 'front', center: 0, lower: -0.13, upper: 0.13, shots: 2 }] } });
    expect(merged.selection.mode).toBe('fast');
    expect(merged.pose.views[0]).toMatchObject({ shots: 2, required: true, max_roll_deg: 30 });
    expect(() => mergeConfig(defaultConfig(), { pose: { views: [
      { name: 'a', center: 0, lower: -0.2, upper: 0.2 }, { name: 'b', center: 0.3, lower: 0.1, upper: 0.5 }] } }))
      .toThrow(/overlap/);
  });
});
