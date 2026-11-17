import { describe, expect, it } from 'vitest';
import { clarityScore, defaultConfig, mergeConfig, TrackShots, type QualityReport, type Shot } from '../src';

const quality = (score: number, detail: number): QualityReport => ({ score, level: 'medium', passed: true,
  reasons: [], metrics: {}, factors: { sharpness: detail, eye_detail: detail } });
const shot = (timeS: number, score: number, detail: number): Shot => ({ view: 'front', trackId: 1,
  frameIndex: Math.round(timeS * 30), timeS, score, quality: quality(score, detail), geometry: null, detection: null,
  frame: null, rotationDeg: 0, confirmedAt: null });
const state = (stable = true) => ({ view: 'front', stable });
const select = (overrides = {}) => new TrackShots(1, ['front'],
  mergeConfig(defaultConfig(), { selection: { mode: 'best', ranking: 'clarity', ...overrides } }).selection);

it('keeps the ranking opt-in and validates it', () => {
  expect(defaultConfig().selection.ranking).toBe('quality');
  expect(() => mergeConfig(defaultConfig(), { selection: { ranking: 'sharpest' } })).toThrow('selection.ranking');
});

describe.each([false, true])('confirmed=%s', (stable) => {
  it.each(['quality', 'clarity'])('orders pending and kept shots by %s', (ranking) => {
    const s = select({ ranking });
    s.offer(shot(0, .85, .4), state(stable), 0);
    const sharper = shot(.1, .6, .9);
    expect(s.wants('front', s.scoreFor(sharper.quality!))).toBe(ranking === 'clarity');
    const [events] = s.offer(sharper, state(stable), .1);
    if (!stable) s.observe(state(), .2);
    expect(s.best('front')!.frameIndex).toBe(ranking === 'clarity' ? 3 : 0);
    if (stable && ranking === 'clarity') expect(events[0]).toMatchObject({ event: 'view_improved', score: .6 });
  });
});

it('uses clarity for top-k spacing and keeps ties stable', () => {
  const s = select({ top_k: 2, min_gap_s: .3 });
  for (const [t, score, detail] of [[0, .85, .4], [.1, .6, .9], [.5, .75, .7], [.6, .9, .6]]) {
    s.offer(shot(t, score, detail), state(), t);
  }
  expect(s.slots.front.map((x) => x.frameIndex)).toEqual([3, 15]);
  expect(s.wants('front', s.scoreFor(quality(.99, .7)))).toBe(false);
});

it.each(['fast', 'hybrid'])('does not trigger %s quality locks using clarity', (mode) => {
  const s = select({ mode, improve_window_s: 1 });
  s.offer(shot(0, .6, .95), state(), 0);
  expect(s.firstGood.front).toBeNull();
  s.observe(state(), 1.1);
  expect(s.locked.size).toBe(0);
  s.observe(state(), 2.01);
  expect([...s.locked]).toEqual(['front']);
});

it('uses fixed detail factors and ranks invalid diagnostics low', () => {
  expect(clarityScore({ sharpness: .8, eye_detail: .4, pose_fit: .01 })).toBeCloseTo(.8 ** .6 * .4 ** .4, 12);
  const invalid: Record<string, number>[] = [{}, { sharpness: NaN, eye_detail: 1 }, { sharpness: 1, eye_detail: 2 }];
  for (const factors of invalid) {
    expect(clarityScore(factors)).toBe(0);
  }
});
