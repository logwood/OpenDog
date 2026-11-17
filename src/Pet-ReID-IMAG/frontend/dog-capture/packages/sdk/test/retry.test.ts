import { expect, it } from 'vitest';
import { CandidateRetry } from '../src/retry';

function failed() {
  const gate = new CandidateRetry(.25);
  gate.observe('front');
  expect(gate.begin(0)).toBe(true);
  gate.finish(0, 'view_unconfirmed', false, true);
  return gate;
}

it('keeps the original wait within the same visit', () => {
  const gate = failed();
  for (const t of [.01, .1, .249]) { gate.observe('front'); expect(gate.begin(t)).toBe(false); }
  expect(gate.begin(.25)).toBe(true);
});

it.each(['front', 'left'])('allows one recovery on a new %s visit, not each flicker', (view) => {
  const gate = failed();
  gate.observe(null); gate.observe(view);
  expect(gate.begin(.03)).toBe(true);
  gate.finish(.03, 'eyes_not_visible', false, true);
  for (let i = 1; i < 20; i++) {
    gate.observe(null); gate.observe(i % 2 ? 'right' : 'left');
    expect(gate.begin(.03 + i * .01)).toBe(false);
  }
  expect(gate.begin(.28)).toBe(true);
});

it.each([null, 'landmarks_unstable'])('resumes temporal evidence after passing checks (%s)', (problem) => {
  const gate = failed();
  gate.observe(null); gate.observe('front');
  expect(gate.begin(.03)).toBe(true);
  gate.finish(.03, problem, true);
  expect(gate.begin(.04)).toBe(true);
  expect(gate.recoveryUsed).toBe(false);
});

it('reserves recovery only for candidates and keeps backoff after an inconclusive check', () => {
  const gate = failed();
  gate.observe(null); expect(gate.begin(.01)).toBe(false);
  gate.observe('left'); gate.observe('right');
  expect(gate.recoveryUsed).toBe(false);
  expect(gate.begin(.02)).toBe(true);
  gate.finish(.02, null, false);
  expect(gate.begin(.03)).toBe(false);
});

it('isolates tracks and supports a zero retry period', () => {
  const first = failed(), other = new CandidateRetry(.25);
  other.observe('front');
  expect(other.begin(.01)).toBe(true); expect(first.begin(.01)).toBe(false);
  const zero = new CandidateRetry(0); zero.observe('front');
  for (const t of [0, .01, .02]) { expect(zero.begin(t)).toBe(true); zero.finish(t, 'landmarks_unreliable', false); }
});

it('keeps original backoff after an initial acquisition failure', () => {
  const gate = new CandidateRetry(.25);
  gate.observe('front'); expect(gate.begin(0)).toBe(true);
  gate.finish(0, 'view_unconfirmed', false);
  for (const view of [null, 'left', null, 'right']) {
    gate.observe(view); expect(gate.begin(.1)).toBe(false);
  }
  expect(gate.begin(.25)).toBe(true);
});
