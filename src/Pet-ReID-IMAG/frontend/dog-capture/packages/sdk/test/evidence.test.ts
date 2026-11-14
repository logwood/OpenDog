import { expect, it } from 'vitest';
import { defaultConfig } from '../src/config';
import { TemporalEvidence, type GateStatus } from '../src/evidence';

it('counts one vote per frame and rejects time reversal', () => {
  const evidence = new TemporalEvidence(), config = defaultConfig().refine;
  for (let i = 0; i < 4; i++) evidence.record(0, [0, 0, 100], 'pass');
  expect(evidence.rejectReason([0, 0, 100], 2, config)).toBe('landmarks_unstable');
  evidence.record(1, [0, 0, 100], 'pass');
  expect(evidence.rejectReason([0, 0, 100], 3, config)).toBeNull();
  evidence.record(1, [0, 0, 100], 'fail');
  expect(evidence.rejectReason([0, 0, 100], 3, config)).toBe('landmarks_unstable');
  expect(() => evidence.record(0, [0, 0, 100], 'pass')).toThrow('decrease');
});

it.each<GateStatus>(['pass', 'present', 'fail'])('keeps %s distinct from missing evidence', (status) => {
  const config = { ...defaultConfig().refine, gate_min_passes: 2, gate_min_ratio: .75 };
  const evidence = new TemporalEvidence();
  evidence.record(0, [0, 0, 100], 'pass');
  evidence.record(1, [0, 0, 100], status);
  expect(evidence.rejectReason([0, 0, 100], 3, config)).toBe(status === 'fail' ? 'landmarks_unstable' : null);
});

it('keeps the spatial boundary and missing-frame penalty', () => {
  const evidence = new TemporalEvidence(), config = defaultConfig().refine;
  evidence.record(0, [0, 0, 100], 'pass'); evidence.record(1, [50, 0, 100], 'pass');
  expect(evidence.rejectReason([50, 0, 100], 3, config)).toBeNull();
  expect(evidence.rejectReason([50.001, 0, 100], 3, config)).toBe('landmarks_unstable');
  expect(evidence.rejectReason([50, 0, 100], 20, config)).toBe('landmarks_unstable');
});

it('releases observations outside the processed-frame window', () => {
  const evidence = new TemporalEvidence();
  for (let i = 0; i < 1000; i++) {
    evidence.trim(i - 4); evidence.record(i, [0, 0, 100], 'present');
    expect(evidence.observations.length).toBeLessThanOrEqual(5);
  }
  expect(evidence.observations[0][0]).toBe(995);
});
