"""The low-compute fallback defers candidates, not their acceptance checks."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from dog_capture.engine import CaptureConfig, CaptureEngine
from dog_capture.engine import cli
from dog_capture.engine.pose import ViewState
from dog_capture.engine.quality import QualityReport
from dog_capture.engine.tracking import Track
from dog_engine_synthetic import render


def test_candidate_timing_is_opt_in_and_round_trips():
    assert CaptureConfig().selection.candidate_timing == 'early'
    config = CaptureConfig().merged({'selection': {'candidate_timing': 'stable'}})
    assert CaptureConfig().merged(config.to_dict()).selection.candidate_timing == 'stable'
    with pytest.raises(ValueError, match='selection.candidate_timing'):
        CaptureConfig().merged({'selection': {'candidate_timing': 'automatic'}})


@pytest.mark.parametrize('timing', ['early', 'stable'])
def test_cli_exposes_candidate_timing(monkeypatch, timing):
    seen = []
    monkeypatch.setattr(cli, 'cmd_run', lambda args: seen.append(cli.load_config(args)) or 0)
    assert cli.main(['run', 'unused.mp4', '--candidate-timing', timing]) == 0
    assert seen[0].selection.candidate_timing == timing


def setup(monkeypatch, timing='stable', *, passes=1, still=False):
    config = CaptureConfig().merged({'selection': {'candidate_timing': timing, 'mode': 'best'},
        'runtime': {'still_images': still}, 'refine': {'gate_min_passes': passes, 'gate_min_ratio': 0}})
    engine = CaptureEngine(config)
    engine.refiner = SimpleNamespace(landmarker=object())
    frame, detections = render(center=(160, 160), iod=50, size=(320, 320))
    detections[0].source = 'test'
    track = Track(1, detections[0], 0, engine.tracker.kalman)
    current = {'index': 0, 'quality': QualityReport(score=.9, level='high', passed=True,
        factors={'sharpness': .9, 'eye_detail': .9})}
    aligned = SimpleNamespace(image=np.zeros((224, 224, 3), np.uint8))
    monkeypatch.setattr(engine.assessor, 'assess', lambda *a, **kw: (copy.deepcopy(current['quality']), aligned))
    precise, verified = [], []
    def sharpen(*args):
        precise.append(current['index'])
        return None, None
    def unreliable(*args):
        verified.append(current['index'])
        return current.get('problem')
    monkeypatch.setattr(engine, '_sharpen', sharpen)
    monkeypatch.setattr(engine, '_unreliable', unreliable)
    context = engine._context(track.id)
    def step(index, stable, score=.9):
        current['index'] = index
        current['quality'].score = score
        current['quality'].factors = {'sharpness': score, 'eye_detail': score}
        if stable is not None:
            monkeypatch.setattr(context.classifier, 'update', lambda *a: ViewState('front', 'front', stable, 0))
        engine.frame_times.append(index / 30)
        return engine._analyse(frame, track, index / 30, index, 0, still=still)[0]
    return engine, context, current, precise, verified, step


@pytest.mark.parametrize('ranking', ['quality', 'clarity'])
@pytest.mark.parametrize('timing,expected', [('early', 0), ('stable', 2)])
def test_only_explicit_fallback_discards_early_better_candidate(monkeypatch, ranking, timing, expected):
    engine, ctx, _, precise, verified, step = setup(monkeypatch, timing)
    engine.config.selection.ranking = ranking
    step(0, False, .9)
    step(1, False, .85)
    step(2, True, .8)
    assert precise == verified == [expected]
    assert ctx.shots.best('front').frame_index == expected
    assert not ctx.shots.pending['front']


def test_stable_pose_still_needs_temporal_verification(monkeypatch):
    _, ctx, _, precise, verified, step = setup(monkeypatch, passes=2)
    assert not step(0, False).accepted
    first = step(1, True)
    assert not first.accepted and 'landmarks_unstable' in first.quality.reasons
    assert step(2, True).accepted
    assert precise == verified == [1, 2]
    assert ctx.shots.best('front').frame_index == 2


@pytest.mark.parametrize('failure', ['quality', 'verification'])
def test_fallback_never_bypasses_rejection(monkeypatch, failure):
    _, ctx, current, _, verified, step = setup(monkeypatch)
    if failure == 'quality':
        current['quality'].passed = False
        current['quality'].reasons = ['too_dark']
    else:
        current['problem'] = 'eyes_not_visible'
    result = step(0, True)
    assert not result.accepted and not ctx.shots.captured()
    assert verified == ([] if failure == 'quality' else [0])


def test_fallback_keeps_near_miss_evidence_before_dwell(monkeypatch):
    engine, ctx, current, precise, verified, step = setup(monkeypatch)
    current['quality'].level = 'low'
    diagnostics = []
    monkeypatch.setattr(engine, '_near_miss', lambda *a: diagnostics.append(True) or 'pass')
    assert not step(0, False, .4).accepted
    assert diagnostics == [True] and not precise and not verified
    assert ctx.evidence.observations[-1][-1] == 'pass'


def test_pose_break_stops_candidate_checks_again(monkeypatch):
    _, _, _, precise, verified, step = setup(monkeypatch)
    for i, stable in enumerate((False, True, False, False, True)):
        step(i, stable, .7 + .02 * i)
    assert precise == verified == [1, 4]


def test_still_photo_does_not_wait_for_video_dwell(monkeypatch):
    _, ctx, _, precise, verified, step = setup(monkeypatch, still=True)
    assert step(0, None).accepted
    assert precise == verified == [0]
    assert ctx.shots.best('front').frame_index == 0
