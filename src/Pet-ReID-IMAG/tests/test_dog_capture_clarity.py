"""Clarity changes ordering, while quality and landmark checks remain mandatory."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from dog_capture.engine import CaptureConfig, CaptureEngine
from dog_capture.engine.pose import face_geometry
from dog_capture.engine.quality import QualityReport
from dog_capture.engine.selection import Shot, TrackShots, clarity_score
from dog_capture.engine.tracking import Track
from dog_engine_synthetic import render


def report(score, detail, **kwargs):
    return QualityReport(score=score, level=kwargs.pop('level', 'medium'),
                         passed=kwargs.pop('passed', True),
                         factors={'sharpness': detail, 'eye_detail': detail}, **kwargs)


def shot(t, score, detail):
    return Shot('front', 1, round(t * 30), t, score, report(score, detail), None, None,
                np.zeros((4, 4, 3), np.uint8), None)


def state(stable=True):
    return SimpleNamespace(view='front', stable=stable)


def selector(**overrides):
    config = CaptureConfig().merged({'selection': {'ranking': 'clarity', 'mode': 'best', **overrides}})
    return TrackShots(1, ['front'], config.selection)


def test_ranking_config_is_opt_in_and_rejects_unknown_value():
    assert CaptureConfig().selection.ranking == 'quality'
    config = CaptureConfig().merged({'selection': {'ranking': 'clarity'}})
    assert CaptureConfig().merged(config.to_dict()).selection.ranking == 'clarity'
    with pytest.raises(ValueError, match='selection.ranking'):
        CaptureConfig().merged({'selection': {'ranking': 'sharpest'}})


@pytest.mark.parametrize('stable', [False, True])
@pytest.mark.parametrize('ranking,expected', [('quality', 0), ('clarity', 3)])
def test_pending_and_confirmed_ordering_use_selected_metric(stable, ranking, expected):
    selected = selector(ranking=ranking)
    first, sharper = shot(0, .85, .4), shot(.1, .60, .9)
    selected.offer(first, state(stable), 0)
    assert selected.wants('front', selected.score_for(sharper.quality)) == (ranking == 'clarity')
    events, _ = selected.offer(sharper, state(stable), .1)
    if not stable:
        selected.observe(state(), .2)
    assert selected.best('front').frame_index == expected
    if stable and ranking == 'clarity':
        assert events[0]['event'] == 'view_improved'
        assert events[0]['score'] == .60  # remains the composite, not the ranking score


def test_top_k_temporal_spacing_and_ties_use_clarity():
    selected = selector(top_k=2, min_gap_s=.3)
    for t, score, detail in [(0, .85, .4), (.1, .6, .9), (.5, .75, .7), (.6, .9, .6)]:
        selected.offer(shot(t, score, detail), state(), t)
    assert [s.frame_index for s in selected.slots['front']] == [3, 15]
    assert not selected.wants('front', selected.score_for(report(.99, .7)))


@pytest.mark.parametrize('mode', ['fast', 'hybrid'])
def test_clarity_does_not_trigger_quality_lock_early(mode):
    selected = selector(mode=mode, improve_window_s=1)
    selected.offer(shot(0, .6, .95), state(), 0)
    assert selected.first_good['front'] is None and not selected.locked
    selected.observe(state(), 1.1)
    assert not selected.locked
    selected.observe(state(), 2.01)  # original bounded fallback for usable lower-quality frames
    assert selected.locked == {'front'}


def test_clarity_factors_are_fixed_and_bad_diagnostics_rank_low():
    assert clarity_score({'sharpness': .8, 'eye_detail': .4, 'pose_fit': .01}) == pytest.approx(.8 ** .6 * .4 ** .4)
    for factors in ({}, {'sharpness': float('nan'), 'eye_detail': 1}, {'sharpness': 1, 'eye_detail': 2}):
        assert clarity_score(factors) == 0


def engine_with_reports(monkeypatch, reports):
    config = CaptureConfig().merged({'selection': {'mode': 'best', 'ranking': 'clarity'},
                                     'pose': {'min_dwell_s': 0, 'min_dwell_obs': 1},
                                     'refine': {'gate_min_passes': 1, 'gate_min_ratio': 0}})
    engine = CaptureEngine(config)
    engine.refiner = SimpleNamespace(landmarker=None)
    image, detections = render(center=(160, 160), iod=50, size=(320, 320))
    aligned = SimpleNamespace(image=np.zeros((224, 224, 3), np.uint8))
    pending = iter(reports)
    monkeypatch.setattr(engine.assessor, 'assess', lambda *args, **kwargs: (copy.deepcopy(next(pending)), aligned))
    track = Track(1, detections[0], 0, engine.tracker.kalman)
    return engine, image, track, aligned


@pytest.mark.parametrize('bad', [report(.49, .99, level='low'), report(.9, .99, passed=False, reasons=['too_dark'])])
def test_sharper_but_ineligible_frame_cannot_replace_valid_shot(monkeypatch, bad):
    engine, image, track, _ = engine_with_reports(monkeypatch, [report(.7, .5), bad])
    monkeypatch.setattr(engine, '_unreliable', lambda *args: None)
    for index in range(2):
        engine.frame_times.append(index / 30)
        engine._analyse(image, track, index / 30, index, 0)
    assert engine.contexts[1].shots.best('front').frame_index == 0


def test_clarity_candidate_is_reranked_after_precise_landmarks(monkeypatch):
    engine, image, track, aligned = engine_with_reports(monkeypatch, [report(.8, .6), report(.6, .9), report(.6, .9)])
    engine.refiner.landmarker = object()
    exact = iter([report(.8, .6), report(.6, .5), report(.6, .9)])
    monkeypatch.setattr(engine, '_sharpen', lambda *args: (
        (track.detection, face_geometry(track.detection, engine.config.pose), next(exact), aligned), None))
    checked = []
    monkeypatch.setattr(engine, '_unreliable', lambda *args: checked.append(True))
    for index in range(3):
        engine.frame_times.append(index / 30)
        result = engine._analyse(image, track, index / 30, index, 0)[0]
        assert result.accepted == (index != 1)
    assert len(checked) == 2 and engine.contexts[1].shots.best('front').frame_index == 2


def test_high_clarity_still_requires_landmark_verification(monkeypatch):
    engine, image, track, _ = engine_with_reports(monkeypatch, [report(.9, .99)])
    monkeypatch.setattr(engine, '_unreliable', lambda *args: 'eyes_not_visible')
    engine.frame_times.append(0)
    result = engine._analyse(image, track, 0, 0, 0)[0]
    assert not result.accepted and not engine.contexts[1].shots.captured()
    assert 'eyes_not_visible' in result.quality.reasons
