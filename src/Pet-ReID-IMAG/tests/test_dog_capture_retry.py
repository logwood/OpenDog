"""Failed candidates must not block a new visit or cause unbounded retry storms."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from dog_capture.engine import CaptureConfig, CaptureEngine
import dog_capture.engine.engine as engine_module
from dog_capture.engine.pose import face_geometry
from dog_capture.engine.quality import QualityReport
from dog_capture.engine.retry import CandidateRetry
from dog_capture.engine.tracking import Track
from dog_engine_synthetic import render


def failed():
    gate = CandidateRetry(.25)
    gate.observe('front')
    assert gate.begin(0)
    gate.finish(0, 'view_unconfirmed', False, had_pending=True)
    return gate


def test_same_visit_keeps_original_backoff():
    gate = failed()
    for t in (.01, .10, .249):
        gate.observe('front')
        assert not gate.begin(t)
    assert gate.begin(.25)


@pytest.mark.parametrize('next_view', ['front', 'left'])
def test_new_visit_has_one_recovery_not_one_per_pose_flicker(next_view):
    gate = failed()
    gate.observe(None)
    gate.observe(next_view)
    assert gate.begin(.03)
    gate.finish(.03, 'eyes_not_visible', False, had_pending=True)
    for i in range(1, 20):
        gate.observe(None)
        gate.observe('right' if i % 2 else 'left')
        assert not gate.begin(.03 + i * .01)
    assert gate.begin(.28)


@pytest.mark.parametrize('problem', [None, 'landmarks_unstable'])
def test_successful_recovery_can_collect_temporal_evidence_immediately(problem):
    gate = failed()
    gate.observe(None)
    gate.observe('front')
    assert gate.begin(.03)
    gate.finish(.03, problem, True)
    assert gate.begin(.04) and not gate.recovery_used


def test_unqualified_frames_do_not_spend_recovery_and_inconclusive_check_keeps_wait():
    gate = failed()
    gate.observe(None)
    assert not gate.begin(.01)
    gate.observe('left')  # no begin() until quality/ranking admits a candidate
    gate.observe('right')
    assert not gate.recovery_used and gate.begin(.02)
    gate.finish(.02, None, False)  # localization alone did not verify the candidate
    assert not gate.begin(.03)


def test_retry_state_is_per_track_and_zero_period_disables_wait():
    first = failed()
    second = CandidateRetry(.25)
    second.observe('front')
    assert second.begin(.01) and not first.begin(.01)
    zero = CandidateRetry(0)
    zero.observe('front')
    for t in (0, .01, .02):
        assert zero.begin(t)
        zero.finish(t, 'landmarks_unreliable', False)


def test_initial_acquisition_failure_does_not_earn_recovery():
    gate = CandidateRetry(.25)
    gate.observe('front')
    assert gate.begin(0)
    gate.finish(0, 'view_unconfirmed', False)
    for view in (None, 'left', None, 'right'):
        gate.observe(view)
        assert not gate.begin(.1)
    assert gate.begin(.25)


def setup_engine(monkeypatch):
    config = CaptureConfig().merged({'selection': {'mode': 'best'},
        'pose': {'min_dwell_s': 0, 'min_dwell_obs': 1},
        'refine': {'gate_min_passes': 1, 'gate_min_ratio': .35}})
    engine = CaptureEngine(config)
    engine.refiner = SimpleNamespace(landmarker=object())
    image, detections = render(center=(160, 160), iod=50, size=(320, 320))
    detections[0].source = 'test'
    track = Track(1, detections[0], 0, engine.tracker.kalman)
    geometry = face_geometry(track.detection, config.pose)
    q = QualityReport(score=.9, level='high', passed=True, factors={'sharpness': .8, 'eye_detail': .8})
    aligned = SimpleNamespace(image=np.zeros((224, 224, 3), np.uint8))
    monkeypatch.setattr(engine.assessor, 'assess', lambda *a, **kw: (copy.deepcopy(q), aligned))
    return engine, image, track, geometry, q, aligned


def test_new_visit_recovers_after_failure_but_still_requires_verification(monkeypatch):
    engine, image, track, geometry, q, aligned = setup_engine(monkeypatch)
    engine.config.pose.min_dwell_s = .12
    q.score = .8  # the next coarse score is .9, so it attempts to improve this pending shot
    sequence = iter([geometry, geometry, replace(geometry, valid=False, reason='inconsistent_landmarks'), geometry])
    monkeypatch.setattr(engine_module, 'face_geometry', lambda *a: next(sequence))
    outcomes = iter([((track.detection, geometry, q, aligned), None), (None, 'view_unconfirmed'),
                     ((track.detection, geometry, q, aligned), None)])
    coarse = replace(q, score=.9)
    monkeypatch.setattr(engine.assessor, 'assess', lambda *a, **kw: (copy.deepcopy(coarse), aligned))
    monkeypatch.setattr(engine, '_sharpen', lambda *a: next(outcomes))
    checks = []
    monkeypatch.setattr(engine, '_unreliable', lambda *a: checks.append(True))
    accepted = []
    for i in range(4):
        engine.frame_times.append(i / 30)
        accepted.append(engine._analyse(image, track, i / 30, i, 0)[0].accepted)
    assert accepted == [True, False, False, True]
    assert len(checks) == 2 and engine.contexts[1].shots.pending['front'][0].frame_index == 3


def test_repeated_invalid_visits_do_not_trigger_a_detector_storm(monkeypatch):
    engine, image, track, geometry, _, _ = setup_engine(monkeypatch)
    sequence = iter([geometry, replace(geometry, valid=False, reason='inconsistent_landmarks')] * 8)
    monkeypatch.setattr(engine_module, 'face_geometry', lambda *a: next(sequence))
    calls = []
    def fail(*args):
        calls.append(True)
        return None, 'view_unconfirmed'
    monkeypatch.setattr(engine, '_sharpen', fail)
    for i in range(16):
        engine.frame_times.append(i / 60)
        assert not engine._analyse(image, track, i / 60, i, 0)[0].accepted
    assert len(calls) == 1  # no prior verified pending shot: original backoff remains


@pytest.mark.parametrize('has_face', [True, False])
def test_benchmark_snapshots_completion_cost_before_remaining_video(tmp_path, has_face):
    import importlib.util
    from pathlib import Path
    import cv2
    from dog_engine_synthetic import ScriptedDetector
    tool = Path(__file__).resolve().parents[1] / 'tools/benchmark_dog_capture_v1.py'
    spec = importlib.util.spec_from_file_location('capture_milestone_test', tool)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    frame, detections = render(center=(160, 160), iod=50, size=(320, 320))
    video = tmp_path / 'face.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 30, (320, 320))
    assert writer.isOpened()
    for _ in range(12): writer.write(frame)
    writer.release()
    cfg = CaptureConfig().merged({'refine': {'enabled': False},
        'selection': {'mode': 'fast', 'good_enough': .5},
        'pose': {'views': [{'name': 'front', 'center': 0, 'lower': -.13, 'upper': .13}],
                 'min_dwell_s': 0, 'min_dwell_obs': 1}})
    detector = ScriptedDetector([copy.deepcopy(detections) if has_face else [] for _ in range(12)])
    result = benchmark.video_benchmark(cfg, detector, None, video, False, tmp_path / 'output', 1, 12)
    assert result['frames_processed'] == result['model_calls']['face'] == 12
    milestone = result['first_complete']
    if has_face:
        assert milestone and milestone['frames_processed'] < result['frames_processed']
        assert milestone['model_calls']['face'] == milestone['frames_processed'] == milestone['frame_index'] + 1
        assert milestone['selected_frames']['front'] == [result['selection']['front'][0]['frame_index']]
        assert milestone['video_time_s'] == pytest.approx(milestone['frame_index'] / 30)
    else:
        assert milestone is None
