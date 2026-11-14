"""Evidence belongs to a track and one coordinate system, not a nearby face."""
import copy
import math
import time
from types import SimpleNamespace

import numpy as np
import pytest

from dog_capture.engine import CaptureConfig, CaptureEngine
from dog_capture.engine.evidence import TemporalEvidence
from dog_capture.engine.quality import QualityReport
from dog_capture.engine.tracking import Track
from dog_engine_synthetic import ScriptedDetector, render


def gate_engine(monkeypatch):
    config = CaptureConfig().merged({'pose': {'min_dwell_s': 0, 'min_dwell_obs': 1},
                                     'selection': {'mode': 'best'},
                                     'refine': {'gate_min_passes': 2, 'gate_min_ratio': 0}})
    engine = CaptureEngine(config)
    engine.refiner = SimpleNamespace(landmarker=None)
    # The checks really pass; only the ownership/counting of temporal evidence
    # is under test. Avoid making pixel quality or model fixtures the oracle.
    monkeypatch.setattr(engine, '_unreliable', lambda *args: None)
    image = np.zeros((224, 224, 3), np.uint8)
    monkeypatch.setattr(engine.assessor, 'assess', lambda *args, **kwargs: (
        QualityReport(score=.9, level='high', passed=True), SimpleNamespace(image=image)))
    return engine


def tracks_for(engine):
    image, detections = render(center=(160, 160), iod=50, size=(320, 320))
    other = copy.deepcopy(detections[0])
    other.box += [40, 0, 40, 0]
    other.keypoints += [40, 0]
    return image, [Track(1, detections[0], 0, engine.tracker.kalman), Track(2, other, 0, engine.tracker.kalman)]


@pytest.mark.parametrize('reverse', [False, True])
def test_nearby_dogs_cannot_confirm_each_other_in_one_frame(monkeypatch, reverse):
    engine = gate_engine(monkeypatch)
    frame, tracks = tracks_for(engine)
    if reverse:
        tracks.reverse()
    for index in range(2):
        timestamp = index / 30
        engine.frame_times.append(timestamp)
        results = [engine._analyse(frame, track, timestamp, index, 0)[0] for track in tracks]
        assert [r.accepted for r in results] == [index == 1, index == 1]


def test_new_track_at_same_place_must_build_its_own_evidence(monkeypatch):
    engine = gate_engine(monkeypatch)
    frame, tracks = tracks_for(engine)
    first = tracks[0]
    for i in range(2):
        engine.frame_times.append(i / 30)
        engine._analyse(frame, first, i / 30, i, 0)
    engine._archive(first.id, 2 / 30)
    replacement = Track(3, copy.deepcopy(first.detection), 2 / 30, engine.tracker.kalman)
    engine.frame_times.append(2 / 30)
    result, _ = engine._analyse(frame, replacement, 2 / 30, 2, 0)
    assert not result.accepted
    assert 'landmarks_unstable' in result.quality.reasons


def test_rotation_switch_does_not_seed_tracker_in_old_coordinates():
    image, detections = render(roll=90)
    engine = CaptureEngine(detector=ScriptedDetector([copy.deepcopy(detections)] * 3))
    engine.process(image, 0, 0)
    engine.roi_checked = 0
    result = engine.process(image, 1 / 30, 1)
    assert engine.orientation.rotation == 270
    assert result.rotation_deg == 0 and result.faces == []  # this frame was analysed at 0 degrees
    assert not engine.tracker.tracks
    assert not engine.frame_times
    assert engine.full_detected == -math.inf and engine.roi_checked == -math.inf


def test_probe_rotation_clears_coordinate_bound_caches():
    _, detections = render()
    engine = CaptureEngine()
    engine._context(7)
    engine.primary_id = 7
    engine.body_boxes = np.array([[0, 0, 100, 100]], np.float32)
    engine.body_scores = np.array([.9], np.float32)
    engine.body_time = engine.body_checked = engine.roi_checked = engine.full_detected = 0
    engine.frame_times.extend([0, .1])
    assert not engine.orientation.report(90, True, True, .1, 0)
    engine._probe(detections, 90, .2, 2, {}, {}, time.perf_counter())
    assert engine.orientation.rotation == 90
    assert not engine.contexts and engine.primary_id is None
    assert not engine.frame_times and len(engine.body_boxes) == len(engine.body_scores) == 0
    assert engine.full_detected == engine.roi_checked == engine.body_checked == -math.inf


def test_evidence_records_one_vote_per_frame_and_rejects_time_reversal():
    evidence = TemporalEvidence()
    config = CaptureConfig().refine
    for _ in range(4):
        evidence.record(0, (0, 0, 100), 'pass')
    assert evidence.reject_reason((0, 0, 100), 2, config) == 'landmarks_unstable'
    evidence.record(1, (0, 0, 100), 'pass')
    assert evidence.reject_reason((0, 0, 100), 3, config) is None
    evidence.record(1, (0, 0, 100), 'fail')
    assert evidence.reject_reason((0, 0, 100), 3, config) == 'landmarks_unstable'
    with pytest.raises(ValueError, match='decrease'):
        evidence.record(0, (0, 0, 100), 'pass')


@pytest.mark.parametrize('status,expected', [('pass', None), ('present', None), ('fail', 'landmarks_unstable')])
def test_evidence_keeps_presence_distinct_from_a_failed_check(status, expected):
    config = CaptureConfig().merged({'refine': {'gate_min_passes': 2, 'gate_min_ratio': .75}}).refine
    evidence = TemporalEvidence()
    evidence.record(0, (0, 0, 100), 'pass')
    evidence.record(1, (0, 0, 100), status)
    assert evidence.reject_reason((0, 0, 100), 3, config) == expected


def test_evidence_keeps_spatial_boundary_and_missing_frame_penalty():
    config = CaptureConfig().refine
    evidence = TemporalEvidence()
    evidence.record(0, (0, 0, 100), 'pass')
    evidence.record(1, (50, 0, 100), 'pass')
    assert evidence.reject_reason((50, 0, 100), 3, config) is None
    assert evidence.reject_reason((50.001, 0, 100), 3, config) == 'landmarks_unstable'
    assert evidence.reject_reason((50, 0, 100), 20, config) == 'landmarks_unstable'


def test_evidence_releases_old_observations_at_processed_frame_window():
    evidence = TemporalEvidence()
    for i in range(1000):
        evidence.trim(i - 4)
        evidence.record(i, (0, 0, 100), 'present')
        assert len(evidence.observations) <= 5
    assert evidence.observations[0][0] == 995


def test_rotation_archives_kept_photos_and_starts_new_evidence(monkeypatch):
    engine = gate_engine(monkeypatch)
    frame, tracks = tracks_for(engine)
    for i in range(2):
        engine.frame_times.append(i / 30)
        engine._analyse(frame, tracks[0], i / 30, i, 0)
    before = engine.best_bundle().best('front')
    engine.refiner.reset = lambda: None
    engine._reset_for_rotation(2 / 30)
    assert not engine.contexts
    assert engine.best_bundle().best('front') is before
