"""Behavioral regressions for capture stability and face-only frame ranking."""
import json
from pathlib import Path
import threading

import cv2
import numpy as np
import pytest

from dog_capture.geometry import EYE_A, EYE_B, NOSE, Pose, TurnTracker, estimate_pose
from dog_capture.models import letterbox, undo_letterbox
from dog_capture.quality import BestFrames, assess_quality
from dog_capture.landmarks import (confidence_model, eye_evidence, landmark_gate,
                                    pose_in_crop, rotate_normalized, uncertainty)


def landmarks(yaw=0.0):
    points = np.full((46, 2), [128, 130], np.float32)
    eye = np.array([[-9, 0], [0, -5], [9, 0], [0, 5]])
    points[EYE_A] = [88, 90] + eye
    points[EYE_B] = [168, 90] + eye
    a = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    points[NOSE] = np.stack([128 + yaw * 80 + 12 * np.cos(a), 137 + 9 * np.sin(a)], 1)
    a = np.linspace(0, np.pi, 14)
    points[32:] = np.stack([128 + 58 * np.cos(a), 155 + 49 * np.sin(a)], 1)
    return points


def textured_face():
    y, x = np.mgrid[:256, :256]
    image = 70 + ((x // 6 + y // 6) % 2) * 100
    return np.repeat(image[..., None], 3, axis=2).astype(np.uint8)


def pose(yaw=0):
    return Pose(yaw_cue=yaw, valid=True, reason="ok")


def feed(tracker, start, yaw, duration=0.6, valid=True):
    output = None
    for t in start + np.arange(round(duration / .04)) * .04:
        output = tracker.update(float(t), pose(yaw), valid)
    return output


def test_mirror_reverses_direction_and_not_strength():
    p = landmarks(0.45)
    original = estimate_pose(p, (40, 30, 220, 230), (256, 256))
    p[:, 0] = 255 - p[:, 0]
    mirrored = estimate_pose(p, (35, 30, 215, 230), (256, 256))
    assert original.valid and mirrored.valid
    assert mirrored.yaw_cue == pytest.approx(-original.yaw_cue, abs=1e-6)


def test_translation_scale_and_roll_do_not_create_yaw():
    p = landmarks()
    angle = np.deg2rad(20)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    transformed = (p - 128) @ rotation.T * 1.8 + [300, 250]
    estimated = estimate_pose(transformed, (100, 50, 500, 450), (600, 600))
    assert estimated.valid
    assert estimated.yaw_cue == pytest.approx(0, abs=1e-5)
    assert estimated.roll_deg == pytest.approx(20, abs=1e-5)


def test_single_spike_and_boundary_jitter_cannot_complete_turn():
    tracker = TurnTracker()
    feed(tracker, 0, 0)
    tracker.update(0.60, pose(-0.9), True)
    feed(tracker, 0.64, 0)
    assert tracker.index == 1
    for i in range(40):
        tracker.update(1.3 + i * .04, pose(-.26 if i % 2 else -.05), True)
    assert tracker.index == 1


def test_ordered_turn_sequence_and_wrong_direction():
    tracker = TurnTracker()
    feed(tracker, 0, 0)
    feed(tracker, .6, .55)
    assert tracker.index == 1
    feed(tracker, 1.2, -.55)
    feed(tracker, 1.8, 0)
    feed(tracker, 2.4, .55)
    final = feed(tracker, 3.0, 0)
    assert final["sequence_complete"]
    assert [e["pose"] for e in tracker.events if e["event"] == "step"] == ["front", "left", "front", "right", "front"]


def test_occlusion_breaks_dwell_and_long_gap_resets_sequence():
    tracker = TurnTracker(smoothing_seconds=0)
    feed(tracker, 0, 0)
    feed(tracker, .6, -.6, duration=.2)
    tracker.update(.8, pose(-.6), False)
    after = tracker.update(.84, pose(-.6), True)
    assert after["stable_state"] == "unknown" and tracker.index == 1
    feed(tracker, 1.0, 0, duration=1.0, valid=False)
    assert tracker.segment == 1 and tracker.index == 0


def test_sparse_frames_cannot_fake_continuous_hold():
    tracker = TurnTracker()
    for t in [0, .4, .8, 1.2]:
        out = tracker.update(t, pose(0), True)
        assert out["stable_state"] == "unknown"


def test_monotonic_timestamps():
    tracker = TurnTracker()
    tracker.update(1.0, pose(), True)
    with pytest.raises(ValueError):
        tracker.update(1.0, pose(), True)


def test_clear_face_beats_blurred_face():
    image, points = textured_face(), landmarks()
    p = estimate_pose(points, (40, 30, 220, 230), image.shape)
    sharp = assess_quality(image, points, (40, 30, 220, 230), p)
    blurred = assess_quality(cv2.GaussianBlur(image, (21, 21), 5), points, (40, 30, 220, 230), p)
    assert sharp.accepted
    assert sharp.score > blurred.score and sharp.laplacian > blurred.laplacian * 3
    json.dumps(sharp.to_dict(), allow_nan=False)


def test_background_detail_does_not_inflate_face_quality():
    image, points = textured_face(), landmarks()
    blurred = cv2.GaussianBlur(image, (21, 21), 5)
    with_background = blurred.copy()
    with_background[:, :55] = image[:, :55]
    with_background[:, 201:] = image[:, 201:]
    p = estimate_pose(points, (40, 30, 220, 230), image.shape)
    q1 = assess_quality(blurred, points, (40, 30, 220, 230), p)
    q2 = assess_quality(with_background, points, (40, 30, 220, 230), p)
    assert q1.score == pytest.approx(q2.score, abs=1e-6)


@pytest.mark.parametrize("value", [0, 255])
def test_flat_dark_and_overexposed_frames_rejected(value):
    points = landmarks()
    image = np.full((256, 256, 3), value, np.uint8)
    p = estimate_pose(points, (40, 30, 220, 230), image.shape)
    assert not assess_quality(image, points, (40, 30, 220, 230), p).accepted


def test_best_frame_keeps_unmodified_pixels_and_rejects_bad_candidates():
    selector = BestFrames()
    frame = textured_face()
    def record(score, accepted=True, segment=0):
        return {"tracking": {"stable_state": "front", "segment": segment},
                "pose": {"yaw_cue": 0, "roll_deg": 0}, "quality": {"accepted": accepted, "score": score}}
    assert selector.consider(frame, record(5))
    assert not selector.consider(frame, record(100, False))
    assert not selector.consider(frame, record(4))
    assert np.array_equal(selector.frames["front"]["image"], frame)
    frame[:] = 0
    assert selector.frames["front"]["image"].max() > 0
    selector.consider(frame, record(0, False, 1))
    assert selector.frames == {}


def test_letterbox_roundtrip_for_portrait_and_odd_sizes():
    for shape in [(719, 1281, 3), (847, 481, 3)]:
        _, mapping = letterbox(np.zeros(shape, np.uint8), 224)
        sx, sy, px, py = mapping
        original = [30, 40, 300, 600]
        scaled = [(original[0]*sx+px)/224, (original[1]*sy+py)/224,
                  (original[2]*sx+px)/224, (original[3]*sy+py)/224]
        assert undo_letterbox(scaled, mapping, 224) == pytest.approx(original)


def test_quarter_rotation_roundtrip_in_non_square_crop():
    points = landmarks()
    base = pose_in_crop(points, (0, 0, 256, 256), 0)
    for k in range(4):
        normalized = rotate_normalized(points / 256, -k)
        restored = rotate_normalized(normalized, k) * 256
        assert restored == pytest.approx(points, abs=1e-4)
        pose_rotated = pose_in_crop(normalized * 256, (0, 0, 256, 256), k)
        assert pose_rotated.yaw_cue == pytest.approx(base.yaw_cue, abs=1e-6)
    points2 = np.array([[.1, .2], [.7, .8]])
    assert rotate_normalized(rotate_normalized(points2, 1), 3) == pytest.approx(points2)


def test_diffuse_heatmaps_are_not_visibility_evidence():
    diffuse = np.full((1, 46, 192 * 192), 1 / (192 * 192), dtype=np.float32)
    concentrated = np.zeros_like(diffuse)
    concentrated[:, :, 1200] = 1
    p, box = landmarks(), (0, 0, 256, 256)
    assert not landmark_gate(p, box, 0, uncertainty(diffuse))[0]
    assert landmark_gate(p, box, 0, uncertainty(concentrated))[0]


def test_bright_fur_cannot_supply_two_eyes():
    frame = np.full((256, 256, 3), 190, np.uint8)
    p = landmarks()
    cv2.circle(frame, (88, 90), 12, (10, 10, 10), -1)
    assert not eye_evidence(frame, p)['accepted']
    cv2.circle(frame, (168, 90), 12, (10, 10, 10), -1)
    assert eye_evidence(frame, p)['accepted']


def test_best_frame_includes_clear_start_of_confirmed_dwell():
    selector = BestFrames()
    image = textured_face()
    def record(score, stable):
        return {'tracking': {'state': 'front', 'stable_state': stable, 'segment': 0},
                'pose': {'yaw_cue': 0, 'roll_deg': 0}, 'quality': {'score': score, 'accepted': True}}
    assert not selector.consider(image, record(9, 'unknown'))
    assert selector.consider(image, record(5, 'front'))
    assert selector.frames['front']['record']['quality']['score'] == 9


def test_two_neural_observations_confirm_photo_but_not_action():
    selector, tracker = BestFrames(), TurnTracker()
    image = textured_face()
    for i, timestamp in enumerate([0, .3]):
        tracking = tracker.update(timestamp, pose(), True)
        record = {'time_s': timestamp, 'tracking': tracking, 'inference_mode': 'neural',
                  'uncertainty': {'eye_entropy': 2.5, 'nose_entropy': 2.7},
                  'pose': {'yaw_cue': 0, 'roll_deg': 0}, 'quality': {'score': 5-i, 'accepted': True}}
        selector.consider(image, record)
    assert tracker.index == 0
    assert selector.frames['front']['record']['selection_basis'] == 'two_neural_front_observations'
    assert selector.frames['front']['record']['quality']['score'] == 5


def test_latest_frame_worker_replaces_backlog_and_preserves_pixels():
    from dog_capture.realtime import LatestFrameWorker
    entered, release = threading.Event(), threading.Event()
    seen = []
    class SlowPipeline:
        def process(self, image, timestamp, index):
            if index == 0:
                entered.set()
                assert release.wait(3)
            seen.append((index, int(image[0, 0, 0])))
            return {'frame_index': index, 'inference_mode': 'mock', 'tracking': {'stable_state': 'unknown'}}
    worker = LatestFrameWorker(SlowPipeline())
    image = np.full((8, 8, 3), 7, np.uint8)
    worker.submit(image, 0, 0)
    assert entered.wait(3)
    for i in range(1, 30):
        worker.submit(image, i / 30, i)
    image[:] = 99
    assert worker.pending.qsize() == 1
    release.set()
    worker.close()
    assert seen == [(0, 7), (29, 7)]
    assert worker.processed + worker.dropped == worker.submitted


def test_confidence_export_preserves_pretrained_coordinates():
    from dog_capture.models import LiteModel
    base = Path(__file__).resolve().parents[3] / 'models/pretrained/dog_face_landmarks_v1/dog_face_landmarks_full.tflite'
    if not base.exists():
        pytest.skip('Pinned models not downloaded')
    original = LiteModel(base)
    augmented = LiteModel(content=confidence_model(base))
    image = np.random.default_rng(31).random((1, 384, 384, 3), dtype=np.float32)
    expected = original(image)[0]
    actual, heatmap = augmented(image)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert heatmap.shape == (1, 46, 36864)
    np.testing.assert_allclose(heatmap.sum(-1), 1, atol=1e-5)


def test_bounded_transition_queue_keeps_chronology_and_limits_backlog():
    from dog_capture.realtime import LatestFrameWorker
    entered, release = threading.Event(), threading.Event()
    seen = []
    class Pipeline:
        def process(self, image, timestamp, index):
            if index == 0:
                entered.set()
                assert release.wait(3)
            seen.append((index, timestamp, int(image[0, 0, 0])))
            return {'frame_index': index, 'inference_mode': 'mock', 'tracking': {'stable_state': 'unknown'}}
    worker = LatestFrameWorker(Pipeline(), queue_capacity=3)
    image = np.full((8, 8, 3), 7, np.uint8)
    worker.submit(image, 0, 0)
    assert entered.wait(3)
    try:
        for index in range(1, 5):
            worker.submit(image, index / 30, index)
        image[:] = 99
        assert worker.pending.qsize() == 3
        assert worker.dropped == 1
    finally:
        release.set()
        worker.close()
    assert seen == [(0, 0, 7), (2, 2/30, 7), (3, 3/30, 7), (4, 4/30, 7)]
    assert worker.processed + worker.dropped == worker.submitted
    with pytest.raises(ValueError):
        LatestFrameWorker(Pipeline(), queue_capacity=0)
