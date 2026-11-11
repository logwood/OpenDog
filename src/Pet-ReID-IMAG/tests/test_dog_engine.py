"""Behaviour tests for the v1 capture engine (no model files needed)."""
import json
import math

import cv2
import numpy as np
import pytest

from dog_capture.engine import CaptureConfig, CaptureEngine, Detection, ViewSpec
from dog_capture.engine.alignment import PETFACE_DOG_224, align_face, umeyama_similarity
from dog_capture.engine.config import PoseConfig
from dog_capture.engine.engine import OrientationLock, rotate_frame
from dog_capture.engine.pose import ViewClassifier, face_geometry
from dog_capture.engine.quality import QualityAssessor, blur_effect
from dog_capture.engine.schema import ANYFACE5, CORE3, DOGFACE6, box_iou_matrix, nms
from dog_capture.engine.selection import Shot, TrackShots
from dog_capture.engine.tracking import ByteTracker, OneEuroFilter
from dog_capture.engine.alignment import load_template
from dog_engine_synthetic import ScriptedDetector, canonical_points, render


def det(points, box=(0, 0, 400, 400), score=0.9, schema=ANYFACE5, kp_scores=None):
    return Detection(np.asarray(box, np.float32), score, np.asarray(points, np.float32), schema, kp_scores)


def face_points(center=(200, 200), iod=100, yaw=0.0, roll=0.0):
    a = math.radians(roll)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return canonical_points(yaw) * iod @ rot.T + center


def run(engine, frames):
    """frames: iterable of (image, detections) at 30 fps; returns results."""
    out = []
    for i, (image, dets) in enumerate(frames):
        out.append(engine.process(image, i / 30, i, detections=dets))
    return out


def scene(n, **kwargs):
    return [render(**kwargs) for _ in range(n)]


# ---------------------------------------------------------------- config
def test_default_config_round_trips_through_json(tmp_path):
    config = CaptureConfig()
    path = tmp_path / 'c.json'
    path.write_text(json.dumps(config.to_dict()))
    loaded = CaptureConfig.load(path)
    assert loaded.to_dict() == config.to_dict()
    assert [v.name for v in loaded.pose.views] == ['front', 'left', 'right']


def test_config_rejects_typos_and_overlapping_views():
    with pytest.raises(ValueError, match='Unknown key'):
        CaptureConfig.from_dict({'selection': {'mdoe': 'best'}})
    with pytest.raises(ValueError, match='overlap'):
        CaptureConfig.from_dict({'pose': {'views': [
            {'name': 'a', 'center': 0, 'lower': -.2, 'upper': .2},
            {'name': 'b', 'center': .3, 'lower': .1, 'upper': .5}]}})
    with pytest.raises(ValueError, match='mode'):
        CaptureConfig.from_dict({'selection': {'mode': 'fastest'}})


def test_config_merge_keeps_unrelated_defaults():
    merged = CaptureConfig().merged({'selection': {'mode': 'fast'}, 'quality': {'weights': {'motion': 0}}})
    assert merged.selection.mode == 'fast' and merged.selection.top_k == 1
    assert merged.quality.weights['motion'] == 0 and merged.quality.weights['sharpness'] == 3.0


def test_yaml_config(tmp_path):
    yaml = pytest.importorskip('yaml')
    path = tmp_path / 'c.yaml'
    path.write_text(yaml.safe_dump({'pose': {'views': [{'name': 'front', 'center': 0, 'lower': -.1, 'upper': .1}]},
                                    'runtime': {'rotate': 90}}))
    config = CaptureConfig.load(path)
    assert config.runtime.rotate == '90' and len(config.pose.views) == 1


# ---------------------------------------------------------------- schema
def test_detection_validation_and_legacy_fields():
    d = Detection.from_dict({'bbox_xyxy': [10, 10, 90, 90], 'confidence': .8,
                             'landmarks_xy': face_points((50, 50), 30).tolist()})
    assert d.validate() is None and d.schema is ANYFACE5
    assert det(face_points(), box=(10, 10, 5, 50)).validate() == 'empty_box'
    assert det(face_points()[:3], schema=ANYFACE5).validate() == 'keypoint_count_mismatch'
    assert det(np.full((5, 2), np.nan)).validate() == 'nonfinite_detection'


def test_nms_and_iou():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30]], np.float32)
    assert list(nms(boxes, [.9, .8, .7], .5)) == [0, 2]
    assert box_iou_matrix(boxes[:1], boxes[:1])[0, 0] == pytest.approx(1)


# ---------------------------------------------------------------- alignment
def test_umeyama_recovers_similarity():
    src = np.random.default_rng(1).random((5, 2)) * 100
    angle, scale, shift = 0.7, 1.8, np.array([30, -12])
    rot = scale * np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    matrix = umeyama_similarity(src, src @ rot.T + shift)
    assert np.allclose(matrix[:, :2], rot, atol=1e-9) and np.allclose(matrix[:, 2], shift, atol=1e-7)


@pytest.mark.parametrize('scale', [0.4, 1.0, 3.5])
def test_alignment_puts_points_on_template_at_any_scale(scale):
    template = {k: np.asarray(v) for k, v in PETFACE_DOG_224.items()}
    names = list(template)
    angle = math.radians(35)
    rot = scale * np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    src = {k: rot @ (template[k] - 112) + [700, 500] for k in names}
    frame = np.zeros((1200, 1400, 3), np.uint8)
    for k, p in src.items():
        cv2.circle(frame, tuple(np.round(p).astype(int)), max(2, round(6 * scale)), (255, 255, 255), -1)
    aligned = align_face(frame, src)
    assert aligned.residual < 1e-6 and aligned.scale == pytest.approx(1 / scale, rel=1e-6)
    for k in ('left_eye', 'right_eye', 'nose'):
        x, y = np.round(template[k]).astype(int)
        assert aligned.image[y, x].mean() > 100, k  # the marker landed on the template point


# ---------------------------------------------------------------- pose
@pytest.mark.parametrize('roll', [0, 37, 90, 180, 270])
def test_pose_is_roll_invariant_and_canonicalises_eye_order(roll):
    config = PoseConfig()
    g = face_geometry(det(face_points(yaw=.3, roll=roll)), config)
    assert g.valid and g.yaw == pytest.approx(.3, abs=1e-5) and g.pitch == pytest.approx(.5, abs=1e-5)
    p = face_points(yaw=.3, roll=roll)
    p[[0, 1]] = p[[1, 0]]
    p[[3, 4]] = p[[4, 3]]
    swapped = face_geometry(det(p), config)
    assert swapped.swapped and swapped.yaw == pytest.approx(.3, abs=1e-5)


def test_pose_mirror_negates_yaw_and_rejects_impossible_geometry():
    config = PoseConfig()
    p = face_points(yaw=.4)
    mirrored = p.copy()
    mirrored[:, 0] = 400 - mirrored[:, 0]
    assert face_geometry(det(mirrored), config).yaw == pytest.approx(-.4, abs=1e-5)
    flat = face_points()
    flat[2, 1] = flat[0, 1] + 2  # nose on the eye line
    assert face_geometry(det(flat), config).reason == 'implausible_nose_position'
    inconsistent = face_points(yaw=.5)
    inconsistent[3:, 0] -= 80  # mouth swings the other way
    assert face_geometry(det(inconsistent), config).reason == 'inconsistent_landmarks'
    tiny_eyes = face_points(iod=10)
    assert face_geometry(det(tiny_eyes), config).reason == 'eyes_unresolved'


def test_dogface6_chin_cue():
    config = PoseConfig()
    p = face_points(yaw=.35)[:3]
    points = np.vstack([p, [p[2] + [0, 40]], p[0] + [-20, -60], p[1] + [20, -60]])
    g = face_geometry(det(points, schema=DOGFACE6, kp_scores=np.ones(6)), config)
    assert g.valid and g.lower_yaw is not None and g.scores['nose'] == 1


# ---------------------------------------------------------------- view classifier
def geometry(yaw, valid=True):
    g = face_geometry(det(face_points(yaw=yaw)), PoseConfig())
    g.valid = valid
    return g


def test_view_needs_dwell_in_time_and_observations():
    c = ViewClassifier(PoseConfig())
    states = [c.update(i / 30, geometry(.45)) for i in range(6)]
    assert not states[1].stable and states[1].view == 'right'
    assert not states[3].stable  # 4 observations, but only 0.10 s < min_dwell_s
    assert [s.stable for s in states].index(True) == 4


def test_view_hysteresis_and_breaks():
    c = ViewClassifier(PoseConfig())
    for i in range(6):
        c.update(i / 30, geometry(.12))
    s = c.update(6 / 30, geometry(.16))  # outside entry band, inside exit margin
    assert s.view == 'front' and s.raw is None
    assert c.update(7 / 30, geometry(.12, valid=False)).view is None  # occlusion breaks dwell
    for i in range(8, 14):
        c.update(i / 30, geometry(0))
    assert c.update(2.0, geometry(0)).observations == 1  # long gap restarts dwell


def test_one_euro_filter_smooths_noise_and_follows_steps():
    f = OneEuroFilter(1.5, 0.5)
    rng = np.random.default_rng(0)
    out = [float(f(i / 30, .3 + rng.normal(0, .05))) for i in range(60)]
    assert np.std(out[30:]) < .05 * .6
    f.reset()
    values = [float(f(i / 30, 0 if i < 10 else 1)) for i in range(40)]
    assert values[-1] == pytest.approx(1, abs=.02)


# ---------------------------------------------------------------- tracking
def box_det(x, y, w=100, score=.9):
    return det(face_points((x + w / 2, y + w / 2), w * .5), box=(x, y, x + w, y + w), score=score)


def test_bytetrack_keeps_identity_through_skips_and_low_scores():
    t = ByteTracker(CaptureConfig().tracker)
    ids = []
    for i in range(20):
        if i in (8, 9):  # frames skipped by the latest-frame queue
            continue
        score = .3 if i == 12 else .9  # a low-score frame continues, never creates
        active = t.update([box_det(100 + 6 * i, 100, score=score)], i / 30)
        ids.append(active[0].id if active else None)
    assert ids[0] is None  # a track needs two hits
    assert len({i for i in ids[1:]}) == 1


def test_bytetrack_separates_two_dogs_and_expires_lost_tracks():
    t = ByteTracker(CaptureConfig().tracker)
    for i in range(5):
        active = t.update([box_det(50, 50), box_det(400, 60)], i / 30)
    assert sorted(a.id for a in active) == [1, 2]
    for i in range(5, 50):
        active = t.update([box_det(50, 50)], i / 30)
    assert [a.id for a in active] == [1] and [tr.id for tr in t.tracks] == [1]


def test_low_score_detection_never_starts_a_track():
    t = ByteTracker(CaptureConfig().tracker)
    for i in range(5):
        assert t.update([box_det(50, 50, score=.3)], i / 30) == []


# ---------------------------------------------------------------- quality
def assessor():
    c = CaptureConfig()
    return QualityAssessor(c.quality, load_template('petface_dog'))


def assess(image, d, target='front', speed=0.0):
    views = {v.name: v for v in CaptureConfig().pose.views}
    g = face_geometry(d, PoseConfig())
    return assessor().assess(image, d, g, views.get(target), speed)[0]


def test_blur_effect_increases_with_blur():
    image, _ = render()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[200:520, 480:800]
    values = [blur_effect(cv2.GaussianBlur(gray, (0, 0), s) if s else gray) for s in (0, 1, 2, 4)]
    assert values == sorted(values) and values[-1] - values[0] > .2


def test_quality_ranks_sharp_over_blurred_and_rejects_bad_exposure():
    sharp_img, d = render()
    sharp = assess(sharp_img, d[0])
    soft = assess(render(blur=1.2)[0], d[0])
    blurred = assess(render(blur=4)[0], d[0])
    dark = assess(render(gain=.12)[0], d[0])
    bright = assess(render(gain=3.5)[0], d[0])
    assert sharp.passed and sharp.level == 'high'
    assert sharp.score > soft.score > blurred.score
    assert 'blurred' in blurred.reasons
    assert 'too_dark' in dark.reasons and not dark.passed
    assert 'overexposed' in bright.reasons


def test_quality_rejects_small_and_clipped_faces_and_penalises_motion():
    small_img, small = render(iod=14)
    assert 'face_too_small' in assess(small_img, small[0]).reasons
    edge_img, edge = render(center=(30, 360))
    assert 'face_clipped' in assess(edge_img, edge[0]).reasons
    img, d = render()
    assert assess(img, d[0], speed=600).score < assess(img, d[0]).score


def test_keypoint_visibility_gates_front_but_not_side():
    img, d = render(yaw=.45)
    one_eye = Detection(d[0].box, .9, d[0].keypoints, ANYFACE5, np.array([.05, .95, .95, .9, .9]))
    assert 'eyes_or_nose_occluded' not in assess(img, one_eye, 'right').reasons
    img, d = render()
    front = Detection(d[0].box, .9, d[0].keypoints, ANYFACE5, np.array([.05, .95, .95, .9, .9]))
    assert 'eyes_or_nose_occluded' in assess(img, front, 'front').reasons


# ---------------------------------------------------------------- selection
def make_shot(view, score, t, track=1):
    from dog_capture.engine.quality import QualityReport
    q = QualityReport(score, 'high', True)
    return Shot(view, track, int(t * 30), t, score, q, None, None, np.zeros((4, 4, 3), np.uint8), None)


class State:
    def __init__(self, view, stable):
        self.view, self.stable = view, stable


def test_pending_shots_are_promoted_only_when_dwell_confirms():
    s = TrackShots(1, ('front', 'left'), CaptureConfig().selection)
    s.observe(State('front', False), 0)
    s.offer(make_shot('front', .9, 0), State('front', False), 0)
    assert not s.captured()
    events = s.observe(State('front', True), .2)
    assert s.best('front').score == .9 and events[0]['event'] == 'view_captured'
    s.observe(State('left', False), .3)
    s.offer(make_shot('left', .95, .3), State('left', False), .3)
    s.observe(State(None, False), .4)  # visit ended unconfirmed
    s.observe(State('left', True), .5)
    assert not s.best('left')


@pytest.mark.parametrize('mode,expected', [('fast', .7), ('best', .9), ('hybrid', .8)])
def test_selection_modes(mode, expected):
    config = CaptureConfig().merged({'selection': {'mode': mode, 'good_enough': .65, 'improve_window_s': 1.0}}).selection
    s = TrackShots(1, ('front',), config)
    for t, score in ((0, .5), (.1, .7), (.5, .8), (1.5, .9)):
        s.observe(State('front', True), t)
        s.offer(make_shot('front', score, t), State('front', True), t)
    assert s.best('front').score == expected


def test_top_k_keeps_temporally_diverse_shots():
    config = CaptureConfig().merged({'selection': {'top_k': 2, 'min_gap_s': .3, 'mode': 'best'}}).selection
    s = TrackShots(1, ('front',), config)
    for t, score in ((0, .8), (.1, .85), (.5, .7), (.6, .6)):
        s.offer(make_shot('front', score, t), State('front', True), t)
    assert [round(x.time_s, 1) for x in s.slots['front']] == [.1, .5]


# ---------------------------------------------------------------- engine end to end
def turn_sequence(yaws, **kwargs):
    return [render(center=(640 + 2 * i, 360), yaw=float(y), **kwargs) for i, y in enumerate(yaws)]


def test_engine_collects_views_in_any_order_from_one_track():
    engine = CaptureEngine()
    yaws = [.45] * 10 + list(np.linspace(.45, -.5, 10)) + [-.5] * 10 + [0] * 10
    results = run(engine, turn_sequence(yaws))
    best = engine.finalize()
    assert set(best.captured()) == {'front', 'left', 'right'}
    assert results[-1].guidance['code'] == 'done' and not results[-1].missing_views
    shot = best.best('left')
    assert shot.aligned.shape == (224, 224, 3) and shot.frame.shape == (720, 1280, 3)
    assert -0.75 <= shot.geometry.yaw <= -0.25


def test_engine_rejects_blur_and_gives_guidance():
    engine = CaptureEngine()
    results = run(engine, scene(12, blur=5))
    assert engine.finalize() is None
    assert results[-1].guidance['code'] == 'blurred'
    small = CaptureEngine()
    assert run(small, scene(6, iod=14))[-1].guidance['code'] == 'face_too_small'
    empty = CaptureEngine()
    blank = np.full((480, 640, 3), 90, np.uint8)
    assert run(empty, [(blank, [])] * 3)[-1].guidance['code'] == 'no_face'


def test_engine_keeps_two_dogs_apart():
    engine = CaptureEngine()
    frames = [render(center=(350, 360), yaw=0, extra_faces=(((950, 360), 70, .45, 0),)) for _ in range(12)]
    results = run(engine, frames)
    bundles = engine.bundles()
    assert len(bundles) == 2
    assert {tuple(b.captured()) for b in bundles} == {('front',), ('right',)}
    larger = max(results[-1].faces, key=lambda f: f.detection.width)
    assert results[-1].primary_track == larger.track_id  # guidance follows the larger dog
    assert results[-1].captured_views == ['front']


def test_engine_timestamp_policy():
    image, dets = render()
    engine = CaptureEngine()
    assert engine.process(image, 1.0, 0, detections=dets) is not None
    assert engine.process(image, 1.0, 1, detections=dets) is None
    assert engine.dropped['non_increasing_timestamp'] == 1
    strict = CaptureEngine(CaptureConfig().merged({'runtime': {'timestamps': 'raise'}}))
    strict.process(image, 1.0, 0, detections=dets)
    with pytest.raises(ValueError):
        strict.process(image, .5, 1, detections=dets)


def test_engine_accepts_gray_and_bgra_and_rejects_bad_input():
    engine = CaptureEngine()
    image, dets = render()
    assert engine.process(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 0, detections=dets) is not None
    assert engine.process(cv2.cvtColor(image, cv2.COLOR_BGR2BGRA), .1, detections=dets) is not None
    with pytest.raises(ValueError):
        engine.process(image.astype(np.float32), .2, detections=dets)


class FakeGate:
    def __init__(self, boxes):
        self.boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        if frame.shape[:2] != (720, 1280):  # head-crop check: nothing
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
        return self.boxes, np.full(len(self.boxes), .8, np.float32)

    def describe(self):
        return {'kind': 'fake'}


def test_species_gate_rejects_faces_outside_dog_bodies():
    frames = [render(center=(350, 360), extra_faces=(((950, 360), 90, 0, 0),)) for _ in range(8)]
    gate = FakeGate([[200, 150, 520, 700]])
    engine = CaptureEngine(detector=None, body_gate=gate)
    results = run(engine, frames)
    assert all(len(r.faces) <= 1 for r in results) and results[-1].rejected['not_dog'] == 1
    assert len(engine.bundles()) == 1
    assert gate.calls < 8 * 2  # rate limited, not once per face per frame
    nobody = CaptureEngine(body_gate=FakeGate([]))
    assert run(nobody, frames[:4])[-1].guidance['code'] == 'not_dog'


class RotatingDetector:
    """Finds the synthetic face only when the frame is upright (like a real detector)."""

    def __init__(self, engine_ref, upright_rotation, image, detection):
        self.engine_ref, self.rotation = engine_ref, upright_rotation
        self.image, self.detection = image, detection

    def detect(self, frame):
        engine = self.engine_ref[0]
        if engine.orientation.rotation != self.rotation:
            return []
        return [self.detection]

    def describe(self):
        return {'kind': 'rotating'}


def test_orientation_search_locks_the_upright_rotation():
    image, dets = render()
    sideways = np.ascontiguousarray(np.rot90(image, 1))  # content rotated 90 deg CCW
    upright = rotate_frame(sideways, 90)
    assert np.array_equal(upright, image)
    holder = []
    engine = CaptureEngine(detector=RotatingDetector(holder, 90, image, dets[0]))
    holder.append(engine)
    results = [engine.process(sideways, i / 30, i) for i in range(12)]
    assert engine.orientation.locked and engine.orientation.rotation == 90
    assert results[-1].rotation_deg == 90 and results[-1].faces


def test_orientation_lock_uses_keypoint_roll_hint():
    lock = OrientationLock('auto')
    assert not lock.report(True, 0.0, 88)
    assert lock.report(True, 0.04, 91) and lock.rotation == 270
    assert not lock.report(True, 0.08, 2) and not lock.report(True, 0.12, -1) and lock.locked


def test_still_image_mode_picks_best_photo_per_view():
    config = CaptureConfig().merged({'runtime': {'still_images': True}})
    engine = CaptureEngine(config)
    photos = [render(blur=1.5), render(), render(yaw=.45), render(blur=6)]
    for i, (image, dets) in enumerate(photos):
        engine.process(image, float(i), i, detections=dets)
    best = engine.finalize()
    assert best.best('front').frame_index == 1 and best.best('right').frame_index == 2


def test_output_size_is_independent_of_quality_crop():
    engine = CaptureEngine(CaptureConfig().merged({'output': {'aligned_size': 112}}))
    run(engine, scene(10))
    assert engine.finalize().best('front').aligned.shape == (112, 112, 3)


def test_guidance_does_not_flicker_between_frames():
    engine = CaptureEngine()
    sharp, dets = render()
    blurred, _ = render(blur=5)
    codes = []
    for i in range(30):  # one blurred frame in three must not take over the prompt
        image = blurred if i % 3 == 0 else sharp
        codes.append(engine.process(image, i / 30, i, detections=dets).guidance['code'])
    changes = sum(a != b for a, b in zip(codes[5:], codes[6:]))
    assert 'blurred' not in codes[5:] and changes <= 2


def test_malformed_upstream_detections_are_counted_not_fatal():
    engine = CaptureEngine()
    image, dets = render()
    result = engine.process(image, 0.0, 0, detections=[{'box': [1, 2]}, dets[0].to_dict()])
    assert result.rejected == {'malformed_detection': 1}


def test_primary_dog_survives_a_one_frame_miss():
    engine = CaptureEngine()
    pair = render(center=(350, 360), extra_faces=(((950, 360), 60, 0, 0),))
    small_only = render(center=(950, 360), iod=60)
    for i in range(6):
        result = engine.process(pair[0], i / 30, i, detections=pair[1])
    primary = result.primary_track
    result = engine.process(small_only[0], 6 / 30, 6, detections=small_only[1])
    assert result.primary_track == primary  # big dog missed once: keep guiding it
    for i in range(7, 30):
        result = engine.process(small_only[0], i / 30, i, detections=small_only[1])
    assert result.primary_track != primary  # gone for good: switch
