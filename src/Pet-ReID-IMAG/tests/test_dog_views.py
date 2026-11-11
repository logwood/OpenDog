"""Behavior tests for shared-output integration and bounded target selection."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from dog_capture.geometry import Pose
from dog_capture.landmarks import confidence_model, landmark_gate, uncertainty
from dog_capture.observations import FaceObservation, estimate_five_point_pose
from dog_capture.views import TargetViewPipeline, ViewTarget
from test_dog_capture import landmarks, textured_face


def observation(i, yaw=0, track='dog-a', time=None):
    p = [[88, 90], [168, 90], [128 + yaw * 80, 137], [100 + yaw * 40, 177], [156 + yaw * 40, 177]]
    return FaceObservation(i, i / 30 if time is None else time, track, (40, 30, 220, 230), tuple(map(tuple, p)), .95)


def feed(pipeline, start, yaw=0, count=8, track='dog-a', image=None):
    image = textured_face() if image is None else image
    records = []
    for i in range(start, start + count):
        records.append(pipeline.process(image, i / 30, i, observation(i, yaw, track)))
    return records


def test_collects_unordered_views_without_returning_front():
    p = TargetViewPipeline()
    feed(p, 0, .45)
    feed(p, 8, -.45)
    records = feed(p, 16, 0)
    result = p.snapshot()
    assert result['complete'] and set(result['frames']) == {'front', 'left', 'right'}
    assert not records[-1]['tracking']['sequence_complete']
    assert records[-1]['tracking']['coverage_complete']
    assert p.models is None and p.verifications == 0
    assert len(p.frames) == 3


def test_no_single_frame_spikes_or_sparse_confirmation():
    p = TargetViewPipeline()
    for i, yaw in enumerate([0, .45, 0, -.45, 0, .45]):
        p.process(textured_face(), i / 30, i, observation(i, yaw))
    assert not p.snapshot()['frames']
    p = TargetViewPipeline()
    for i in range(10):
        t = i * .3
        p.process(textured_face(), t, i, observation(i, time=t))
    assert not p.snapshot()['frames']


def test_reject_breaks_confirmation_but_keeps_existing_good_view():
    p = TargetViewPipeline()
    feed(p, 0, count=8)
    assert 'front' in p.frames
    feed(p, 8, -.45, count=3)
    p.process(textured_face(), 11 / 30, 11)
    feed(p, 12, -.45, count=3)
    assert set(p.frames) == {'front'}


def test_track_ids_do_not_mix_view_sets():
    p = TargetViewPipeline()
    feed(p, 0, 0)
    feed(p, 8, -.45, track='dog-b')
    assert not p.snapshot()['complete'] and len(p.snapshot()['frames']) == 1
    assert set(p.frames) == {'left'}
    assert p.segment == 1


def test_multiple_dogs_resets_even_brief_interruption():
    p = TargetViewPipeline()
    feed(p, 0)
    p.process(textured_face(), 8 / 30, 8, replace(observation(8), dog_count=2))
    feed(p, 9, -.45)
    assert p.segment == 1 and set(p.frames) == {'left'}
    assert not p.snapshot()['complete']


def test_long_loss_and_frame_size_change_reset_bundle():
    p = TargetViewPipeline()
    feed(p, 0)
    p.process(textured_face(), 1.5, 45, observation(45, -.45))
    assert p.segment == 1 and not p.frames
    p.process(np.zeros((300, 300, 3), np.uint8), 1.6, 48, observation(48))
    assert p.segment == 2


def test_original_winner_not_aliased_and_blur_cannot_replace_it():
    p = TargetViewPipeline()
    image = textured_face()
    feed(p, 0, image=image)
    selected = p.frames['front']['image'].copy()
    image[:] = 0
    feed(p, 8, image=cv2.GaussianBlur(textured_face(), (31, 31), 9))
    assert np.array_equal(p.frames['front']['image'], selected)


@pytest.mark.parametrize('mutation,reason', [
    ({'frame_index': 9}, 'stale_upstream_observation'),
    ({'time_s': float('nan')}, 'stale_upstream_observation'),
    ({'track_id': ''}, 'missing_track_id'),
    ({'confidence': .2}, 'upstream_confidence_low'),
    ({'confidence': float('nan')}, 'upstream_confidence_low'),
    ({'bbox_xyxy': (30, 20, 10, 40)}, 'invalid_upstream_geometry'),
    ({'dog_count': 0}, 'no_dog'),
    ({'landmarks_xy': ((1, 2),)}, 'invalid_upstream_geometry'),
    ({'schema': 'human5'}, 'unsupported_landmark_schema'),
])
def test_invalid_upstream_results_are_rejected(mutation, reason):
    result = replace(observation(0), **mutation).detect(textured_face(), 0, 0)
    assert not result['valid'] and result['reason'] == reason


def test_anyface_adapter_accepts_existing_contract_without_importing_pet_id():
    source = observation(0)
    detection = SimpleNamespace(bbox_xyxy=source.bbox_xyxy, landmarks_xy=source.landmarks_xy, confidence=.95)
    actual = FaceObservation.from_anyface(detection, frame_index=0, time_s=0, track_id='known-dog')
    assert actual.detect(textured_face(), 0, 0)['valid']
    assert actual.track_id == 'known-dog' and actual.schema == 'anyface5'


def test_five_point_direction_mirror_and_translation_scale():
    points = np.array(observation(0, -.45).landmarks_xy)
    a = estimate_five_point_pose(points, (40, 30, 220, 230))
    p = points.copy(); p[:, 0] = 255 - p[:, 0]
    b = estimate_five_point_pose(p, (35, 30, 215, 230))
    c = estimate_five_point_pose(points * 2 + 100, (180, 160, 540, 560))
    assert a.valid and b.valid and c.valid
    assert a.yaw_cue == pytest.approx(-b.yaw_cue)
    assert c.yaw_cue == pytest.approx(a.yaw_cue)


def test_anyface_quarter_turn_is_resolved_from_existing_geometry():
    from dog_capture.landmarks import rotate_normalized
    obs = observation(0, -.35)
    points = np.array(obs.landmarks_xy)
    for k in range(4):
        rotated = rotate_normalized(points / 256, -k) * 256
        x = replace(obs, bbox_xyxy=(0, 0, 256, 256), landmarks_xy=tuple(map(tuple, rotated)))
        result = x.detect(textured_face(), 0, 0)
        assert result['valid'] and result['rotation_quarters'] == k
        assert result['pose'].yaw_cue == pytest.approx(-.35, abs=1e-6)


def test_blurred_neighbour_and_small_exit_jitter_can_confirm_clear_peak():
    p = TargetViewPipeline()
    clear = textured_face()
    blur = cv2.GaussianBlur(clear, (31, 31), 2.2)
    for i, yaw in enumerate([-.09, -.145, -.08]):
        t = i / 15
        p.process(clear if i != 1 else blur, t, i, observation(i, yaw, time=t))
    assert 'front' in p.frames
    chosen = p.frames['front']
    assert chosen['record']['frame_index'] in (0, 2)
    assert chosen['record']['quality']['accepted']
    assert np.array_equal(chosen['image'], clear)


def test_target_band_prevents_extreme_profile_winning():
    p = TargetViewPipeline()
    feed(p, 0, .45)
    before = p.frames['right']['record']['frame_index']
    records = feed(p, 8, .9)
    assert p.frames['right']['record']['frame_index'] == before
    assert all('outside_target_bands' in r['quality']['reasons'] for r in records)


def test_custom_position_subset_and_invalid_config():
    p = TargetViewPipeline(targets=(ViewTarget('front', 0, -.1, .1),))
    feed(p, 0)
    assert p.snapshot()['complete'] and len(p.frames) == 1
    with pytest.raises(ValueError):
        TargetViewPipeline(targets=(ViewTarget('a', 0, -.2, .2), ViewTarget('b', .15, .1, .3)))
    with pytest.raises(ValueError):
        TargetViewPipeline(targets=(ViewTarget('../escape', 0, -.1, .1),))


def test_tracked_peak_requires_verification_of_its_own_pixels():
    seen = []
    class Models:
        def infer(self, frame, t):
            i = round(t * 30)
            return {'valid': True, 'pose': Pose(0, 0, .6, 80, True, 'ok'),
                    'box': (40, 30, 220, 230), 'points': landmarks(),
                    'uncertainty': {'eye_entropy': 2., 'nose_entropy': 2., 'core_entropy': 2.},
                    'inference_mode': 'neural' if i in (0, 6) else 'optical_flow', 'timings_ms': {}}
        def verify_candidate(self, frame, record):
            seen.append((record['frame_index'], frame.copy()))
            return {'valid': False, 'reason': 'uncertain_landmarks', 'inference_mode': 'candidate_neural'}
    p = TargetViewPipeline(Models())
    clear = textured_face()
    for i in range(7):
        frame = clear if i == 3 else cv2.GaussianBlur(clear, (3, 3), 1.1)
        p.process(frame, i / 30, i)
    assert seen and seen[0][0] == 3 and np.array_equal(seen[0][1], clear)
    assert p.verification_rejects == 1 and not p.frames


def test_nonfinite_confidence_is_rejected():
    valid, reason, _ = landmark_gate(landmarks(), (40, 30, 220, 230), 0,
                                     {'eye_entropy': float('nan'), 'nose_entropy': 2., 'core_entropy': 2.})
    assert not valid and reason == 'nonfinite_confidence'


def test_stream_reader_missing_and_duplicate_indices(tmp_path):
    from dog_capture.view_io import ObservationReader
    path = tmp_path / 'input.jsonl'
    path.write_text('\n'.join([json.dumps({'frame_index': i, 'observation': None}) for i in (0, 2, 2)]))
    reader = ObservationReader(path)
    try:
        assert reader.get(0) is None and reader.get(1) is None and reader.get(2) is None
        with pytest.raises(ValueError):
            reader.get(3)
    finally:
        reader.close()


def test_compact_export_preserves_coordinates_and_entropy_on_cpu():
    from dog_capture.models import LiteModel
    base = Path(__file__).resolve().parents[3] / 'models/pretrained/dog_face_landmarks_v1/dog_face_landmarks_full.tflite'
    if not base.exists():
        pytest.skip('Pinned models not downloaded')
    original = LiteModel(content=confidence_model(base))
    compact = LiteModel(content=confidence_model(base, compact=True))
    for x in (np.zeros((1, 384, 384, 3), np.float32), np.random.default_rng(7).random((1, 384, 384, 3), dtype=np.float32)):
        a, full = original(x)
        b, entropy = compact(x)
        np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-6)
        assert sum(t.nbytes for t in (b, entropy)) == 552
        assert np.isfinite(entropy).all()
        old, new = uncertainty(full), uncertainty(entropy)
        for key in old:
            assert abs(old[key] - new[key]) < .014
