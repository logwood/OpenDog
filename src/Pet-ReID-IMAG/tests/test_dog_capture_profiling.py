import cv2
import pytest

from dog_capture.engine import CaptureEngine
from dog_capture.engine.config import CaptureConfig
from dog_capture.engine.io import LatestFrameRunner
from dog_capture.engine.profiling import CaptureProfiler
from dog_engine_synthetic import BlobDetector, render


def test_profile_keeps_results_and_reconciles_nested_wall_time():
    original = CaptureEngine(CaptureConfig(), BlobDetector())
    traced = CaptureEngine(CaptureConfig(), BlobDetector())
    resize = cv2.resize
    with CaptureProfiler(traced) as profile:
        for index in range(8):
            frame, _ = render()
            want = original.process(frame, index / 30, index).to_dict()
            got = traced.process(frame, index / 30, index).to_dict()
            want.pop('timings_ms'); got.pop('timings_ms')
            assert want == got
    assert cv2.resize is resize
    assert 'process' not in vars(traced)
    result = profile.summary()
    assert result['frames'] == 8
    assert sum(row['total_ms'] for row in result['all']['components']) == pytest.approx(result['all']['total_ms'])
    assert all(span[3] >= 0 for frame in profile.frames for span in frame['spans'])
    assert profile.chrome_trace()['traceEvents']


def test_profile_restores_hooks_after_inference_exception():
    detector = BlobDetector()
    def failed(*args, **kwargs):
        raise ValueError('inference failed')
    detector.detect = failed
    engine = CaptureEngine(CaptureConfig(), detector)
    warp = cv2.warpAffine
    with pytest.raises(ValueError, match='inference failed'):
        with CaptureProfiler(engine):
            engine.process(render()[0], 0, 0)
    assert cv2.warpAffine is warp and detector.detect is failed
    assert 'process' not in vars(engine)


def test_profile_records_worker_thread_frames_and_excludes_other_calls():
    engine = CaptureEngine(CaptureConfig(), BlobDetector())
    frame = render()[0]
    with CaptureProfiler(engine) as profile:
        runner = LatestFrameRunner(engine)
        runner.submit(frame, 0, 0)
        cv2.resize(frame, (32, 32))  # producer-side work is not a child of worker inference
        runner.close()
    assert len(profile.frames) == runner.processed == 1
    profile.summary()  # verifies every exclusive span reconciles


def test_comparison_preserves_diagnostics_already_present_in_both_versions():
    import copy
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'tools/compare_dog_capture_runtime.py'
    spec = importlib.util.spec_from_file_location('capture_comparison_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old = {'faces': [{'detection': {'verify': {'score': .4, 'eyes': [.9, .9], 'agreement': 0,
                                             'mirror': None, 'mirror_skipped': 'score'}}}]}
    new = copy.deepcopy(old)
    assert module.normalize_diagnostics(old, new, CaptureConfig().refine) == 0
    assert old == new and new['faces'][0]['detection']['verify']['mirror_skipped'] == 'score'
    new['faces'][0]['detection']['verify']['mirror_skipped'] = 'eyes'
    module.normalize_diagnostics(old, new, CaptureConfig().refine)
    assert old != new  # a different reason must not be masked
    old['faces'][0]['detection']['verify'].pop('mirror_skipped')
    old['faces'][0]['detection']['verify']['mirror'] = .1
    new['faces'][0]['detection']['verify']['mirror_skipped'] = 'score'
    assert module.normalize_diagnostics(old, new, CaptureConfig().refine) == 1
    assert old == new  # the original pre-short-circuit reference is still supported
