"""Pixel parity, conditional-head boundaries and strict comparison safeguards."""
import copy
import importlib.util
import json
from pathlib import Path
import threading

import cv2
import numpy as np
import pytest

from dog_capture.engine import resampling


def tool(name):
    path = Path(__file__).resolve().parents[1] / 'tools' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('shape,size', [((1280, 720), (76, 134)), ((720, 1280), (117, 66)),
                                       ((1280, 720), (61, 109)), ((720, 1280), (320, 180)),
                                       ((546, 547), (316, 316)), ((1200, 800), (77, 91)),
                                       ((720, 1260), (128, 73)), ((909, 720), (256, 323)),
                                       ((802, 706), (112, 127)), ((1280, 720), (50, 89))])
def test_area_strips_preserve_pixels_and_strided_roi(shape, size):
    rng = np.random.default_rng(711)
    h, w = shape
    parent = rng.integers(0, 256, (h, w + 17, 3), np.uint8)
    image = parent[:, 7:7 + w]
    before = image.copy()
    actual = resampling.resize_area(image, size)
    expected = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(image, before)


def test_area_sampling_across_integer_fractional_and_edge_patterns():
    rng = np.random.default_rng(482)
    accepted = 0
    for height in (600, 720, 960, 1080, 1280):
        frame = rng.integers(0, 256, (height, 720, 3), np.uint8)
        frame[::7] = 255
        frame[1::7] = 0
        for out in range(63, 144, 5):
            size = (max(1, round(out * 720 / height)), out)
            accepted += resampling._split(height, out) is not None
            np.testing.assert_array_equal(resampling.resize_area(frame, size),
                                          cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
    assert accepted > 20  # exercise the new path, not just the fallback


def test_area_column_sampling_preserves_horizontal_rounding():
    rng = np.random.default_rng(385)
    accepted = 0
    for width in (600, 720, 960, 1080, 1280):
        # Coprime heights force the column path. Keep a parent row stride.
        parent = rng.integers(0, 256, (719, width + 13, 3), np.uint8)
        frame = parent[:, :width]
        frame[:, ::7] = 255
        frame[:, 1::7] = 0
        for out in range(63, 144, 5):
            size = (out, 101)
            accepted += resampling._split(width, out) is not None
            np.testing.assert_array_equal(resampling.resize_area(frame, size),
                                          cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
    assert accepted > 20


@pytest.mark.parametrize('view', ['mirror', 'gray', 'float', 'small', 'upscale'])
def test_area_other_layouts_keep_opencv_behavior(view):
    image = np.arange(360 * 720 * 3, dtype=np.uint8).reshape(360, 720, 3)
    size = (130, 90)
    if view == 'mirror': image = image[:, ::-1]
    if view == 'gray': image = image[:, :, 0]
    if view == 'float': image = image.astype(np.float32)
    if view == 'small': image = image[:100, :100]
    if view == 'upscale': size = (900, 500)
    np.testing.assert_array_equal(resampling.resize_area(image, size),
                                  cv2.resize(image, size, interpolation=cv2.INTER_AREA))


def test_area_worker_failure_propagates_and_finishes(monkeypatch):
    original = cv2.resize
    completed = threading.Event()
    def fail_in_worker(*args, **kwargs):
        if threading.current_thread().name.startswith('capture-area'):
            completed.set()
            raise ValueError('resize worker failed')
        return original(*args, **kwargs)
    monkeypatch.setattr(cv2, 'resize', fail_in_worker)
    with pytest.raises(ValueError, match='resize worker failed'):
        resampling.resize_area(np.zeros((720, 1280, 3), np.uint8), (320, 180))
    assert completed.is_set()


def pose_graph(scores):
    onnx = pytest.importorskip('onnx')
    from onnx import helper as h, numpy_helper as nh, TensorProto as T
    scores = np.asarray(scores, np.float32).reshape(1, 1, -1)
    count = scores.shape[-1]
    weights = [nh.from_array(scores, 'scores'), nh.from_array(np.ones((1, 4, count), np.float32), 'boxes'),
               nh.from_array(np.ones((1, 18, count), np.float32), 'points')]
    graph = h.make_graph([h.make_node('Identity', ['scores'], ['class_scores']),
                          h.make_node('Concat', ['boxes', 'class_scores', 'points'], ['output0'], axis=1)],
                         'pose', [h.make_tensor_value_info('images', T.FLOAT, [1, 3, 32, 32])],
                         [h.make_tensor_value_info('output0', T.FLOAT, [1, 23, count])], weights)
    model = h.make_model(graph, opset_imports=[h.make_opsetid('', 17)], ir_version=8)
    h.set_model_props(model, {'names': "{0: 'dog_face'}", 'kpt_shape': '[6, 3]'})
    return model


@pytest.mark.parametrize('scores,threshold,kept', [([.1, .2], .35, False), ([.5, .1], .5, True),
                                                  ([float('nan'), .9], .5, True),
                                                  ([float('nan'), .1], .5, False)])
def test_conditional_heads_keep_threshold_nan_and_full_output(scores, threshold, kept):
    ort = pytest.importorskip('onnxruntime')
    compile_tool = tool('compile_dog_capture_cpu')
    original = pose_graph(scores)
    model, detail = compile_tool.conditional_heads(original)
    session = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
    result = session.run(None, {'images': np.zeros((1, 3, 32, 32), np.float32),
                               detail['conditional_score_input']: np.array(threshold, np.float32)})[0]
    if kept:
        reference = ort.InferenceSession(original.SerializeToString(), providers=['CPUExecutionProvider'])
        expected = reference.run(None, {'images': np.zeros((1, 3, 32, 32), np.float32)})[0]
        np.testing.assert_array_equal(result, expected)
    else:
        assert result.shape == (1, 23, 0)


def test_compiled_adapter_uses_runtime_threshold_and_rejects_wrong_target(tmp_path, monkeypatch):
    onnx = pytest.importorskip('onnx')
    from dog_capture.engine.detectors import YoloPoseDetector
    from dog_capture.engine.runtime import cpu_model_target
    from dog_capture.engine.zoo import sha256
    model, detail = tool('compile_dog_capture_cpu').conditional_heads(pose_graph([.5, .1]))
    path = tmp_path / 'cpu.onnx'
    onnx.save(model, path)
    sidecar = {'kind': 'yolo_pose', 'schema': 'dogface6', 'kpt_shape': [6, 3], 'input_size': 32,
               'sha256': sha256(path), 'cpu_compiled': {'version': 1, 'target': cpu_model_target(), **detail}}
    path.with_suffix('.json').write_text(json.dumps(sidecar))
    detector = YoloPoseDetector(path, providers='cpu', threads=1, score_threshold=.5)
    tensor = np.zeros((1, 3, 32, 32), np.float32)
    assert detector._run(tensor)[0].shape[-1] == 2
    detector.score_threshold = .6
    assert detector._run(tensor)[0].shape[-1] == 0
    run = detector.session.run
    warmed = []
    def record(*args, **kwargs):
        result = run(*args, **kwargs)
        warmed.append(result[0].shape[-1])
        return result
    with monkeypatch.context() as patch:
        patch.setattr(detector.session, 'run', record)
        detector.warmup(3)
    assert warmed == [2, 2, 2]
    assert detector.score_threshold == .6
    assert detector._run(tensor)[0].shape[-1] == 0
    verification = tool('benchmark_dog_capture_v1').verify_provider(detector, tensor, tmp_path, 'test')
    assert verification['node_executions_by_provider']['CPUExecutionProvider'] > 0
    sidecar['cpu_compiled']['target']['ort_version'] = 'wrong'
    path.with_suffix('.json').write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match='target differs'):
        YoloPoseDetector(path, providers='cpu', threads=1)


def test_conditional_comparison_cannot_hide_positive_or_missing_reference():
    compare = tool('compare_dog_capture_runtime').compare_outputs
    key = ('face', (1, 3, 32, 32), '<f4', 'input-hash')
    old = {key: {'outputs': ['original-hash'], 'empty': True, 'skipped_heads': False}}
    new = {key: {'outputs': ['empty-shape'], 'empty': True, 'skipped_heads': True}}
    assert compare(old, new) == ([], 1)
    positive = copy.deepcopy(old)
    positive[key]['empty'] = False
    assert compare(positive, new) == (['face'], 0)
    assert compare({}, new) == (['face'], 0)
    new[key]['skipped_heads'] = False
    assert compare(old, new) == (['face'], 0)


def test_conditional_compiler_rejects_wrong_head_order():
    compiler = tool('compile_dog_capture_cpu')
    model = pose_graph([.1, .8])
    final = model.graph.node[-1]
    final.input[:] = ['class_scores', 'boxes', 'points']
    with pytest.raises(ValueError, match='boxes, scores, keypoints in that order'):
        compiler.conditional_heads(model)
