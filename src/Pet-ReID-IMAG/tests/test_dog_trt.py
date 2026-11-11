"""CPU-only postprocessing tests plus optional real TensorRT runtime checks."""
import importlib.util
from pathlib import Path
import threading

import numpy as np
import pytest

from dog_capture.trt import fast_nms


def test_detector_keeps_winning_class_before_dog_filter():
    boxes = np.array([[0, 0, 1, 1], [.02, .02, 1, 1], [2, 2, 3, 3]], np.float32)
    b, c, s, n = fast_nms(boxes, [.95, .9, .8], [0, 17, 17])
    assert int(n[0]) == 2
    assert list(c[0, :2]) == [0, 17]
    assert b[0, 1] == pytest.approx(boxes[2])


def test_nms_does_not_truncate_dense_overlaps_before_suppression():
    boxes = np.repeat(np.array([[0, 0, 1, 1]], np.float32), 1500, axis=0)
    boxes[-1] = [2, 2, 3, 3]
    scores = np.linspace(.99, .4, 1500)
    b, c, s, n = fast_nms(boxes, scores, np.full(1500, 17))
    assert int(n[0]) == 2 and s[0, 1] == pytest.approx(.4)
    assert b[0, 1] == pytest.approx(boxes[-1])


def test_nms_rejects_nonfinite_and_applies_original_threshold_semantics():
    boxes = [[0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5], [float('nan'), 0, 1, 1]]
    b, c, s, n = fast_nms(boxes, [.35, .349, float('nan'), .99], [17] * 4)
    assert int(n[0]) == 1 and s[0, 0] == pytest.approx(.35)
    assert np.isfinite(b).all() and np.isfinite(s).all()


def test_nms_empty_shapes_and_maximum_count():
    b, c, s, n = fast_nms(np.empty((0, 4)), [], [])
    assert b.shape == (1, 25, 4) and n[0] == 0
    boxes = np.array([[i * 2, 0, i * 2 + 1, 1] for i in range(40)])
    b, c, s, n = fast_nms(boxes, np.linspace(.99, .5, 40), np.full(40, 17))
    assert n[0] == 25


def test_tensorrt_graph_matches_direct_enqueue_across_worker_thread():
    if importlib.util.find_spec('tensorrt') is None:
        pytest.skip('TensorRT environment not installed')
    from dog_capture.trt import TrtModel
    path = Path(__file__).resolve().parents[3] / 'models/pretrained/dog_capture_trt_v3/landmarks.engine'
    if not path.exists():
        pytest.skip('TensorRT engines not built')
    x = np.full((1, 384, 384, 3), .5, np.float32)
    with TrtModel(path, use_cuda_graph=False) as direct:
        reference = [a.copy() for a in direct(x)]
    with TrtModel(path) as graph:
        result, errors = [], []
        def worker():
            try:
                result.extend(a.copy() for a in graph(x))
            except BaseException as exc:
                errors.append(exc)
        thread = threading.Thread(target=worker)
        thread.start(); thread.join(timeout=15)
        assert not thread.is_alive() and not errors
        for expected, actual in zip(reference, result):
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        with pytest.raises(ValueError):
            graph(np.zeros((1, 128, 128, 3), np.float32))
    with pytest.raises(RuntimeError):
        graph(x)


def test_contiguous_head_preserves_coordinates_entropy_and_coordinate_grid():
    onnx = pytest.importorskip('onnx')
    ort = pytest.importorskip('onnxruntime')
    from dog_capture.trt_export import contiguous_landmark_head, rename_tensor
    path = Path(__file__).resolve().parents[3] / 'models/pretrained/dog_capture_trt_v3/landmarks.raw.onnx'
    if not path.exists():
        pytest.skip('Exported reference graph not available')
    original = onnx.load(path)
    for output, name in zip(list(original.graph.output), ['landmarks', 'landmark_entropy']):
        rename_tensor(original, output.name, name)
    heat = next(n for n in original.graph.node if n.op_type == 'Conv' and '/heatmap_conv/' in n.name)
    heat_name = heat.output[0]
    weights = {v.name: onnx.numpy_helper.to_array(v) for v in original.graph.initializer}
    grids = []
    for suffix in ('soft_argmax/mul_1', 'soft_argmax/mul_2'):
        mul = next(n for n in original.graph.node if n.name.endswith(suffix))
        grids.append(next(weights[v] for v in mul.input if v in weights).reshape(192).astype(np.float64))
    optimized = contiguous_landmark_head(onnx.ModelProto.FromString(original.SerializeToString()))
    sessions = []
    for model in (original, optimized):
        nodes = list(model.graph.node)
        heat_index = next(i for i, n in enumerate(nodes) if heat_name in n.output)
        tail = nodes[heat_index + 1:]
        del model.graph.node[:]
        model.graph.node.extend(tail)
        del model.graph.input[:]
        model.graph.input.extend([onnx.helper.make_tensor_value_info(heat_name, onnx.TensorProto.FLOAT, [1, 46, 192, 192])])
        used = {name for n in tail for name in n.input}
        weights = [v for v in model.graph.initializer if v.name in used]
        del model.graph.initializer[:]
        model.graph.initializer.extend(weights)
        del model.graph.value_info[:]
        model.ir_version = 10
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        sessions.append(ort.InferenceSession(model.SerializeToString(), options, providers=['CPUExecutionProvider']))
    rng = np.random.default_rng(73)
    for scale in (0., 4., 25.):
        heatmap = rng.normal(size=(1, 46, 192, 192)).astype(np.float32) * scale
        a, b = [session.run(None, {heat_name: heatmap}) for session in sessions]
        probabilities = np.exp(heatmap.astype(np.float64) - heatmap.max(axis=(2, 3), keepdims=True))
        probabilities /= probabilities.sum(axis=(2, 3), keepdims=True)
        x = (probabilities * grids[0].reshape(1, 1, 1, 192)).sum(axis=(2, 3))
        y = (probabilities * grids[1].reshape(1, 1, 192, 1)).sum(axis=(2, 3))
        exact = [np.stack((x, y), axis=-1).reshape(1, 92),
                 -(probabilities * np.log(np.maximum(probabilities, 1e-6))).sum(axis=(2, 3))]
        for index, (old, actual, expected) in enumerate(zip(a, b, exact)):
            # The old strided FP32 reduction accumulates ~6e-5 error on dense
            # maps (<0.04 px at 384). Check the rewrite against FP64 as well.
            np.testing.assert_allclose(actual, old, rtol=1e-4, atol=1e-4)
            # Uniform-map entropy sums 36,864 FP32 terms (about 1.3e-4
            # absolute rounding error in ORT); coordinates use tighter units.
            np.testing.assert_allclose(actual, expected, rtol=2e-5,
                                       atol=1e-5 if index == 0 else 2e-4)
