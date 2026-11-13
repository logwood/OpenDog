"""Rejected-scale pruning must preserve scores, active anchors and ordering."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def tool(name):
    path = Path(__file__).resolve().parents[1] / 'tools' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scale_graph(bits):
    onnx = pytest.importorskip('onnx')
    from onnx import helper as h, numpy_helper as nh, TensorProto as T
    nodes, weights, info, boxes, points, scores = [], [], [], [], [], []
    for i, count in enumerate((1600, 400, 100)):
        names = [f'/model.23/Reshape{s}_output_0' for s in (('', '_1', '_2')[i], ('_6', '_7', '_8')[i])]
        for j, (name, channels) in enumerate(zip(names, (64, 18))):
            key = f'weights_{i}_{j}'
            weights.append(nh.from_array(np.full((1, channels, count), i + j + 2, np.float32), key))
            nodes.append(h.make_node('Identity', [key], [name]))
            info.append(h.make_tensor_value_info(name, T.FLOAT, [1, channels, count]))
        boxes.append(names[0]); points.append(names[1])
        scores.extend([.5 if bits & (1 << i) else .1] * count)
    weights.extend([nh.from_array(np.asarray(scores, np.float32).reshape(1, 1, -1), 'scores'),
                    nh.from_array(np.array([0], np.int64), 'start'), nh.from_array(np.array([4], np.int64), 'end'),
                    nh.from_array(np.array([1], np.int64), 'axis')])
    nodes.extend([h.make_node('Concat', boxes, ['raw_boxes'], axis=2),
                  h.make_node('Slice', ['raw_boxes', 'start', 'end', 'axis'], ['boxes']),
                  h.make_node('Concat', points, ['points'], axis=2),
                  h.make_node('Identity', ['scores'], ['class_scores']),
                  h.make_node('Concat', ['boxes', 'class_scores', 'points'], ['output0'], axis=1)])
    info.extend(h.make_tensor_value_info(name, T.FLOAT, [1, channels, 2100])
                for name, channels in (('boxes', 4), ('class_scores', 1), ('points', 18)))
    graph = h.make_graph(nodes, 'scales', [h.make_tensor_value_info('images', T.FLOAT, [1, 3, 320, 320])],
                         [h.make_tensor_value_info('output0', T.FLOAT, [1, 23, 2100])], weights, value_info=info)
    model = h.make_model(graph, opset_imports=[h.make_opsetid('', 17)], ir_version=8)
    h.set_model_props(model, {'names': "{0: 'dog_face'}", 'kpt_shape': '[6, 3]'})
    return model


@pytest.mark.parametrize('bits', range(8))
def test_scale_graph_all_branch_combinations_and_threshold_equality(bits):
    ort = pytest.importorskip('onnxruntime')
    original = scale_graph(bits)
    compiled, detail = tool('compile_dog_capture_cpu').conditional_scale_heads(original)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    before = ort.InferenceSession(original.SerializeToString(), options, providers=['CPUExecutionProvider'])
    after = ort.InferenceSession(compiled.SerializeToString(), options, providers=['CPUExecutionProvider'])
    image = np.zeros((1, 3, 320, 320), np.float32)
    expected = before.run(None, {'images': image})[0]
    for threshold in (.5, -np.inf):  # equality keeps the scale; warmup must execute every head
        actual = after.run(None, {'images': image, detail['conditional_score_input']: np.asarray(threshold, np.float32)})[0]
        keep = expected[0, 4] >= threshold
        if not keep.any():
            assert actual.shape == (1, 23, 0)
            continue
        np.testing.assert_array_equal(actual[:, 4:5], expected[:, 4:5])
        np.testing.assert_array_equal(actual[:, :, keep], expected[:, :, keep])
        if keep.all():
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('change', ['inactive_head', 'active_head', 'score', 'threshold', 'undeclared'])
def test_comparison_allows_only_explicit_inactive_head_changes(change):
    comparison = tool('compare_dog_capture_runtime')
    reference = np.ones((1, 23, 3), np.float32)
    reference[:, 4] = [.1, .5, .2]
    current = reference.copy()
    current[0, 0 if change != 'score' else 4, 1 if change == 'active_head' else 0] += .01
    key = ('face', (1, 3, 320, 320), '<f4', 'same-input')
    old = {'outputs': ['old'], 'empty': False, 'skipped_heads': False,
           'eligible_outputs': comparison.eligible_pose_signature(reference, 1, .5)}
    new = {'outputs': ['new'], 'empty': False, 'skipped_heads': False,
           'masked_scale_heads': change != 'undeclared',
           'eligible_outputs': comparison.eligible_pose_signature(current, 1, .4 if change == 'threshold' else .5)}
    counts = {'inactive_scale_calls': 0}
    result = comparison.compare_outputs({key: old}, {key: new}, counts=counts)
    assert result == (([], 0) if change == 'inactive_head' else (['face'], 0))
    assert counts['inactive_scale_calls'] == (change == 'inactive_head')
    assert comparison.compare_outputs({}, {key: new}) == (['face'], 0)


def test_scale_builder_rejects_unknown_checkpoint_before_creating_output(tmp_path, monkeypatch):
    from dog_capture.engine.config import CaptureConfig
    from dog_capture.engine import zoo
    monkeypatch.setattr(zoo, 'find_face_model', lambda config: ('yolo_pose', tmp_path / 'unknown.onnx'))
    monkeypatch.setattr(zoo, 'verify_model', lambda path: ('unknown-sha', {}))
    output = tmp_path / 'output'
    with pytest.raises(ValueError, match='only the archived'):
        tool('compile_dog_capture_cpu').compile_config(CaptureConfig(), output, per_scale=True)
    assert not output.exists()


def test_scale_builder_rejects_changed_layout():
    model = scale_graph(3)
    model.graph.value_info[0].type.tensor_type.shape.dim[1].dim_value = 32
    with pytest.raises(ValueError, match='head layout'):
        tool('compile_dog_capture_cpu').conditional_scale_heads(model)


def test_scale_builder_rejects_intermediate_used_outside_the_head():
    from onnx import helper
    model = scale_graph(1)
    name = model.graph.node[0].output[0]
    model.graph.node[0].output[0] = name + '_internal'
    model.graph.node.insert(1, helper.make_node('Identity', [name + '_internal'], [name]))
    model.graph.node.append(helper.make_node('Identity', [name + '_internal'], ['unexpected_consumer']))
    with pytest.raises(ValueError, match='outside consumer'):
        tool('compile_dog_capture_cpu').conditional_scale_heads(model)


def test_scale_output_contract_must_match_sidecar(tmp_path):
    onnx = pytest.importorskip('onnx')
    from dog_capture.engine.detectors import YoloPoseDetector
    from dog_capture.engine.runtime import cpu_model_target
    from dog_capture.engine.zoo import sha256
    model, detail = tool('compile_dog_capture_cpu').conditional_scale_heads(scale_graph(2))
    path = tmp_path / 'scales.onnx'
    onnx.save(model, path)
    metadata = {'kind': 'yolo_pose', 'schema': 'dogface6', 'kpt_shape': [6, 3], 'input_size': 320,
                'sha256': sha256(path), 'cpu_compiled': {'version': 1, 'target': cpu_model_target(), **detail}}
    path.with_suffix('.json').write_text(json.dumps(metadata))
    assert YoloPoseDetector(path, providers='cpu', threads=1).masked_scale_heads
    metadata['cpu_compiled'].pop('raw_output_contract')
    path.with_suffix('.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='output contract'):
        YoloPoseDetector(path, providers='cpu', threads=1)
