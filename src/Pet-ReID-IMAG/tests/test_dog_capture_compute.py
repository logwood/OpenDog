"""Arithmetic-accounting guards; no model execution or timing assertions."""
import importlib.util
from pathlib import Path

import pytest

onnx = pytest.importorskip('onnx')
spec = importlib.util.spec_from_file_location('capture_compute', Path(__file__).resolve().parents[1] / 'tools/analyze_dog_capture_compute.py')
compute = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compute)


@pytest.mark.parametrize('outputs,groups', [(16, 1), (16, 2), (8, 8)])
def test_convolution_group_and_depthwise_accounting(outputs, groups):
    node = onnx.helper.make_node('Conv', ['image', 'weight'], ['out'], group=groups)
    shapes = {'image': [1, 8, 5, 5], 'weight': [outputs, 8 // groups, 3, 3], 'out': [1, outputs, 3, 3]}
    expected = 9 * outputs * (8 // groups) * 9
    assert compute.dense_macs(node, shapes) == expected
    shapes['weight'][1] += 1
    with pytest.raises(ValueError, match='Conv/group'):
        compute.dense_macs(node, shapes)


@pytest.mark.parametrize('trans_a,trans_b', [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_gemm_transposition_preserves_contraction_work(trans_a, trans_b):
    node = onnx.helper.make_node('Gemm', ['a', 'b'], ['out'], transA=trans_a, transB=trans_b)
    shapes = {'a': [3, 5] if trans_a else [5, 3], 'b': [7, 3] if trans_b else [3, 7], 'out': [5, 7]}
    assert compute.dense_macs(node, shapes) == 105


@pytest.mark.parametrize('a,b,out,expected', [
    ([3], [3], [], 3), ([3], [3, 7], [7], 21),
    ([5, 3], [3], [5], 15), ([2, 1, 5, 3], [4, 3, 7], [2, 4, 5, 7], 840)])
def test_matmul_vectors_and_broadcast_batches(a, b, out, expected):
    node = onnx.helper.make_node('MatMul', ['a', 'b'], ['out'])
    assert compute.dense_macs(node, {'a': a, 'b': b, 'out': out}) == expected


def test_unknown_shapes_and_mismatched_contraction_are_not_counted_as_zero():
    node = onnx.helper.make_node('MatMul', ['a', 'b'], ['out'])
    with pytest.raises(ValueError, match='Unresolved'):
        compute.dense_macs(node, {'a': ['batch', 3], 'b': [3, 7], 'out': ['batch', 7]})
    with pytest.raises(ValueError, match='Inconsistent'):
        compute.dense_macs(node, {'a': [5, 3], 'b': [4, 7], 'out': [5, 7]})


def test_conditional_costs_count_shared_work_once_and_empty_skips_decode():
    groups = {key: {'macs': value} for key, value in zip(
        ('shared_and_scores', 'fine_40', 'middle_20', 'coarse_10', 'positive_decode'), (100, 20, 5, 2, 1))}
    assert compute.pose_call_macs(groups, '000') == 100
    assert compute.pose_call_macs(groups, '010') == 106
    assert compute.pose_call_macs(groups, '101') == 123
    assert compute.pose_call_macs(groups, '111') == 128
    with pytest.raises(ValueError, match='three-bit'):
        compute.pose_call_macs(groups, 'x01')


def test_unsupported_operators_fail_closed(tmp_path):
    h, t = onnx.helper, onnx.TensorProto
    graph = h.make_graph([h.make_node('ConvTranspose', ['x', 'w'], ['y'])], 'unsupported',
                         [h.make_tensor_value_info('x', t.FLOAT, [1, 1, 3, 3]),
                          h.make_tensor_value_info('w', t.FLOAT, [1, 1, 2, 2])],
                         [h.make_tensor_value_info('y', t.FLOAT, [1, 1, 4, 4])])
    path = tmp_path / 'unsupported.onnx'
    onnx.save(h.make_model(graph), path)
    with pytest.raises(ValueError, match='supported standard operators'):
        compute.analyze_model(path)
