"""Layout permutation proofs and conditional decode precision contracts."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip('onnx')
ort = pytest.importorskip('onnxruntime')
from onnx import helper as h, numpy_helper as nh, TensorProto as T


def tool(name):
    path = Path(__file__).resolve().parents[1] / 'tools' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rewrite = tool('dog_capture_cpu_graph')


def session(model):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(model.SerializeToString(), options, providers=['CPUExecutionProvider'])


@pytest.fixture(scope='module')
def block():
    return rewrite.cpu_channel_block()


def layout_graph(block, batch=1, spatial=(3, 7), split_sizes=None):
    sizes = split_sizes or [block, block]
    shape = [batch, sum(sizes), *spatial]
    half = [batch, sizes[0], *spatial]
    nodes = [h.make_node('ReorderInput', ['x'], ['packed'], domain=rewrite.NCHWC),
             h.make_node('ReorderOutput', ['packed'], ['plain'], domain=rewrite.NCHWC, channels=sum(sizes)),
             h.make_node('Split', ['plain', 'sizes'], ['a', 'b'], axis=1),
             h.make_node('Add', ['a', 'a'], ['sum']),
             h.make_node('Concat', ['sum', 'b'], ['joined'], axis=1),
             h.make_node('ReorderInput', ['joined'], ['repacked'], domain=rewrite.NCHWC),
             h.make_node('ReorderOutput', ['repacked'], ['y'], domain=rewrite.NCHWC, channels=sum(sizes))]
    info = [h.make_tensor_value_info(name, T.FLOAT, dims) for name, dims in (
        ('plain', shape), ('a', half), ('b', [batch, sizes[1], *spatial]), ('sum', half), ('joined', shape))]
    model = h.make_model(h.make_graph(nodes, 'layout', [h.make_tensor_value_info('x', T.FLOAT, shape)],
        [h.make_tensor_value_info('y', T.FLOAT, shape)], [nh.from_array(np.array(sizes, np.int64), 'sizes')],
        value_info=info), opset_imports=[h.make_opsetid('', 17), h.make_opsetid(rewrite.NCHWC, 1)], ir_version=8)
    return model


@pytest.mark.parametrize('batch,spatial', [(1, (3, 7)), (2, (1, 1)), (3, (5, 13))])
def test_aligned_split_add_concat_are_byte_exact(block, batch, spatial):
    original = layout_graph(block, batch, spatial)
    snapshot = original.SerializeToString()
    changed, detail = rewrite.preserve_blocked_layout(original, block)
    assert original.SerializeToString() == snapshot
    assert detail['removed_reorders'] == {'ReorderOutput': 1, 'ReorderInput': 1}
    assert detail['removed_reorder_output_payload_bytes'] == 2 * batch * 2 * block * np.prod(spatial) * 4
    x = np.random.default_rng(23).normal(size=(batch, 2 * block, *spatial)).astype(np.float32)
    # Include signed zeros; array_equal alone does not check their bits.
    x.reshape(-1)[:2] = [-0., +0.]
    assert session(original).run(None, {'x': x})[0].tobytes() == session(changed).run(None, {'x': x})[0].tobytes()


@pytest.mark.parametrize('unsafe', ['unaligned', 'spatial_axis', 'broadcast', 'channels_last', 'dynamic', 'public_packed'])
def test_unsafe_layouts_are_not_rewritten(block, unsafe):
    original = layout_graph(block, split_sizes=[block - 1, block + 1] if unsafe == 'unaligned' else None)
    if unsafe == 'spatial_axis':
        original.graph.node[2].attribute[0].i = 2
    elif unsafe == 'broadcast':
        original.graph.initializer.append(nh.from_array(np.ones((1,), np.float32), 'bias'))
        original.graph.node[3].input[1] = 'bias'
    elif unsafe == 'channels_last':
        original.graph.node[1].attribute.append(h.make_attribute('channels_last', 1))
    elif unsafe == 'dynamic':
        dim = original.graph.value_info[0].type.tensor_type.shape.dim[1]
        dim.ClearField('dim_value')
        dim.dim_param = 'unknown_channels'
    elif unsafe == 'public_packed':
        original.graph.output.append(h.make_tensor_value_info('repacked', T.FLOAT, [1, 2 * block, 3, 7]))
    _, detail = rewrite.preserve_blocked_layout(original, block)
    assert not detail['removed_reorders']


def test_layout_reserved_names_and_invalid_block_fail(block):
    model = layout_graph(block)
    model.graph.node[0].name = 'capture_layout_existing'
    with pytest.raises(ValueError, match='reserved'):
        rewrite.preserve_blocked_layout(model, block)
    with pytest.raises(ValueError, match='power of two'):
        rewrite.preserve_blocked_layout(model, 3)


def dfl_graph(bits, block):
    # Reuse the scale fixture, replacing its placeholder box slice with the
    # real seven-operation CPU DFL chain and nonuniform distributions.
    from test_dog_capture_scale_heads import scale_graph
    model = scale_graph(bits)
    rng = np.random.default_rng(7)
    for weight in model.graph.initializer:
        if weight.name.startswith('weights_'):
            weight.CopyFrom(nh.from_array(rng.normal(size=list(weight.dims)).astype(np.float32), weight.name))
    source, target = '/model.23/Concat_output_0', '/model.23/dfl/Reshape_1_output_0'
    chain = [h.make_node('Reshape', [source, 'shape0'], ['d0']),
             h.make_node('Transpose', ['d0'], ['d1'], perm=[0, 2, 1, 3]),
             h.make_node('Softmax', ['d1'], ['d2'], axis=1),
             h.make_node('ReorderInput', ['d2'], ['d3'], domain=rewrite.NCHWC),
             h.make_node('Conv', ['d3', 'dfl_weights'], ['d4'], domain=rewrite.NCHWC, kernel_shape=[1, 1]),
             h.make_node('ReorderOutput', ['d4'], ['d5'], domain=rewrite.NCHWC, channels=1),
             h.make_node('Reshape', ['d5', 'shape1'], [target])]
    replaced = []
    for node in model.graph.node:
        if node.op_type == 'Slice':
            replaced.extend(chain)
        else:
            for i, name in enumerate(node.input):
                if name == 'boxes': node.input[i] = target
            for i, name in enumerate(node.output):
                if name == 'raw_boxes': node.output[i] = source
            replaced.append(copy.deepcopy(node))
    model.graph.ClearField('node')
    model.graph.node.extend(replaced)
    for value in model.graph.value_info:
        if value.name == 'boxes': value.name = target
    model.graph.initializer.extend([
        nh.from_array(np.array([1, 4, 16, 2100], np.int64), 'shape0'),
        nh.from_array(np.array([1, 4, 2100], np.int64), 'shape1'),
        nh.from_array(rng.normal(size=(block, 16, 1, 1)).astype(np.float32), 'dfl_weights')])
    model.opset_import.append(h.make_opsetid(rewrite.NCHWC, 1))
    return model


@pytest.mark.parametrize('bits', range(8))
def test_dfl_all_scale_combinations_keep_scores_and_eligible_anchor_bits(block, bits):
    original = dfl_graph(bits, block)
    conditional, detail = tool('compile_dog_capture_cpu').conditional_scale_heads(original)
    changed, decoded = rewrite.conditional_scale_decode(conditional)
    assert decoded['per_scale_decode_nodes'] == 7
    x = np.zeros((1, 3, 320, 320), np.float32)
    expected = session(original).run(None, {'images': x})[0]
    compiled = session(changed)
    for threshold in (.5, -np.inf, .51):
        actual = compiled.run(None, {'images': x, detail['conditional_score_input']: np.asarray(threshold, np.float32)})[0]
        keep = expected[0, 4] >= threshold
        if not keep.any():
            assert actual.shape == (1, 23, 0)
        else:
            assert actual[:, 4:5].tobytes() == expected[:, 4:5].tobytes()
            assert actual[:, :, keep].tobytes() == expected[:, :, keep].tobytes()


@pytest.mark.parametrize('change', ['axis', 'shape', 'spatial_conv', 'outside_consumer', 'dependency'])
def test_changed_dfl_graph_is_rejected(block, change):
    original = dfl_graph(2, block)
    if change == 'axis':
        next(n for n in original.graph.node if n.op_type == 'Softmax').attribute[0].i = 3
    elif change == 'shape':
        value = next(v for v in original.graph.initializer if v.name == 'shape0')
        value.CopyFrom(nh.from_array(np.array([1, 16, 4, 2100], np.int64), 'shape0'))
    elif change == 'spatial_conv':
        next(n for n in original.graph.node if n.op_type == 'Conv').attribute[0].ints[:] = [1, 3]
    elif change == 'outside_consumer':
        original.graph.node.append(h.make_node('Identity', ['d2'], ['unrelated']))
    elif change == 'dependency':
        next(n for n in original.graph.node if n.op_type == 'Softmax').input[0] = 'd0'
    compiled, _ = tool('compile_dog_capture_cpu').conditional_scale_heads(original)
    with pytest.raises(ValueError, match='DFL'):
        rewrite.conditional_scale_decode(compiled)


def test_decode_option_requires_scale_gating_before_writing(tmp_path):
    from dog_capture.engine.config import CaptureConfig
    output = tmp_path / 'compiled'
    with pytest.raises(ValueError, match='requires --per-scale'):
        tool('compile_dog_capture_cpu').compile_config(CaptureConfig(), output, scale_decode=True)
    assert not output.exists()
