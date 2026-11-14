"""Exact, device-local rewrites applied AFTER ORT has fixed its CPU fusions.

No weights, convolution attributes, floating-point reduction order or thresholds
are changed. Unknown layouts are left alone (layout propagation) or rejected
(the checkpoint-specific DFL rewrite). Never optimize the rewritten graph again.
"""
from __future__ import annotations

from collections import Counter
import copy
import math

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T

NCHWC = 'com.microsoft.nchwc'


def attributes(node):
    return {a.name: h.get_attribute_value(a) for a in node.attribute}


def cpu_channel_block():
    """Ask the deployment CPU's MLAS kernel, without guessing from ISA flags."""
    import onnxruntime as ort
    model = h.make_model(h.make_graph([
        h.make_node('ReorderInput', ['x'], ['y'], domain=NCHWC, channels_last=0)],
        'block_probe', [h.make_tensor_value_info('x', T.FLOAT, [1, 4, 1, 1])],
        [h.make_tensor_value_info('y', T.FLOAT, [1, None, 1, 1])]),
        opset_imports=[h.make_opsetid('', 17), h.make_opsetid(NCHWC, 1)], ir_version=8)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(model.SerializeToString(), options, providers=['CPUExecutionProvider'])
    block = session.run(None, {'x': np.zeros((1, 4, 1, 1), np.float32)})[0].shape[1]
    if block < 4 or block & (block - 1):
        raise ValueError(f'Unsupported CPU channel block: {block}')
    return block


def restore_original_shapes(base, original):
    """Recover static facts for surviving logical names, lost by ORT export.

    This only attaches shape metadata; it does not execute or replace kernels.
    Optimized internal names have no original counterpart and remain untouched.
    """
    from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
    inferred = SymbolicShapeInference.infer_shapes(original, auto_merge=True, verbose=0)
    model = copy.deepcopy(base)
    seen = {v.name for v in [*model.graph.input, *model.graph.output, *model.graph.value_info]}
    live = {x for n in model.graph.node for x in n.output}
    for value in inferred.graph.value_info:
        if value.name in live and value.name not in seen:
            model.graph.value_info.append(copy.deepcopy(value))
            seen.add(value.name)
    return model


def preserve_blocked_layout(base, block):
    """Commute channel-aligned Split/Concat and equal-shape Add with NCHWc.

    With C and every split boundary divisible by B, each group occupies the
    same contiguous C*H*W bytes in NCHW and NCHWc. Add commutes with the same
    permutation on both inputs. No broadcast, padding, spatial-axis split or
    channels-last layout is accepted. The original graph is never mutated.
    """
    if block < 4 or block & (block - 1):
        raise ValueError('Channel block must be a power of two >= 4')
    model = copy.deepcopy(base)
    nodes = list(model.graph.node)
    if any(a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
           for n in nodes for a in n.attribute):
        raise ValueError('Layout propagation requires a flat CPU graph before conditional heads')
    all_names = {x for n in nodes for x in [n.name, *n.input, *n.output]}
    if any(x.startswith('capture_layout_') for x in all_names):
        raise ValueError('Graph uses reserved capture_layout_ names or is already rewritten')
    producers = {x: i for i, n in enumerate(nodes) for x in n.output}
    info = {v.name: v for v in [*model.graph.input, *model.graph.value_info, *model.graph.output]}
    shapes = {name: [d.dim_value for d in v.type.tensor_type.shape.dim] for name, v in info.items()}
    weights = {v.name: nh.to_array(v) for v in model.graph.initializer}
    memo, created, new_info = {}, [], []

    def aligned(name):
        shape = shapes.get(name, [])
        return (len(shape) == 4 and all(d > 0 for d in shape) and shape[1] % block == 0
                and info[name].type.tensor_type.elem_type == T.FLOAT)

    def blocked(name):
        if name in memo:
            return memo[name]
        memo[name] = None
        if name not in producers or not aligned(name):
            return None
        node = nodes[producers[name]]
        attrs = attributes(node)
        if (node.domain == NCHWC and node.op_type == 'ReorderOutput'
                and not attrs.get('channels_last', 0) and attrs.get('channels') == shapes[name][1]):
            memo[name] = node.input[0]
            return memo[name]
        inputs = None
        if node.domain in ('', 'ai.onnx') and node.op_type == 'Split' and attrs.get('axis', 0) == 1:
            if aligned(node.input[0]) and all(aligned(x) for x in node.output):
                source = shapes[node.input[0]]
                sizes = [shapes[x][1] for x in node.output]
                fixed = weights.get(node.input[1]) if len(node.input) > 1 else attrs.get('split')
                valid_sizes = (fixed is not None and np.array_equal(fixed, sizes)) or (
                    fixed is None and len(node.input) == 1 and len(set(sizes)) == 1)
                if (valid_sizes and sum(sizes) == source[1]
                        and all(shapes[x][0] == source[0] and shapes[x][2:] == source[2:] for x in node.output)):
                    first = blocked(node.input[0])
                    if first is not None:
                        inputs = [first, *node.input[1:]]
        elif node.domain in ('', 'ai.onnx') and node.op_type == 'Concat' and attrs.get('axis') == 1:
            if (all(aligned(x) and shapes[x][0] == shapes[name][0]
                    and shapes[x][2:] == shapes[name][2:] for x in node.input)
                    and sum(shapes[x][1] for x in node.input) == shapes[name][1]):
                inputs = [blocked(x) for x in node.input]
        elif node.domain in ('', 'ai.onnx') and node.op_type == 'Add':
            if len(node.input) == 2 and all(shapes.get(x) == shapes[name] for x in node.input):
                inputs = [blocked(x) for x in node.input]
        if inputs is None or any(x is None for x in inputs):
            return None
        clone = copy.deepcopy(node)
        clone.name = 'capture_layout_' + str(len(created))
        clone.ClearField('input')
        clone.input.extend(inputs)
        clone.ClearField('output')
        for i, output in enumerate(node.output):
            changed = clone.name + '_' + str(i)
            clone.output.append(changed)
            memo[output] = changed
            value = copy.deepcopy(info[output])
            value.name = changed
            new_info.append(value)
        created.append(clone)
        return memo[name]

    aliases = {}
    graph_outputs = {v.name for v in model.graph.output}
    for node in nodes:
        if (node.domain == NCHWC and node.op_type == 'ReorderInput'
                and not attributes(node).get('channels_last', 0) and node.output[0] not in graph_outputs):
            replacement = blocked(node.input[0])
            if replacement is not None:
                aliases[node.output[0]] = replacement

    def resolve(name):
        while name in aliases:
            name = aliases[name]
        return name

    expanded = nodes + created
    for node in expanded:
        for i, name in enumerate(node.input):
            node.input[i] = resolve(name)
    producer = {name: node for node in expanded for name in node.output}
    ordered, visited = [], set()

    def visit(name):
        if name not in producer:
            return
        node = producer[name]
        if node.output[0] in visited:
            return
        visited.add(node.output[0])
        for source in node.input:
            visit(source)
        ordered.append(node)

    for output in model.graph.output:
        visit(output.name)
    live = {x for node in ordered for x in node.output}
    removed = [n for n in nodes if n.domain == NCHWC and n.op_type in ('ReorderInput', 'ReorderOutput')
               and n.output[0] not in live]
    # Output payload is a logical byte count, not measured DRAM read/write traffic.
    payload = sum(4 * math.prod(shapes.get(n.output[0]) or shapes[n.input[0]]) for n in removed)
    model.graph.ClearField('node')
    model.graph.node.extend(ordered)
    retained = [v for v in [*model.graph.value_info, *new_info] if v.name in live]
    model.graph.ClearField('value_info')
    model.graph.value_info.extend(retained)
    onnx.checker.check_model(model)
    return model, {'block': block, 'old_nodes': len(nodes), 'new_nodes': len(ordered),
                   'removed_reorders': dict(Counter(n.op_type for n in removed)),
                   'removed_reorder_output_payload_bytes': payload,
                   'cloned_operations': dict(Counter(n.op_type for n in created if n.output[0] in live))}


def conditional_scale_decode(compiled):
    """Move the archived, anchor-independent DFL chain into each active scale.

    compile_config pins the portable checkpoint SHA. These additional guards
    reject changed optimizer layouts: Softmax stays on the 16 distribution
    channels and the convolution stays pointwise. Discarded anchors may differ;
    scores and every anchor reaching the threshold retain their exact values.
    """
    model = copy.deepcopy(compiled)
    metadata = {p.key: p.value for p in model.metadata_props}
    if metadata.get('dog_capture.raw_output_contract') != 'scores_and_eligible_anchors_exact':
        raise ValueError('Scale decode requires per-scale conditional heads')
    outer = next((n for n in model.graph.node if n.name == 'capture_gate_heads'), None)
    if outer is None:
        raise ValueError('Missing conditional pose heads')
    positive = next(a.g for a in outer.attribute if a.name == 'then_branch')
    nodes = list(positive.node)
    if any(x.startswith('capture_dfl_') for n in nodes for x in [n.name, *n.input, *n.output]):
        raise ValueError('Reserved capture_dfl_ names or already rewritten decoder')
    producers = {out: i for i, n in enumerate(nodes) for out in n.output}
    source = '/model.23/Concat_output_0'
    target = '/model.23/dfl/Reshape_1_output_0'
    required = set()

    def visit(name):
        if name == source:
            return
        index = producers.get(name)
        if index is None or index in required:
            return
        required.add(index)
        for value in nodes[index].input:
            visit(value)

    visit(target)
    chain = [n for i, n in enumerate(nodes) if i in required]
    expected = [('', 'Reshape'), ('', 'Transpose'), ('', 'Softmax'), (NCHWC, 'ReorderInput'),
                (NCHWC, 'Conv'), (NCHWC, 'ReorderOutput'), ('', 'Reshape')]
    if [(n.domain, n.op_type) for n in chain] != expected or source not in producers:
        raise ValueError('Unsupported CPU DFL chain')
    weights = {v.name: nh.to_array(v) for v in model.graph.initializer}
    concat = nodes[producers[source]]
    attrs = [attributes(n) for n in chain]
    if (concat.op_type != 'Concat' or attributes(concat).get('axis') not in (2, -1) or len(concat.input) != 3
            or not np.array_equal(weights.get(chain[0].input[1]), [1, 4, 16, 2100])
            or not np.array_equal(weights.get(chain[-1].input[1]), [1, 4, 2100])
            or attrs[1].get('perm') != [0, 2, 1, 3] or attrs[2].get('axis') != 1
            or attrs[3].get('channels_last', 0) or attrs[5].get('channels_last', 0)
            or attrs[5].get('channels') != 1 or attrs[4].get('kernel_shape') != [1, 1]
            or attrs[4].get('strides', [1, 1]) != [1, 1]
            or attrs[4].get('pads', [0, 0, 0, 0]) != [0, 0, 0, 0]
            or attrs[4].get('dilations', [1, 1]) != [1, 1] or attrs[4].get('group', 1) != 1):
        raise ValueError('Unsupported anchor-dependent DFL layout')
    previous = source
    for node in chain:
        if (len(node.output) != 1 or node.input[0] != previous
                or any(x not in weights for x in node.input[1:])):
            raise ValueError('Unexpected DFL dependency')
        previous = node.output[0]
    removed = required | {producers[source]}
    internal = {x for i in removed for x in nodes[i].output} - {target}
    if any(x in internal for i, n in enumerate(nodes) if i not in removed for x in n.input):
        raise ValueError('DFL intermediate has an outside consumer')
    outputs = []
    for level, width in enumerate((1600, 400, 100)):
        gate = next((n for n in nodes if n.name == f'capture_scale_{level}_heads'), None)
        if gate is None or len(gate.output) != 2 or gate.output[0] != concat.input[level]:
            raise ValueError('Unexpected scale head outputs')
        yes = next(a.g for a in gate.attribute if a.name == 'then_branch')
        no = next(a.g for a in gate.attribute if a.name == 'else_branch')
        if (len(no.node) != 2 or no.node[0].op_type != 'Constant'
                or no.output[0].name != no.node[0].output[0]):
            raise ValueError('Unexpected empty scale branch')
        prefix = f'capture_dfl_{level}_'
        aliases = {source: gate.output[0]}
        for index, node in enumerate(chain):
            clone = copy.deepcopy(node)
            clone.name = prefix + str(index)
            for j, name in enumerate(clone.input):
                if name in aliases:
                    clone.input[j] = aliases[name]
                elif node.op_type == 'Reshape' and j == 1:
                    value = weights[name].copy()
                    value[-1] = width
                    key = prefix + 'shape_' + str(index)
                    model.graph.initializer.append(nh.from_array(value, key))
                    clone.input[j] = key
            for j, name in enumerate(node.output):
                aliases[name] = prefix + 'value_' + str(index) + '_' + str(j)
                clone.output[j] = aliases[name]
            yes.node.append(clone)
        output = aliases[target]
        outputs.append(output)
        yes.output[0].CopyFrom(h.make_tensor_value_info(output, T.FLOAT, [1, 4, width]))
        gate.output[0] = output
        value_attr = next(a for a in no.node[0].attribute if a.name == 'value')
        value_attr.t.CopyFrom(nh.from_array(np.zeros((1, 4, width), np.float32)))
        no.output[0].CopyFrom(h.make_tensor_value_info(no.node[0].output[0], T.FLOAT, [1, 4, width]))
    remaining = [n for i, n in enumerate(nodes) if i not in removed]
    last_gate = max(i for i, n in enumerate(remaining) if n.name in {f'capture_scale_{j}_heads' for j in range(3)})
    remaining.insert(last_gate + 1, h.make_node('Concat', outputs, [target], axis=2, name='capture_dfl_concat'))
    positive.ClearField('node')
    positive.node.extend(remaining)
    metadata['dog_capture.per_scale_decode'] = '1'
    h.set_model_props(model, metadata)
    onnx.checker.check_model(model)
    return model, {'per_scale_decode_nodes': len(chain), 'decoded_anchors_per_active_scale': [1600, 400, 100]}
