"""Count logical capture-network work without executing models or replaying video.

MACs cover Conv/Gemm/MatMul only; 2 * MACs is their conventional FLOP count,
not a total including activations, interpolation, memory traffic or scheduling.
The pinned pose graph can additionally be partitioned by its conditional heads.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import onnx

POSE_SHA256 = '48c2b9f580a6b29f6ec92f81d58bffcb06c609b4da4d77e4c85f259dc1b8d74c'
WEIGHTED = {'Conv', 'Gemm', 'MatMul'}
# Listed separately, never silently interpreted as zero total work.
OTHER = {'Constant', 'Identity', 'Shape', 'Gather', 'Slice', 'Concat', 'Split',
         'Reshape', 'Transpose', 'Unsqueeze', 'Squeeze', 'Flatten', 'Cast',
         'Add', 'Sub', 'Mul', 'Div', 'Neg', 'Exp', 'Sigmoid', 'Softmax', 'Relu',
         'Clip', 'ReduceSum', 'ReduceMean', 'ReduceL2', 'MaxPool', 'Resize'}


def attributes(node):
    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}


def fixed(shape):
    return shape is not None and all(isinstance(d, int) and d >= 0 for d in shape)


def tensor_info(model):
    info = {}
    for value in (*model.graph.input, *model.graph.value_info, *model.graph.output):
        tensor = value.type.tensor_type
        shape = ([int(d.dim_value) if d.HasField('dim_value') else d.dim_param
                  for d in tensor.shape.dim] if tensor.HasField('shape') else None)
        info[value.name] = (shape, tensor.elem_type)
    for value in model.graph.initializer:
        info[value.name] = (list(value.dims), value.data_type)
    return info


def dense_macs(node, shapes):
    """Logical dense dot-product count; weight Cin already includes Conv groups."""
    if node.op_type not in WEIGHTED:
        raise ValueError(f'Not a supported dot-product operator: {node.op_type}')
    names = [*node.input[:2], node.output[0]]
    if any(not fixed(shapes.get(name)) for name in names):
        raise ValueError(f'Unresolved shape at {node.name or node.op_type}')
    left, right, output = (shapes[name] for name in names)
    attrs = attributes(node)
    if node.op_type == 'Conv':
        group = attrs.get('group', 1)
        if (len(left) < 3 or len(left) != len(right) or len(output) != len(left)
                or group < 1 or left[1] != right[1] * group or right[0] % group
                or output[:2] != [left[0], right[0]]):
            raise ValueError('Inconsistent Conv/group shapes')
        return math.prod(output) * math.prod(right[1:])
    if node.op_type == 'Gemm':
        if len(left) != 2 or len(right) != 2:
            raise ValueError('Gemm requires matrices')
        m, k = left[::-1] if attrs.get('transA', 0) else left
        kb, n = right[::-1] if attrs.get('transB', 0) else right
        if k != kb or output != [m, n]:
            raise ValueError('Inconsistent Gemm shapes')
        return m * n * k
    if not left or not right:
        raise ValueError('MatMul does not accept scalars')
    a = [1, *left] if len(left) == 1 else left
    b = [*right, 1] if len(right) == 1 else right
    expected = [*np.broadcast_shapes(tuple(a[:-2]), tuple(b[:-2])), a[-2], b[-1]]
    if len(left) == 1:
        expected.pop(-2)
    if len(right) == 1:
        expected.pop(-1)
    if a[-1] != b[-2] or output != expected:
        raise ValueError('Inconsistent MatMul shapes')
    return math.prod(output) * a[-1]


def analyze_model(path):
    model = onnx.load(path)
    if any(n.domain not in ('', 'ai.onnx') or n.op_type not in WEIGHTED | OTHER for n in model.graph.node):
        raise ValueError('Use a portable graph with supported standard operators; nested/fused graphs are not counted')
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    info = tensor_info(model)
    method = 'onnx.shape_inference'
    if any(not fixed(info.get(out, (None,))[0]) for n in model.graph.node for out in n.output):
        # Resolves exported Shape/Gather/Slice arithmetic from fixed input sizes.
        # This is graph analysis: no InferenceSession, tensor inference or video.
        from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
        model = SymbolicShapeInference.infer_shapes(model, auto_merge=True, verbose=0)
        info = tensor_info(model)
        method += ' + ORT symbolic shape inference'
    shapes = {name: value[0] for name, value in info.items()}
    rows = []
    for index, node in enumerate(model.graph.node):
        if any(not fixed(shapes.get(out)) for out in node.output):
            raise ValueError(f'Unresolved output shape: {node.name or node.op_type}')
        elements = sum(math.prod(shapes[out]) for out in node.output)
        output_bytes = sum(math.prod(shapes[out]) * np.dtype(onnx.helper.tensor_dtype_to_np_dtype(info[out][1])).itemsize
                           for out in node.output)
        rows.append({'index': index, 'node': node.name, 'op': node.op_type,
                     'output_shapes': [shapes[out] for out in node.output],
                     'macs': dense_macs(node, shapes) if node.op_type in WEIGHTED else 0,
                     'logical_output_elements': elements, 'logical_output_bytes': output_bytes})
    summary = {'filename': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'inputs': {v.name: shapes[v.name] for v in model.graph.input}, 'shape_inference': method,
               **summarize(rows), 'largest_mac_nodes': sorted(rows, key=lambda r: -r['macs'])[:12]}
    return model, rows, summary


def summarize(rows):
    macs = sum(r['macs'] for r in rows)
    auxiliary = Counter()
    for r in rows:
        if r['op'] not in WEIGHTED:
            auxiliary[r['op']] += r['logical_output_elements']
    return {'nodes': len(rows), 'operators': dict(Counter(r['op'] for r in rows)),
            'macs': macs, 'dot_product_flops_convention': 2 * macs,
            'logical_output_bytes_sum': sum(r['logical_output_bytes'] for r in rows),
            'largest_single_node_output_bytes': max((r['logical_output_bytes'] for r in rows), default=0),
            'non_dot_output_elements_by_operator': dict(auxiliary)}


def partition_pose(model, rows):
    producers = {name: i for i, node in enumerate(model.graph.node) for name in node.output}

    def ancestors(names, stop=()):
        found, pending = set(), list(names)
        while pending:
            index = producers.get(pending.pop())
            if index is None or index in found or index in stop:
                continue
            found.add(index)
            pending.extend(model.graph.node[index].input)
        return found

    final = model.graph.node[producers[model.graph.output[0].name]]
    if final.op_type != 'Concat' or len(final.input) != 3 or attributes(final).get('axis') != 1:
        raise ValueError('Expected archived box/score/keypoint Concat')
    shared = ancestors([final.input[1]])
    groups = {'shared_and_scores': shared}
    used = set(shared)
    for label, suffixes in zip(('fine_40', 'middle_20', 'coarse_10'), (('', '_6'), ('_1', '_7'), ('_2', '_8'))):
        names = [f'/model.23/Reshape{s}_output_0' for s in suffixes]
        if any(name not in producers for name in names):
            raise ValueError('Missing archived head output')
        indices = ancestors(names, shared)
        if not indices or used & indices:
            raise ValueError('Unexpected shared scale-head nodes')
        groups[label] = indices
        used.update(indices)
    groups['positive_decode'] = set(range(len(rows))) - used
    for name, indices in groups.items():
        for i in indices:
            rows[i]['execution_group'] = name
    return {name: summarize([rows[i] for i in sorted(indices)]) for name, indices in groups.items()}


def pose_call_macs(groups, bits):
    if len(bits) != 3 or any(c not in '01' for c in bits):
        raise ValueError('Scale occupancy must be a three-bit string')
    return (groups['shared_and_scores']['macs']
            + (groups['positive_decode']['macs'] if '1' in bits else 0)
            + sum(groups[name]['macs'] for name, bit in zip(('fine_40', 'middle_20', 'coarse_10'), bits) if bit == '1'))


def audit_costs(audit, models, groups):
    parity = {Path(v['video']).stem: v for v in audit['equivalence']['videos']}
    output = []
    for video in audit['dependency_audit']:
        name = Path(video['source']).stem
        record = parity[name]
        counts = record['model_calls']['optimized']
        if record['frames'] != video['frames'] or counts != record['model_calls']['reference']:
            raise ValueError('Audit and equivalence counts do not agree')
        stages = {}
        for reason, stage in video['network_stages'].items():
            histogram = stage['active_scale_histogram']
            if sum(histogram.values()) != stage['calls'] or any(n < 0 for n in histogram.values()):
                raise ValueError('Invalid scale histogram')
            old = sum(n * pose_call_macs(groups, '000' if bits == '000' else '111') for bits, n in histogram.items())
            new = sum(n * pose_call_macs(groups, bits) for bits, n in histogram.items())
            stages[reason] = {'calls': stage['calls'], 'whole_image_gate_macs': old, 'per_scale_gate_macs': new,
                              'saved_macs': old - new, 'active_scale_histogram': histogram}
        if sum(s['calls'] for s in stages.values()) != counts['face']:
            raise ValueError('Face call count mismatch')
        other = {name: counts[name] * models[name]['macs'] for name in ('body', 'landmarks')}
        before = sum(s['whole_image_gate_macs'] for s in stages.values()) + sum(other.values())
        after = sum(s['per_scale_gate_macs'] for s in stages.values()) + sum(other.values())
        output.append({'video': name, 'frames': video['frames'], 'calls': counts, 'face_stages': stages,
                       'other_network_macs': other, 'all_networks_before_macs': before,
                       'all_networks_after_macs': after, 'saved_network_mac_fraction': (before - after) / before,
                       'after_network_macs_per_frame': after / video['frames']})
    return output


def pixel_budgets(size, *, normalized):
    elements = size * size * 3
    return {'side': size, 'bgr_u8_payload_bytes': elements, 'nchw_fp32_payload_bytes': elements * 4,
            'pack_input_plus_output_payload_bytes': elements * 5,
            'bilinear_max_sample_contributions': elements * 4,
            'normalization_scalar_multiplies': elements if normalized else 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('face', 'landmarks', 'body'):
        parser.add_argument('--' + key, type=Path, required=True, help='Original portable ONNX checkpoint')
    parser.add_argument('--audit', type=Path, help='Existing candidate_optimization_benchmark.json; no replay')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--nodes-output', type=Path, help='Optional complete per-node ledger')
    args = parser.parse_args()
    models, ledgers, graphs = {}, {}, {}
    for key in ('face', 'landmarks', 'body'):
        graphs[key], ledgers[key], models[key] = analyze_model(getattr(args, key))
    if models['face']['sha256'] != POSE_SHA256:
        raise ValueError('Conditional cost decomposition requires the archived pose checkpoint')
    groups = partition_pose(graphs['face'], ledgers['face'])
    patterns = {format(i, '03b'): pose_call_macs(groups, format(i, '03b')) for i in range(8)}
    if sum(g['macs'] for g in groups.values()) != models['face']['macs']:
        raise AssertionError('Pose partition does not cover the complete graph')
    report = {'schema': 'dog_capture_static_compute_v1',
              'method': {'unit': 'One MAC is one multiply-accumulate in Conv/Gemm/MatMul; M=1e6, G=1e9.',
                         'execution': 'Static shape inference only. No model inference, video replay or timed benchmark.',
                         'flops': '2*MACs is a dot-product convention, excludes bias, activations, reductions and control.',
                         'memory': 'Logical graph tensor payloads, not DRAM traffic or peak resident memory; fusions and reuse change physical traffic.',
                         'limits': 'Logical dense operators, not padded CPU kernels, Winograd transforms, SIMD instructions or runtime latency.'},
              'models': models, 'pose_execution_groups': groups, 'pose_call_macs_by_scale_bits': patterns,
              'one_track_examples': {'tracking_only': models['landmarks']['macs'],
                 'full_frame_and_tracking_middle_scale': patterns['010'] + models['landmarks']['macs'],
                 'candidate_tracking_precise_verify_mirror_middle_scale': 3 * patterns['010'] + models['landmarks']['macs'],
                 'candidate_plus_full_frame_and_second_precise_middle_scale': 5 * patterns['010'] + models['landmarks']['macs'],
                 'five_full_heads_and_one_tracker': 5 * patterns['111'] + models['landmarks']['macs']},
              'pixel_work_units': [pixel_budgets(n, normalized=n in (128, 320)) for n in (128, 320, 416)],
              'pixel_work_assumptions': 'Archived face/landmark inputs multiply RGB by 1/255; archived YOLOX uses raw BGR float values.',
              'sources': ['https://onnx.ai/onnx/operators/onnx__Conv.html',
                          'https://onnx.ai/onnx/operators/onnx__MatMul.html',
                          'https://onnx.ai/onnx/operators/onnx__Gemm.html',
                          'https://onnx.ai/onnx/repo-docs/ShapeInference.html']}
    if args.audit:
        report['existing_audit_sha256'] = hashlib.sha256(args.audit.read_bytes()).hexdigest()
        report['existing_video_call_costs'] = audit_costs(json.loads(args.audit.read_text(encoding='utf8')), models, groups)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    if args.nodes_output:
        args.nodes_output.parent.mkdir(parents=True, exist_ok=True)
        args.nodes_output.write_text(json.dumps(ledgers, indent=2) + '\n', encoding='utf8')
    print(json.dumps({'model_macs': {k: v['macs'] for k, v in models.items()},
                      'pose_group_macs': {k: v['macs'] for k, v in groups.items()}, 'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
