"""Compile a device-local CPU pose graph with exact conditional box/keypoint heads.

Keep the original checkpoint. Run this on the deployment CPU with its ORT build;
the resulting graph is a CPU artifact, not a portable/mobile/GPU checkpoint.
Only compilation needs the onnx package. Inference still uses NumPy and ORT.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# This decoder's DFL/keypoint operations never mix anchor positions. The scale
# rewrite relies on that property, not merely on a YOLO-looking output shape.
SCALE_CHECKPOINT_SHA256 = '48c2b9f580a6b29f6ec92f81d58bffcb06c609b4da4d77e4c85f259dc1b8d74c'
SCALE_OUTPUT_CONTRACT = 'scores_and_eligible_anchors_exact'


def conditional_heads(model):
    """Move non-score ancestors behind If after the CPU fusions are fixed.

The same class scores and runtime threshold decide whether there can be any
decoded face. Empty results skip only unused heads. Nonempty results preserve
all original nodes, weights, raw anchors and NMS ordering.
"""
    model = copy.deepcopy(model)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError('Expected one image input and one raw YOLO pose output')
    metadata = {p.key: p.value for p in model.metadata_props}
    try:
        classes = len(ast.literal_eval(metadata['names']))
        keypoints = ast.literal_eval(metadata['kpt_shape'])
        width = 4 + classes + int(keypoints[0]) * int(keypoints[1])
    except (KeyError, ValueError, TypeError, SyntaxError, IndexError) as error:
        raise ValueError('Expected YOLO pose names and kpt_shape metadata') from error
    output = model.graph.output[0]
    dims = [d.dim_value for d in output.type.tensor_type.shape.dim]
    if (output.type.tensor_type.elem_type != TensorProto.FLOAT or len(dims) != 3
            or dims[0] != 1 or dims[1] != width or dims[2] <= 0):
        raise ValueError('Expected fixed raw pose output [1, 4+classes+keypoints, anchors]')
    producers = {name: i for i, node in enumerate(model.graph.node) for name in node.output}
    final = model.graph.node[producers[output.name]]
    axis = next((a.i for a in final.attribute if a.name == 'axis'), None)
    if final.op_type != 'Concat' or axis != 1 or len(final.input) != 3:
        raise ValueError('Expected separate box, score and keypoint inputs to the final Concat')
    shapes = {v.name: (v.type.tensor_type.elem_type,
                       [d.dim_value for d in v.type.tensor_type.shape.dim])
              for v in model.graph.value_info}
    if any(name not in shapes for name in final.input):
        inferred = onnx.shape_inference.infer_shapes(model)
        shapes.update({v.name: (v.type.tensor_type.elem_type,
                               [d.dim_value for d in v.type.tensor_type.shape.dim])
                       for v in inferred.graph.value_info})
    shapes.update({v.name: (v.data_type, list(v.dims)) for v in model.graph.initializer})
    expected = [4, classes, width - 4 - classes]
    if any(shapes.get(name) != (TensorProto.FLOAT, [1, channels, dims[2]])
           for name, channels in zip(final.input, expected)):
        raise ValueError('Final Concat inputs must be FP32 boxes, scores, keypoints in that order')
    names = set(producers) | {v.name for v in model.graph.input} | {v.name for v in model.graph.initializer}
    if any(name.startswith('capture_gate_') for name in names):
        raise ValueError('Graph is already conditional or uses reserved capture_gate_ names')
    needed = set()

    def require(name):
        if name not in producers or producers[name] in needed:
            return
        index = producers[name]
        needed.add(index)
        for source in model.graph.node[index].input:
            require(source)

    require(final.input[1])
    outside = [node for i, node in enumerate(model.graph.node) if i in needed]
    inside = [node for i, node in enumerate(model.graph.node) if i not in needed]
    threshold = 'capture_gate_threshold'
    empty_name = 'capture_gate_empty'
    empty = numpy_helper.from_array(np.empty((1, width, 0), np.float32))
    otherwise = helper.make_graph([helper.make_node('Constant', [], [empty_name], value=empty)],
                                  'empty_face_result', [],
                                  [helper.make_tensor_value_info(empty_name, TensorProto.FLOAT, [1, width, 0])])
    positive = helper.make_graph(inside, 'original_pose_heads', [], [copy.deepcopy(output)])
    gate = [helper.make_node('GreaterOrEqual', [final.input[1], threshold], ['capture_gate_passed']),
            helper.make_node('Cast', ['capture_gate_passed'], ['capture_gate_int'], to=TensorProto.INT64),
            helper.make_node('ReduceMax', ['capture_gate_int'], ['capture_gate_any'], keepdims=0),
            helper.make_node('Cast', ['capture_gate_any'], ['capture_gate_run'], to=TensorProto.BOOL),
            helper.make_node('If', ['capture_gate_run'], [output.name], name='capture_gate_heads',
                             then_branch=positive, else_branch=otherwise)]
    model.graph.ClearField('node')
    model.graph.node.extend(outside + gate)
    model.graph.ClearField('value_info')
    model.graph.input.append(helper.make_tensor_value_info(threshold, TensorProto.FLOAT, []))
    dim = model.graph.output[0].type.tensor_type.shape.dim[2]
    dim.ClearField('dim_value')
    dim.dim_param = 'anchors'
    metadata['dog_capture.conditional_score_input'] = threshold
    helper.set_model_props(model, metadata)
    onnx.checker.check_model(model)
    return model, {'conditional_score_input': threshold, 'head_nodes': len(inside),
                   'score_and_shared_nodes': len(outside), 'output_width': width}


def conditional_scale_heads(base):
    """Specialize the archived 320 px decoder; compile_config pins its source.

    Scores and all anchors reaching the supplied threshold remain exact. Only
    box/keypoint intermediates of a wholly rejected scale may be replaced by
    zeros; these anchors are discarded by the unchanged decoder before NMS.
    Keep the original operator fusions and the whole-image empty fast path.
    """
    nodes = list(base.graph.node)
    producers = {name: i for i, node in enumerate(nodes) for name in node.output}
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in base.graph.value_info}
    model, detail = conditional_heads(base)
    scores = nodes[producers[base.graph.output[0].name]].input[1]
    shared = set()

    def require(name, indices, stop=()):
        index = producers.get(name)
        if index is None or index in indices or index in stop:
            return
        indices.add(index)
        for source in nodes[index].input:
            require(source, indices, stop)

    require(scores, shared)
    positive = next(a.g for a in model.graph.node[-1].attribute if a.name == 'then_branch')
    original = list(positive.node)
    moved, gates, levels = set(), [], []
    boundaries = (0, 1600, 2000, 2100)
    for level, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        suffixes = (('', '_1', '_2')[level], ('_6', '_7', '_8')[level])
        outputs = [f'/model.23/Reshape{suffix}_output_0' for suffix in suffixes]
        if any(shapes.get(name) != [1, channels, end - start]
               for name, channels in zip(outputs, (64, 18))):
            raise ValueError('Unsupported archived pose head layout for scale compilation')
        needed = set()
        for name in outputs:
            require(name, needed, shared)
        if not needed or moved & needed:
            raise ValueError('Scale heads share unexpected intermediate nodes')
        internal = {name for i in needed for name in nodes[i].output} - set(outputs)
        if any(name in internal for i, node in enumerate(nodes) if i not in needed for name in node.input):
            raise ValueError('A scale head intermediate has an outside consumer')
        moved.update(needed)
        prefix = f'capture_scale_{level}_'
        for name, value in (('start', [start]), ('end', [end]), ('axis', [2])):
            model.graph.initializer.append(numpy_helper.from_array(np.asarray(value, np.int64), prefix + name))
        yes = helper.make_graph([node for i, node in enumerate(nodes) if i in needed], prefix + 'heads', [],
                                [helper.make_tensor_value_info(name, TensorProto.FLOAT, shapes[name]) for name in outputs])
        no = helper.make_graph([
            helper.make_node('Constant', [], [prefix + 'empty' + str(i)],
                             value=numpy_helper.from_array(np.zeros(shapes[name], np.float32)))
            for i, name in enumerate(outputs)], prefix + 'empty', [], [
                helper.make_tensor_value_info(prefix + 'empty' + str(i), TensorProto.FLOAT, shapes[name])
                for i, name in enumerate(outputs)])
        gates.extend([
            helper.make_node('Slice', [scores, prefix + 'start', prefix + 'end', prefix + 'axis'], [prefix + 'scores']),
            helper.make_node('GreaterOrEqual', [prefix + 'scores', detail['conditional_score_input']], [prefix + 'pass']),
            helper.make_node('Cast', [prefix + 'pass'], [prefix + 'int'], to=TensorProto.INT64),
            helper.make_node('ReduceMax', [prefix + 'int'], [prefix + 'any'], keepdims=0),
            helper.make_node('Cast', [prefix + 'any'], [prefix + 'run'], to=TensorProto.BOOL),
            helper.make_node('If', [prefix + 'run'], outputs, name=prefix + 'heads', then_branch=yes, else_branch=no)])
        levels.append({'anchors': [start, end], 'head_nodes': len(needed)})
    positive.ClearField('node')
    positive.node.extend(gates + [node for node in original if producers[node.output[0]] not in moved])
    metadata = {p.key: p.value for p in model.metadata_props}
    metadata['dog_capture.raw_output_contract'] = SCALE_OUTPUT_CONTRACT
    helper.set_model_props(model, metadata)
    onnx.checker.check_model(model)
    return model, {**detail, 'conditional_scales': levels, 'raw_output_contract': SCALE_OUTPUT_CONTRACT}


def compile_config(config, output, *, per_scale=False):
    import onnxruntime as ort
    from dog_capture.engine.runtime import _session_options, cpu_model_target
    from dog_capture.engine.zoo import find_face_model, sha256, verify_model
    kind, source = find_face_model(config)
    if kind != 'yolo_pose':
        raise ValueError('CPU conditional compilation currently supports YOLO pose only')
    original_sha, sidecar = verify_model(source)
    if sidecar.get('cpu_compiled'):
        raise ValueError('Compile from the original checkpoint, not an already compiled graph')
    if per_scale and original_sha != SCALE_CHECKPOINT_SHA256:
        raise ValueError('Per-scale compilation supports only the archived dog_face_pose checkpoint; '
                         'other models can use whole-image conditional compilation')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    base = output / 'base_cpu.onnx'
    r = config.runtime
    settings = _session_options(ort, r.threads, r.thread_spinning, r.spin_duration_us, r.spin_backoff_max)
    settings.optimized_model_filepath = str(base)
    optimizer = ort.InferenceSession(str(source), settings, providers=['CPUExecutionProvider'])
    del optimizer
    compiled, detail = (conditional_scale_heads if per_scale else conditional_heads)(onnx.load(base))
    path = output / 'dog_face_pose_cpu.onnx'
    onnx.save(compiled, path)
    profile = {'version': 1, 'target': cpu_model_target(), 'source_sha256': original_sha, **detail}
    metadata = {**sidecar, 'kind': 'yolo_pose', 'sha256': sha256(path), 'cpu_compiled': profile}
    path.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf8')
    optimized = config.merged({'detector': {'model': str(path)},
                               'runtime': {'providers': 'cpu', 'cuda_graph': False}})
    config_path = output / 'capture.json'
    config_path.write_text(json.dumps(optimized.to_dict(), indent=2), encoding='utf8')
    return config_path, profile


def main():
    from dog_capture.engine.config import CaptureConfig
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New local model directory')
    parser.add_argument('--per-scale', action='store_true',
                        help='Also skip rejected scales of the archived 320 px pose checkpoint')
    args = parser.parse_args()
    path, profile = compile_config(CaptureConfig.load(args.config), args.output, per_scale=args.per_scale)
    print(json.dumps({'config': str(path), 'cpu_compiled': profile}, indent=2))


if __name__ == '__main__':
    main()
