"""Reproducible pinned TFLite -> ONNX -> TensorRT build, outside the hot path."""
from __future__ import annotations

import json
from pathlib import Path
import platform
import subprocess
import time
import zipfile

import numpy as np

from .landmarks import confidence_model
from .models import ASSETS, ensure_models, file_hash


def detector_intermediate(source, directory):
    import flatbuffers
    from flatbuffers.flexbuffers import Loads
    from ai_edge_litert import schema_py_generated as schema
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(Path(source).read_bytes(), 0))
    graph = model.subgraphs[0]
    op = graph.operators[-1]
    if model.operatorCodes[op.opcodeIndex].customCode != b'TFLite_Detection_PostProcess':
        raise ValueError('Unsupported detection graph')
    options = Loads(bytes(op.customOptions))
    if options['use_regular_nms'] or options['max_classes_per_detection'] != 1:
        raise ValueError('Expected single-class fast NMS contract')
    anchor_tensor = graph.tensors[op.inputs[2]]
    anchors = np.frombuffer(model.buffers[anchor_tensor.buffer].data.tobytes(), np.float32).reshape(anchor_tensor.shape)
    np.save(directory / 'detector_anchors.npy', anchors)
    input_tensor = graph.tensors[graph.inputs[0]]
    with zipfile.ZipFile(source) as labels:
        dog_class = labels.read('labelmap.txt').decode().splitlines().index('dog')
    contract = {'source_sha256': file_hash(source), 'input_domain': 'tflite_real',
                'scale': float(input_tensor.quantization.scale[0]),
                'zero_point': int(input_tensor.quantization.zeroPoint[0]),
                'iou_threshold': float(options['nms_iou_threshold']), 'max_detections': int(options['max_detections']),
                'dog_class': dog_class,
                'box_scales': {key: float(options[key]) for key in ('y_scale', 'x_scale', 'h_scale', 'w_scale')},
                'note': 'Float network from dequantized pretrained INT8 weights; activation quantization removed. Revalidate detections.'}
    (directory / 'detector_contract.json').write_text(json.dumps(contract, indent=2), encoding='utf8')
    graph.outputs = list(op.inputs[:2])
    graph.operators = graph.operators[:-1]
    model.signatureDefs = []
    builder = flatbuffers.Builder(0)
    builder.Finish(model.Pack(builder), file_identifier=b'TFL3')
    path = directory / 'detector_raw.tflite'
    path.write_bytes(bytes(builder.Output()))
    return path


def rename_tensor(model, old, new):
    for node in model.graph.node:
        for names in (node.input, node.output):
            for i, name in enumerate(names):
                if name == old:
                    names[i] = new
    for values in (model.graph.input, model.graph.output, model.graph.value_info, model.graph.initializer):
        for value in values:
            if value.name == old:
                value.name = new


def compact_detector_onnx(model, directory):
    """Decode boxes and collapse 90 class channels on GPU, retain all anchors."""
    import onnx
    from onnx import helper as h, numpy_helper as nh, TensorProto as T
    boxes = next(o.name for o in model.graph.output if o.type.tensor_type.shape.dim[-1].dim_value == 4)
    scores = next(o.name for o in model.graph.output if o.type.tensor_type.shape.dim[-1].dim_value == 90)
    anchors = np.load(directory / 'detector_anchors.npy')
    contract = json.loads((directory / 'detector_contract.json').read_text(encoding='utf8'))
    scale = contract['box_scales']
    constants = {'dc_anchor_xy': anchors[None, :, :2], 'dc_anchor_hw': anchors[None, :, 2:],
                 'dc_xy_scales': np.array([scale['y_scale'], scale['x_scale']], np.float32),
                 'dc_hw_scales': np.array([scale['h_scale'], scale['w_scale']], np.float32),
                 'dc_half': np.array(.5, np.float32),
                 'dc_s0': np.array([0], np.int64), 'dc_s2': np.array([2], np.int64),
                 'dc_s4': np.array([4], np.int64), 'dc_axes2': np.array([2], np.int64)}
    model.graph.initializer.extend(nh.from_array(value, name) for name, value in constants.items())
    nodes = [h.make_node('Slice', [boxes, 'dc_s0', 'dc_s2', 'dc_axes2'], ['dc_raw_xy']),
             h.make_node('Slice', [boxes, 'dc_s2', 'dc_s4', 'dc_axes2'], ['dc_raw_hw']),
             h.make_node('Div', ['dc_raw_xy', 'dc_xy_scales'], ['dc_scaled_xy']),
             h.make_node('Div', ['dc_raw_hw', 'dc_hw_scales'], ['dc_scaled_hw']),
             h.make_node('Mul', ['dc_scaled_xy', 'dc_anchor_hw'], ['dc_offset']),
             h.make_node('Add', ['dc_offset', 'dc_anchor_xy'], ['dc_center']),
             h.make_node('Exp', ['dc_scaled_hw'], ['dc_exp']),
             h.make_node('Mul', ['dc_exp', 'dc_anchor_hw'], ['dc_size']),
             h.make_node('Mul', ['dc_size', 'dc_half'], ['dc_radius']),
             h.make_node('Sub', ['dc_center', 'dc_radius'], ['dc_low']),
             h.make_node('Add', ['dc_center', 'dc_radius'], ['dc_high']),
             h.make_node('Concat', ['dc_low', 'dc_high'], ['detection_boxes'], axis=2),
             h.make_node('ReduceMax', [scores], ['detection_scores'], axes=[2], keepdims=0),
             h.make_node('ArgMax', [scores], ['dc_classes64'], axis=2, keepdims=0, select_last_index=0),
             h.make_node('Cast', ['dc_classes64'], ['detection_classes'], to=T.INT32)]
    for index, node in enumerate(nodes):
        node.name = f'dc_postprocess_{index}_{node.op_type}'
    model.graph.node.extend(nodes)
    del model.graph.output[:]
    count = len(anchors)
    model.graph.output.extend([h.make_tensor_value_info('detection_boxes', T.FLOAT, [1, count, 4]),
                               h.make_tensor_value_info('detection_scores', T.FLOAT, [1, count]),
                               h.make_tensor_value_info('detection_classes', T.INT32, [1, count])])
    if model.graph.input[0].type.tensor_type.elem_type == T.UINT8:
        # tf2onnx preserves uint8 image IO and inserts Cast -> (x-zp)*scale.
        # Accept raw-valued float pixels instead: TensorRT can fuse the existing
        # preprocessing without needing a uint8 network binding. Do not apply
        # normalization a second time in the host adapter.
        model.graph.input[0].type.tensor_type.elem_type = T.FLOAT
        contract['input_domain'] = 'pixel_0_255'
        (directory / 'detector_contract.json').write_text(json.dumps(contract, indent=2), encoding='utf8')
    elif model.graph.input[0].type.tensor_type.elem_type != T.FLOAT:
        raise ValueError('Unsupported dequantized detector input type')
    return onnx.shape_inference.infer_shapes(model)


def contiguous_landmark_head(model):
    """Equivalent spatial softmax/expectation, retaining contiguous NCHW axes.

    The TFLite conversion makes a long *strided* spatial softmax on NHWC and
    transposes back and forth. Keep each landmark's spatial plane contiguous.
    Reuse the trained graph's float16-rounded coordinate grid exactly.
    """
    import onnx
    from onnx import helper as h, numpy_helper as nh
    nodes = list(model.graph.node)
    heat = next(n for n in nodes if n.op_type == 'Conv' and '/heatmap_conv/' in n.name)
    weights = {v.name: nh.to_array(v) for v in model.graph.initializer}
    xmul = next(n for n in nodes if n.name.endswith('soft_argmax/mul_1'))
    ymul = next(n for n in nodes if n.name.endswith('soft_argmax/mul_2'))
    xgrid = next(weights[v] for v in xmul.input if v in weights).reshape(-1)
    ygrid = next(weights[v] for v in ymul.input if v in weights).reshape(-1)
    if xgrid.shape != (192,) or ygrid.shape != (192,):
        raise ValueError('Unexpected landmark coordinate grid')
    prefix = 'soft_argmax/contiguous/'
    constants = {'flat_shape': np.array([1, 46, 192 * 192], np.int64),
                 'map_shape': np.array([1, 46, 192, 192], np.int64),
                 'out_shape': np.array([1, 92], np.int64),
                 'xy_axes': np.array([2, 3], np.int64), 'flat_axis': np.array([2], np.int64),
                 'x_grid': xgrid.reshape(1, 1, 1, 192), 'y_grid': ygrid.reshape(1, 1, 192, 1),
                 'epsilon': np.array(1e-6, np.float32)}
    model.graph.initializer.extend(nh.from_array(v, prefix + k) for k, v in constants.items())
    tail = []
    def add(op, inputs, output, **attrs):
        full_inputs = [heat.output[0] if name == 'heat' else prefix + name for name in inputs]
        full_output = output if output in ('landmarks', 'landmark_entropy') else prefix + output
        tail.append(h.make_node(op, full_inputs, [full_output], name=prefix + output, **attrs))
    add('Reshape', ['heat', 'flat_shape'], 'flat')
    add('Softmax', ['flat'], 'probabilities', axis=2)
    add('Reshape', ['probabilities', 'map_shape'], 'maps')
    add('Mul', ['maps', 'x_grid'], 'weighted_x')
    add('Mul', ['maps', 'y_grid'], 'weighted_y')
    add('ReduceSum', ['weighted_x', 'xy_axes'], 'x', keepdims=0)
    add('ReduceSum', ['weighted_y', 'xy_axes'], 'y', keepdims=0)
    add('Unsqueeze', ['x', 'flat_axis'], 'x_column')
    add('Unsqueeze', ['y', 'flat_axis'], 'y_column')
    add('Concat', ['x_column', 'y_column'], 'xy', axis=2)
    add('Reshape', ['xy', 'out_shape'], 'landmarks')
    add('Max', ['probabilities', 'epsilon'], 'clipped')
    add('Log', ['clipped'], 'log')
    add('Mul', ['probabilities', 'log'], 'weighted_log')
    add('ReduceSum', ['weighted_log', 'flat_axis'], 'negative_entropy', keepdims=0)
    add('Neg', ['negative_entropy'], 'landmark_entropy')
    kept = nodes[:nodes.index(heat) + 1] + tail
    del model.graph.node[:]
    model.graph.node.extend(kept)
    used = {name for node in kept for name in node.input}
    initializers = [v for v in model.graph.initializer if v.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)
    del model.graph.value_info[:]
    result = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(result)
    return result


def export_onnx(source_directory, destination, export_python):
    import onnx
    source_directory = ensure_models(source_directory)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    landmark = destination / 'landmarks_compact.tflite'
    landmark.write_bytes(confidence_model(source_directory / 'dog_face_landmarks_full.tflite', compact=True))
    detector = detector_intermediate(source_directory / 'efficientdet_lite0_nms.tflite', destination)
    inputs = {'localizer': source_directory / 'dog_face_localizer.tflite', 'landmarks': landmark, 'detector': detector}
    records = {}
    for name, source in inputs.items():
        temporary = destination / (name + '.raw.onnx')
        command = [str(export_python), '-m', 'tf2onnx.convert', '--tflite', str(source),
                   '--opset', '17', '--output', str(temporary)]
        if name == 'detector':
            command.append('--dequantize')
        print('Exporting', name, flush=True)
        with (destination / (name + '.export.log')).open('w', encoding='utf8') as log:
            subprocess.run(command, stdout=log, stderr=log, check=True)
        model = onnx.load(temporary)
        rename_tensor(model, model.graph.input[0].name, 'images')
        if name == 'detector':
            model = compact_detector_onnx(model, destination)
        else:
            output_names = ['face_box'] if name == 'localizer' else ['landmarks', 'landmark_entropy']
            for output, new_name in zip(list(model.graph.output), output_names):
                rename_tensor(model, output.name, new_name)
            if name == 'landmarks':
                model = contiguous_landmark_head(model)
        model.ir_version = min(model.ir_version, 10)
        onnx.checker.check_model(model)
        path = destination / (name + '.onnx')
        onnx.save(model, path)
        records[name] = {'onnx_sha256': file_hash(path), 'bytes': path.stat().st_size,
                         'input_names': [x.name for x in model.graph.input],
                         'output_names': [x.name for x in model.graph.output],
                         'source_tflite_sha256': file_hash(source)}
    report = {'models': records, 'original_model_sha256': {name: value[1] for name, value in ASSETS.items()},
              'onnx_opset': 17, 'detector_activation_quantization_removed': True,
              'landmark_head_layout': 'NCHW contiguous spatial softmax; original rounded coordinate grids; FP32 reductions',
              'note': 'Face coordinates and confidence retain the trained graph. Detector float export requires semantic regression, not bitwise equivalence.'}
    (destination / 'export.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    return report


def build_engines(directory, *, precision='fp16', workspace_mb=512, optimization_level=3,
                  engine_names=('localizer', 'landmarks', 'detector')):
    from .trt import load_cuda_dlls, cuda_check
    load_cuda_dlls()
    import tensorrt as trt
    import onnx
    from cuda import cuda
    directory = Path(directory)
    cuda_check(cuda.cuInit(0))
    device = cuda_check(cuda.cuDeviceGet(0))
    major = cuda_check(cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device))
    minor = cuda_check(cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device))
    gpu_name = cuda_check(cuda.cuDeviceGetName(256, device)).split(b'\0', 1)[0].decode()
    logger = trt.Logger(trt.Logger.INFO)
    records = {name: json.loads((directory / (name + '.json')).read_text(encoding='utf8'))
               for name in ('localizer', 'landmarks', 'detector') if (directory / (name + '.json')).exists()}
    for name in engine_names:
        started = time.perf_counter()
        source = directory / (name + '.onnx')
        builder = trt.Builder(logger)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse(source.read_bytes()):
            errors = '\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f'TensorRT cannot parse {source}:\n{errors}')
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024)
        config.builder_optimization_level = optimization_level
        config.clear_flag(trt.BuilderFlag.TF32)
        protected = []
        if precision == 'fp16':
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            safe_types = {trt.LayerType.SOFTMAX, trt.LayerType.REDUCE, trt.LayerType.UNARY,
                          trt.LayerType.ELEMENTWISE, trt.LayerType.ACTIVATION}
            for i in range(network.num_layers):
                layer = network.get_layer(i)
                names = [layer.name] + [layer.get_output(j).name for j in range(layer.num_outputs)]
                if layer.type in safe_types and any('soft_argmax' in x or 'confidence/' in x for x in names):
                    layer.precision = trt.float32
                    for j in range(layer.num_outputs):
                        if layer.get_output(j).dtype == trt.float32:
                            layer.set_output_type(j, trt.float32)
                    protected.append(layer.name)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        cache_path = directory / (name + '.timing.cache')
        cache = config.create_timing_cache(cache_path.read_bytes() if cache_path.exists() else b'')
        config.set_timing_cache(cache, ignore_mismatch=False)
        print(f'Building {name} ({precision}, SM {major}.{minor})', flush=True)
        engine = builder.build_serialized_network(network, config)
        if engine is None:
            raise RuntimeError(f'TensorRT build failed: {name}')
        path = directory / (name + '.engine')
        path.write_bytes(bytes(engine))
        cache_path.write_bytes(bytes(config.get_timing_cache().serialize()))
        graph = onnx.load(source)
        record = {'tensorrt_version': trt.__version__, 'precision': precision, 'compute_capability': [major, minor],
                  'gpu_name': gpu_name, 'platform': platform.system(),
                  'onnx_sha256': file_hash(source), 'engine_sha256': file_hash(path),
                  'engine_bytes': path.stat().st_size, 'build_seconds': time.perf_counter() - started,
                  'output_names': [x.name for x in graph.graph.output], 'protected_fp32_layers': protected,
                  'workspace_mb': workspace_mb, 'builder_optimization_level': optimization_level}
        if name == 'detector':
            record['contract_sha256'] = file_hash(directory / 'detector_contract.json')
        path.with_suffix('.json').write_text(json.dumps(record, indent=2), encoding='utf8')
        records[name] = record
    (directory / 'build.json').write_text(json.dumps(records, indent=2), encoding='utf8')
    return records
