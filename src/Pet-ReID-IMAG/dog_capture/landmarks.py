"""Expose the existing heatmap's concentration; no retraining or new weights.

Entropy is an uncertainty heuristic, not a calibrated visibility probability.
"""
from pathlib import Path

import cv2
import numpy as np

from .geometry import EYE_A, EYE_B, estimate_pose


def confidence_model(base, compact=False):
    """Add confidence without changing coordinates or pretrained weights.

    Compact mode reduces heatmap entropy inside the graph: 46 float outputs
    instead of copying 46 x 192 x 192 probabilities to the application.
    This is an output/transfer optimization, not a smaller neural backbone.
    """
    import flatbuffers
    from ai_edge_litert import schema_py_generated as schema
    base = Path(base)
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(base.read_bytes(), 0))
    graph = model.subgraphs[0]
    candidates = [i for i, t in enumerate(graph.tensors)
                  if t.name.decode().endswith('soft_argmax/Softmax')]
    if len(candidates) != 1:
        raise ValueError('Unsupported landmark graph; expected one heatmap softmax')
    output = candidates[0]
    if compact:
        def tensor(name, shape, dtype=schema.TensorType.FLOAT32, data=None):
            item = schema.TensorT()
            item.name, item.shape, item.type = name.encode(), np.array(shape, np.int32), dtype
            if data is not None:
                buffer = schema.BufferT()
                buffer.data = np.frombuffer(data, np.uint8).copy()
                item.buffer = len(model.buffers)
                model.buffers.append(buffer)
            graph.tensors.append(item)
            return len(graph.tensors) - 1

        def operator(code, inputs, outputs, options=None, options_type=0):
            matches = [i for i, c in enumerate(model.operatorCodes) if c.builtinCode == code]
            if matches:
                index = matches[0]
            else:
                entry = schema.OperatorCodeT()
                entry.builtinCode = code
                entry.deprecatedBuiltinCode = min(code, 127)
                entry.version = 1
                index = len(model.operatorCodes)
                model.operatorCodes.append(entry)
            op = schema.OperatorT()
            op.opcodeIndex, op.inputs, op.outputs = index, inputs, outputs
            op.builtinOptionsType, op.builtinOptions = options_type, options
            graph.operators.append(op)

        shape = list(graph.tensors[output].shape)
        # 1e-12 underflows on fp16 GPU delegates and makes 0 * log(0) NaN.
        # 1e-6 is representable; maximum per-point entropy bias is < 0.014.
        epsilon = tensor('confidence/epsilon', [1], data=np.array([1e-6], np.float32).tobytes())
        axis = tensor('confidence/axis', [1], schema.TensorType.INT32, np.array([2], np.int32).tobytes())
        clipped = tensor('confidence/clipped', shape)
        logged = tensor('confidence/log', shape)
        weighted = tensor('confidence/weighted', shape)
        reduced = tensor('confidence/negative_entropy', shape[:2])
        entropy = tensor('confidence/entropy', shape[:2])
        operator(schema.BuiltinOperator.MAXIMUM, [output, epsilon], [clipped])
        operator(schema.BuiltinOperator.LOG, [clipped], [logged])
        operator(schema.BuiltinOperator.MUL, [output, logged], [weighted], schema.MulOptionsT(), schema.BuiltinOptions.MulOptions)
        reduction = schema.ReducerOptionsT()
        reduction.keepDims = False
        operator(schema.BuiltinOperator.SUM, [weighted, axis], [reduced], reduction, schema.BuiltinOptions.ReducerOptions)
        operator(schema.BuiltinOperator.NEG, [reduced], [entropy])
        output = entropy
    graph.outputs = list(graph.outputs) + [output]
    for signature in model.signatureDefs or []:
        item = schema.TensorMapT()
        item.name = b'landmark_entropy' if compact else b'heatmap_probabilities'
        item.tensorIndex = output
        signature.outputs = list(signature.outputs) + [item]
    builder = flatbuffers.Builder(0)
    offset = model.Pack(builder)
    builder.Finish(offset, file_identifier=b'TFL3')
    return bytes(builder.Output())


def rotate_normalized(points, quarter_turns):
    """Map points in the original crop to an np.rot90-normalized crop."""
    points = np.asarray(points, dtype=np.float32).copy()
    for _ in range(quarter_turns % 4):
        points = np.column_stack((points[:, 1], 1 - points[:, 0]))
    return points


def canonical_points(points, box, quarter_turns):
    x1, y1, x2, y2 = box
    normalized = (points - [x1, y1]) / [x2 - x1, y2 - y1]
    canonical = rotate_normalized(normalized, quarter_turns)
    width, height = x2 - x1, y2 - y1
    if quarter_turns % 2:
        width, height = height, width
    return canonical * [width, height], (0, 0, width, height)


def pose_in_crop(points, box, quarter_turns):
    canonical, shape = canonical_points(points, box, quarter_turns)
    return estimate_pose(canonical, shape, (shape[3], shape[2], 3))


def uncertainty(heatmaps):
    if heatmaps.size == 46:
        entropy = heatmaps.reshape(46)
    else:
        probabilities = heatmaps.reshape(46, -1)
        entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)), axis=1)
    return {'eye_entropy': float(max(entropy[EYE_A].mean(), entropy[EYE_B].mean())),
            'nose_entropy': float(entropy[24:32].mean()),
            'core_entropy': float(entropy[16:32].mean())}


def landmark_gate(points, box, quarter_turns, confidence):
    pose = pose_in_crop(points, box, quarter_turns)
    if not all(np.isfinite(confidence[key]) for key in ('eye_entropy', 'nose_entropy', 'core_entropy')):
        return False, 'nonfinite_confidence', pose
    if confidence['eye_entropy'] > 4.2 or confidence['nose_entropy'] > 4.5:
        return False, 'uncertain_landmarks', pose
    if not pose.valid:
        return False, pose.reason, pose
    # A profile/occluded eye can make the regressor collapse both eyes around
    # one texture feature. Do not promote those nearly flat triangles to front.
    crop_width = box[3] - box[1] if quarter_turns % 2 else box[2] - box[0]
    if abs(pose.yaw_cue) < .20 and (pose.nose_height < .25 or
                                  pose.eye_distance < .20 * crop_width):
        return False, 'front_geometry_unresolved', pose
    return True, 'ok', pose


def eye_evidence(frame, points):
    """Reject a confidently regressed eye sitting on featureless bright fur.

    This is a conservative image check, not a learned visibility classifier.
    It can reject unusual eye coloration/illumination; diagnostics are emitted.
    """
    evidence = []
    for indices in (EYE_A, EYE_B):
        eye = points[indices]
        center = eye.mean(0)
        radius = max(2, float(np.linalg.norm(eye[0] - eye[2])) * .22)
        x1, y1 = np.maximum(0, np.floor(center - radius).astype(int))
        x2, y2 = np.minimum([frame.shape[1], frame.shape[0]], np.ceil(center + radius + 1).astype(int))
        patch = frame[y1:y2, x1:x2]
        # Grayscale conversion is pixel-local. Crop first so each orientation
        # checks only the two eye patches, retaining exactly the same pixels.
        if patch.size:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        evidence.append(float(np.percentile(patch, 20)) if patch.size else 255.0)
    return {'eye_dark_p20': evidence, 'accepted': max(evidence) <= 105}
