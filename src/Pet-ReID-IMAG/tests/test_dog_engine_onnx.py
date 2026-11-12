"""Decoder and ONNX Runtime tests using tiny constant-output ONNX graphs."""
import json

import numpy as np
import pytest

onnx = pytest.importorskip('onnx')
pytest.importorskip('onnxruntime')
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from dog_capture.engine import CaptureConfig, CaptureEngine  # noqa: E402
from dog_capture.engine import detectors  # noqa: E402
from dog_capture.engine.detectors import (AnyFaceDetector, OnnxDetector, ScrfdDetector,  # noqa: E402
                                          YoloPoseDetector, YoloxBodyGate, letterbox)
from dog_capture.engine.runtime import create_session, resolve_providers  # noqa: E402
from dog_engine_synthetic import render  # noqa: E402


def constant_model(path, input_name, size, outputs, metadata=None):
    """Graph whose outputs are fixed arrays (+ 0 * mean(input) so the input is used)."""
    nodes = [helper.make_node('ReduceMean', [input_name], ['mean'], axes=[1, 2, 3], keepdims=0),
             helper.make_node('Mul', ['mean', 'zero'], ['nothing'])]
    initializers = [numpy_helper.from_array(np.zeros(1, np.float32), 'zero')]
    graph_outputs = []
    for i, (name, value) in enumerate(outputs):
        value = np.asarray(value, np.float32)
        initializers.append(numpy_helper.from_array(value, f'const{i}'))
        nodes.append(helper.make_node('Add', [f'const{i}', 'nothing'], [name]))
        graph_outputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, list(value.shape)))
    graph = helper.make_graph(nodes, 'constant', [helper.make_tensor_value_info(
        input_name, TensorProto.FLOAT, [1, 3, size, size])], graph_outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
    model.ir_version = 8
    for key, value in (metadata or {}).items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(model, str(path))
    return path


FRAME = np.zeros((480, 640, 3), np.uint8)  # letterbox to 320: ratio 0.5, pad_y 40 (centred)


def to_frame(points, pad=(0, 40), ratio=0.5):
    return (np.asarray(points, np.float64) - pad) / ratio


def test_letterbox_geometry():
    canvas, ratio, pad = letterbox(FRAME, 320)
    assert canvas.shape == (320, 320, 3) and ratio == .5 and pad == (0, 40)
    _, _, pad = letterbox(FRAME, 320, center=False)
    assert pad == (0, 0)


@pytest.mark.parametrize('scale,mean', [(1 / 255.0, 0.0), (1 / 128.0, 127.5)])
@pytest.mark.parametrize('layout', ['contiguous', 'mirror', 'quarter_turn', 'strided'])
def test_rgb_tensor_preserves_all_pixel_values_and_nchw_layout(scale, mean, layout):
    # Exhaust the uint8 range in distinct B/G/R channels. Mirrored/rotated ROI
    # views can have negative strides; the model must still receive packed RGB.
    values = np.arange(256, dtype=np.uint8)
    image = np.tile(np.stack([values, values[::-1], np.roll(values, 37)], -1)[None], (5, 1, 1))
    if layout == 'mirror':
        image = image[:, ::-1]
    elif layout == 'quarter_turn':
        image = np.rot90(image)
    elif layout == 'strided':
        image = np.repeat(image, 2, axis=1)[:, ::2]
    before = image.copy()
    # Reference is the previous deployed FP32 preprocessing, including its
    # rounding order. Input equality matters at confidence/view thresholds.
    import cv2
    expected = ((cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) - mean)
                * scale).transpose(2, 0, 1)[None]
    actual = detectors._rgb_tensor(image, scale, mean)
    assert actual.dtype == np.float32 and actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(image, before)


def test_yolo_pose_decoder(tmp_path):
    k, n = 6, 10
    out = np.zeros((1, 4 + 1 + 3 * k, n), np.float32)
    kpts = np.array([[140, 150], [180, 150], [160, 175], [160, 195], [125, 125], [195, 125]], np.float32)
    out[0, :5, 3] = [160, 160, 100, 80, .9]
    out[0, 5:, 3] = np.c_[kpts, [.9, .2, .8, .7, .6, .5]].reshape(-1)
    out[0, :5, 5] = [162, 160, 98, 80, .8]  # duplicate, removed by NMS
    model = constant_model(tmp_path / 'pose.onnx', 'images', 320, [('output0', out)],
                           {'kpt_shape': '[6, 3]', 'names': "{0: 'dog_face'}"})
    dets = YoloPoseDetector(model, providers='cpu').detect(FRAME)
    assert len(dets) == 1 and dets[0].schema.name == 'dogface6'
    assert np.allclose(dets[0].box, [220, 160, 420, 320], atol=1e-3)
    assert np.allclose(dets[0].keypoints, to_frame(kpts), atol=1e-3)
    assert np.allclose(dets[0].kp_scores, [.9, .2, .8, .7, .6, .5], atol=1e-6)


def test_yolo_pose_end_to_end_export_layout(tmp_path):
    k = 5
    out = np.zeros((1, 300, 6 + 3 * k), np.float32)
    kpts = np.array([[140, 150], [180, 150], [160, 175], [145, 190], [175, 190]], np.float32)
    out[0, 0, :6] = [110, 120, 210, 200, .9, 0]
    out[0, 0, 6:] = np.c_[kpts, np.ones(5)].reshape(-1)
    model = constant_model(tmp_path / 'e2e.onnx', 'images', 320, [('output0', out)], {'kpt_shape': '[5, 3]'})
    dets = YoloPoseDetector(model, providers='cpu').detect(FRAME)
    assert len(dets) == 1 and dets[0].schema.name == 'anyface5'
    assert np.allclose(dets[0].box, [220, 160, 420, 320], atol=1e-3)


def test_anyface_decoder(tmp_path):
    out = np.zeros((1, 20, 16), np.float32)
    lmk = np.array([[140, 150], [180, 150], [160, 175], [145, 190], [175, 190]], np.float32)
    out[0, 7, :5] = [160, 160, 100, 80, .95]
    out[0, 7, 5:15] = lmk.reshape(-1)
    out[0, 7, 15] = .9
    out[0, 2, :5] = [60, 60, 30, 30, .9]
    out[0, 2, 15] = .1  # obj * cls = .09 < threshold
    model = constant_model(tmp_path / 'anyface.onnx', 'input', 320, [('output', out)])
    dets = AnyFaceDetector(model, providers='cpu').detect(FRAME)
    assert len(dets) == 1 and dets[0].score == pytest.approx(.855)
    assert np.allclose(dets[0].box, [220, 160, 420, 320], atol=1e-3)
    assert np.allclose(dets[0].keypoints, to_frame(lmk), atol=1e-3)


def test_scrfd_decoder(tmp_path):
    size = 320
    outputs = {}
    for stride in (8, 16, 32):
        count = (size // stride) ** 2 * 2
        outputs[f'score_{stride}'] = np.zeros((1, count, 1), np.float32)
        outputs[f'bbox_{stride}'] = np.zeros((1, count, 4), np.float32)
        outputs[f'kps_{stride}'] = np.zeros((1, count, 10), np.float32)
    stride, gx, gy = 16, 10, 8
    index = (gy * (size // stride) + gx) * 2 + 1  # second anchor of cell (10, 8)
    centre = np.array([gx * stride, gy * stride], np.float32)  # (160, 128)
    outputs['score_16'][0, index, 0] = .88
    outputs['bbox_16'][0, index] = np.array([50, 40, 50, 40]) / stride
    lmk = np.array([[-20, -10], [20, -10], [0, 15], [-15, 30], [15, 30]], np.float32)
    outputs['kps_16'][0, index] = (lmk / stride).reshape(-1)
    order = [f'{kind}_{s}' for kind in ('score', 'bbox', 'kps') for s in (8, 16, 32)]
    model = constant_model(tmp_path / 'scrfd.onnx', 'input.1', size, [(n, outputs[n]) for n in order])
    dets = ScrfdDetector(model, providers='cpu').detect(FRAME)
    assert len(dets) == 1 and dets[0].score == pytest.approx(.88)
    # SCRFD letterbox is top-left aligned: frame = canvas / 0.5
    assert np.allclose(dets[0].box, np.r_[centre - [50, 40], centre + [50, 40]] / .5, atol=1e-3)
    assert np.allclose(dets[0].keypoints, (centre + lmk) / .5, atol=1e-3)


def test_yolox_gate_decoder_keeps_only_winning_dog_class(tmp_path):
    out = np.zeros((1, 3549, 85), np.float32)
    cell = 12 * 52 + 10  # stride 8, grid (10, 12)
    out[0, cell, :5] = [.5, .5, np.log(100 / 8), np.log(60 / 8), .9]
    out[0, cell, 5 + 16] = .95
    other = 13 * 52 + 30
    out[0, other, :5] = [.5, .5, np.log(80 / 8), np.log(80 / 8), .9]
    out[0, other, 5 + 16], out[0, other, 5 + 0] = .5, .9  # "person" wins: not a dog
    model = constant_model(tmp_path / 'yolox.onnx', 'images', 416, [('output', out)])
    frame = np.zeros((416, 832, 3), np.uint8)  # ratio 0.5, top-left padding
    boxes, scores = YoloxBodyGate(model, providers='cpu').detect(frame)
    assert len(boxes) == 1 and scores[0] == pytest.approx(.855)
    cx, cy = 10.5 * 8 / .5, 12.5 * 8 / .5
    assert np.allclose(boxes[0], [cx - 100, cy - 60, cx + 100, cy + 60], atol=1e-3)


def test_providers_and_session_report(tmp_path):
    entries = resolve_providers('auto')
    assert entries[-1][0] == 'CPUExecutionProvider'
    with pytest.raises(RuntimeError, match='not available'):
        resolve_providers('definitely_not_a_provider')
    model = constant_model(tmp_path / 'c.onnx', 'images', 64, [('y', np.ones((1, 2)))])
    session, info = create_session(model, 'cpu')
    assert info['used'] == ['CPUExecutionProvider'] and not info['fallback_errors']


@pytest.mark.parametrize('failure', ['initialization', 'replay'])
def test_cuda_graph_fallback_retries_ordinary_inference(tmp_path, monkeypatch, failure):
    model = constant_model(tmp_path / 'fallback.onnx', 'images', 32, [('y', np.ones((1, 2)))])
    if failure == 'replay':
        class FailingGraph:
            shape = (1, 3, 32, 32)
            closed = False

            def run(self, _tensor):
                raise RuntimeError('injected replay failure')

            def close(self):
                self.closed = True

        graph = FailingGraph()

        def open_graph(path, **settings):
            session, info = create_session(path, **settings)
            return session, info, graph

        monkeypatch.setattr(detectors, 'create_cuda_graph_session', open_graph)

    detector = OnnxDetector(model, providers='cpu', threads=1, cuda_graph=True, thread_spinning=False)
    tensor = np.zeros((1, 3, 32, 32), np.float32)
    np.testing.assert_array_equal(detector._run(tensor)[0], np.ones((1, 2)))
    assert detector.runtime['used'] == ['CPUExecutionProvider']
    status = detector.runtime['cuda_graph']
    assert status['requested'] and not status['active']
    assert status['fallback_reason'].startswith(failure)
    if failure == 'replay':
        assert graph.closed
    settings = detector.session.get_session_options()
    assert settings.get_session_config_entry('session.intra_op.allow_spinning') == '0'
    assert settings.get_session_config_entry('session.inter_op.allow_spinning') == '0'


def test_engine_runs_on_a_real_onnx_session(tmp_path):
    """Full path: ORT detector -> letterbox inversion -> tracking -> shots."""
    image, dets = render(size=(480, 640), center=(320, 240), iod=80)
    d = dets[0]
    canvas_kpts = d.keypoints * .5 + [0, 40]
    canvas_box = d.box * .5 + [0, 40, 0, 40]
    out = np.zeros((1, 4 + 1 + 15, 4), np.float32)
    out[0, 0] = (canvas_box[0] + canvas_box[2]) / 2
    out[0, 1] = (canvas_box[1] + canvas_box[3]) / 2
    out[0, 2] = canvas_box[2] - canvas_box[0]
    out[0, 3] = canvas_box[3] - canvas_box[1]
    out[0, 4, 0], out[0, 4, 1:] = .9, 0
    out[0, 5:, 0] = np.c_[canvas_kpts, np.ones(5)].reshape(-1)
    model = constant_model(tmp_path / 'pose5.onnx', 'images', 320, [('output0', out)], {'kpt_shape': '[5, 3]'})
    detector = YoloPoseDetector(model, providers='cpu')
    # A constant model cannot follow crops: this test covers the ONNX session, not refinement.
    engine = CaptureEngine(CaptureConfig().merged({'runtime': {'rotate': '0'}, 'refine': {'enabled': False}}), detector)
    for i in range(10):
        result = engine.process(image, i / 30, i)
    assert result.faces and result.faces[0].quality.passed, result.faces[0].quality.reasons
    assert engine.finalize().captured() == ['front']
    assert json.dumps(engine.describe()['detector']['runtime']['used']) == '["CPUExecutionProvider"]'


def test_yolo_pose_schema_from_ultralytics_kpt_names(tmp_path):
    out = np.zeros((1, 4 + 1 + 3 * 4, 8), np.float32)
    out[0, :5, 0] = [160, 160, 100, 80, .9]
    out[0, 5:, 0] = np.c_[[[160, 175], [140, 150], [180, 150], [160, 195]], np.ones(4)].reshape(-1)
    names = "{0: ['nose', 'left_eye', 'right_eye', 'chin']}"
    model = constant_model(tmp_path / 'named.onnx', 'images', 320, [('output0', out)],
                           {'kpt_shape': '[4, 3]', 'kpt_names': names})
    detector = YoloPoseDetector(model, providers='cpu')
    assert detector.schema.names == ('nose', 'left_eye', 'right_eye', 'chin')
    assert ('left_eye', 'right_eye') in detector.schema.pairs
    dets = detector.detect(FRAME)
    assert dets[0].validate() is None
    engine = CaptureEngine(CaptureConfig().merged({'runtime': {'rotate': '0'}}))
    assert engine.process(np.zeros((480, 640, 3), np.uint8), 0.0, 0, detections=dets) is not None


def test_model_integrity_and_metadata_disagreement_fail_before_capture(tmp_path):
    from dog_capture.engine.zoo import sha256
    model = constant_model(tmp_path / 'pose.onnx', 'images', 320,
                           [('output0', np.zeros((1, 23, 10)))], {'kpt_shape': '[6, 3]'})
    sidecar = model.with_suffix('.json')
    sidecar.write_text(json.dumps({'sha256': '0' * 64}), encoding='utf8')
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        YoloPoseDetector(model, providers='cpu')
    sidecar.write_text(json.dumps({'sha256': sha256(model), 'kpt_shape': [5, 3]}), encoding='utf8')
    with pytest.raises(ValueError, match='kpt_shape disagrees'):
        YoloPoseDetector(model, providers='cpu')
    sidecar.write_text(json.dumps({'sha256': sha256(model), 'input_size': 640}), encoding='utf8')
    with pytest.raises(ValueError, match='input_size disagrees'):
        YoloPoseDetector(model, providers='cpu')


def test_corrupt_default_is_not_silently_replaced_by_another_detector(tmp_path, monkeypatch):
    from dog_capture.engine import zoo
    monkeypatch.setenv('DOG_CAPTURE_MODELS', str(tmp_path))
    (tmp_path / 'dog_face_pose.onnx').write_bytes(b'incomplete download')
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        zoo.find_face_model(CaptureConfig())
