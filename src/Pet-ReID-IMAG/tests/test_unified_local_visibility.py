"""Engineering tests on synthetic tensors, not visibility-training evidence."""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from pet_id.local_visibility_data import masked_visibility_loss
from pet_id.unified import GeometryPrediction
from pet_id.unified_external_model import GradientControlledFusionRefiner
from pet_id.unified_local_path import (
    LEGACY_RELIABILITY_DEFAULTS,
    LocalPathUnifiedPetReID,
    LocalPathUnifiedPetReIDExport,
    load_local_path_checkpoint,
    normalize_local_path_configuration,
)
from pet_id.unified_local_visibility import (
    LocalPartVisibilityHead,
    LocalVisibilityUnifiedPetReID,
    fuse_with_visibility,
    load_local_visibility_checkpoint,
    save_untrained_visibility_checkpoint,
    visibility_admission,
)
from pet_id.unified_training import sha256_file


class _Geometry(nn.Module):
    crop_size = 8

    def __init__(self):
        super().__init__()
        self.geometry = nn.Module()
        self.geometry.reduction = nn.Sequential(nn.Conv2d(3, 4, 1))
        self.bn = nn.BatchNorm2d(3)
        self.encoder = nn.Linear(3, 512)
        self.register_buffer("concentration", torch.tensor(0.3))

    def _normalize(self, rgb):
        return rgb / 255.0

    def _backbone_descriptor(self, rgb):
        return F.normalize(self.encoder(rgb.mean(dim=(2, 3))), dim=1)

    def _localize(self, rgb):
        # The localization canvas is fixed at 32; match the real backbone's
        # fixed-stride operators instead of a tracer-unsupported adaptive pool.
        features = self.bn(F.avg_pool2d(rgb / 255.0, kernel_size=8, stride=8))
        boxes = rgb.new_tensor([[[0.5, 0.5, 0.5, 0.5], [0.5, 0.6, 0.2, 0.2]]]).expand(rgb.shape[0], -1, -1)
        attention = torch.ones_like(features[:, :2]) / 16
        return GeometryPrediction(
            boxes, rgb.new_zeros(rgb.shape[0]), attention,
            rgb.new_zeros((rgb.shape[0], 2, 4)),
            self.concentration.expand(rgb.shape[0], 2),
        ), features


class _Discretizer(nn.Module):
    def forward(self, boxes, angles):
        return boxes, angles


class _Nose(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 512)
        self.model = nn.Module()
        self.model.register_buffer("pixel_mean", torch.zeros(1, 3, 1, 1))

    def forward(self, rgb):
        return F.normalize(self.encoder(rgb.mean(dim=(2, 3)) / 255), dim=1)


class _Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("maximum_nose_weight", torch.tensor(0.25))

    def nose_weight(self, concentration):
        return self.maximum_nose_weight * ((0.5 - concentration) / 0.05).sigmoid()

    def forward(self, face, nose, concentration):
        weight = self.nose_weight(concentration)[:, None]
        return F.normalize(F.normalize(face, dim=1) * (1 - weight) + F.normalize(nose, dim=1) * weight, dim=1)


def tiny_released():
    torch.manual_seed(731)
    semantic = nn.Module()
    semantic.input_size = 32
    semantic.geometry_frontend = _Geometry()
    semantic.geometry_discretizer = _Discretizer()
    semantic.nose_crop_size = 8
    semantic.face_crop_scales = (1.0, 1.6)
    semantic.register_buffer("face_crop_weights_tensor", torch.tensor([0.05, 0.95]))
    semantic.register_buffer("nose_feather_mask", torch.ones(1, 1, 8, 8) * 0.75)
    semantic.nose_encoder = _Nose()
    semantic.nose_adapter = nn.Linear(512, 512)
    semantic.fusion = _Fusion()
    base = nn.Module()
    base.input_size = 32
    base.base_model = semantic
    base.refiner = GradientControlledFusionRefiner(hidden_dim=8)
    with torch.no_grad():
        base.refiner.direction_gain_logit.fill_(1.0)
        base.refiner.interaction[-1].bias.fill_(0.2)
    return base.eval()


def test_visibility_admission_has_mathematically_exact_zero():
    logits = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0])
    gate = visibility_admission(logits)
    assert torch.equal(gate[:3], torch.zeros(3))
    assert torch.all((gate[3:] > 0) & (gate[3:] <= 1))


def test_disabled_nose_and_refiner_are_exactly_face_and_have_zero_gradient():
    face = (F.normalize(torch.randn(5, 512), dim=1) * (1 + 1e-7)).requires_grad_()
    nose = torch.randn(5, 512, requires_grad=True)
    logits = torch.tensor([[-1., 10.], [10., -1.], [0., 10.], [10., 0.], [-20., -20.]])
    refiner = tiny_released().refiner
    output = fuse_with_visibility(face, nose, logits, torch.tensor(0.25), refiner)
    assert torch.equal(output["embedding"], face)
    assert torch.equal(output["base_embedding"], face)
    assert torch.equal(output["nose_weight"], torch.zeros(5))
    assert torch.equal(output["refiner_admission"], torch.zeros(5))
    changed = fuse_with_visibility(face, nose * 1000 + 5, logits, torch.tensor(0.25), refiner)
    assert torch.equal(changed["embedding"], face)
    output["embedding"].sum().backward()
    assert torch.count_nonzero(nose.grad) == 0
    assert torch.count_nonzero(face.grad) > 0
    for parameter in refiner.parameters():
        assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0


def test_admitted_parts_affect_embedding_and_receive_gradient():
    face = F.normalize(torch.randn(2, 512), dim=1)
    nose = torch.randn(2, 512, requires_grad=True)
    logits = torch.full((2, 2), 2.0, requires_grad=True)
    output = fuse_with_visibility(face, nose, logits, torch.tensor(0.25), tiny_released().refiner)
    assert not torch.equal(output["embedding"], face)
    assert torch.all(output["nose_weight"] > 0)
    assert torch.all(output["nose_weight"] <= 0.25)
    output["embedding"].sum().backward()
    assert nose.grad.abs().sum() > 0
    assert logits.grad.abs().sum() > 0


def test_train_mode_freezes_released_parameters_and_bn_buffers():
    model = LocalVisibilityUnifiedPetReID(tiny_released(), hidden_channels=4)
    before = {k: v.clone() for k, v in model.model.state_dict().items()}
    model.train()
    assert model.visibility_head.training
    assert all(not module.training for module in model.model.modules())
    assert all(not parameter.requires_grad for parameter in model.model.parameters())
    assert all(parameter.requires_grad for parameter in model.visibility_head.parameters())
    optimizer = torch.optim.SGD(model.visibility_head.parameters(), lr=0.01)
    initial = model.visibility_head.classifiers[0][-1].bias.detach().clone()
    result = model(torch.rand(2, 3, 17, 11) * 255, return_aux=True)
    assert torch.equal(result["embedding"], result["face_descriptor"])
    loss = masked_visibility_loss(result["visibility_logits"], torch.tensor([[1., 0.], [1., 1.]]), torch.ones(2, 2))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert not torch.equal(initial, model.visibility_head.classifiers[0][-1].bias)
    assert all(torch.equal(value, model.model.state_dict()[key]) for key, value in before.items())


def test_attention_concentration_is_not_an_input_to_new_fusion():
    model = LocalVisibilityUnifiedPetReID(tiny_released(), hidden_channels=4).eval()
    for classifier in model.visibility_head.classifiers:
        classifier[-1].bias.data.fill_(2.0)
    source = torch.rand(2, 3, 19, 13) * 255
    first = model(source)
    model.model.base_model.geometry_frontend.concentration.fill_(0.99)
    second = model(source)
    assert torch.equal(first, second)


def test_head_rejects_mismatched_geometry_features():
    with pytest.raises(ValueError, match="attention"):
        LocalPartVisibilityHead(3)(torch.ones(1, 3, 4, 4), torch.ones(1, 2, 3, 3))


def test_current_and_exact_old_configs_restore_without_dropping_fields(tmp_path, monkeypatch):
    import pet_id.unified_local_path as module
    source = tmp_path / "source.pth"
    torch.save({"fixture": True}, source)
    original = LocalPathUnifiedPetReID(tiny_released()).eval()
    current = original.configuration()
    legacy = {k: v for k, v in current.items() if k not in LEGACY_RELIABILITY_DEFAULTS}
    assert normalize_local_path_configuration(legacy, current) == current
    monkeypatch.setattr(module, "build_local_path_candidate", lambda *a, **kw: (LocalPathUnifiedPetReID(tiny_released()).eval(), {}))
    rgb = torch.rand(1, 3, 13, 21) * 255
    for index, config in enumerate((legacy, current)):
        payload = {
            "schema_version": 1, "model_type": module.MODEL_TYPE, "model_config": config,
            "source_v3_checkpoint": {"path": str(source), "sha256": sha256_file(source)},
            "model": original.state_dict(),
        }
        path = tmp_path / f"old_{index}.pth"
        torch.save(payload, path)
        restored, _ = load_local_path_checkpoint(path)
        assert torch.equal(restored(rgb), original(rgb))
    with pytest.raises(ValueError, match="fields"):
        normalize_local_path_configuration({**legacy, "mystery_gate": 1}, current)
    with pytest.raises(ValueError, match="fields"):
        normalize_local_path_configuration({k: v for k, v in current.items() if k != "reliability_mode"}, current)


def test_opt_in_checkpoint_roundtrip_and_frozen_source_guard(tmp_path, monkeypatch):
    import pet_id.unified_local_visibility as module
    source = tmp_path / "source.pth"
    torch.save({"fixture": True}, source)
    monkeypatch.setattr(module, "build_released_v3_safely", lambda *a, **kw: (tiny_released(), {}))
    model = LocalVisibilityUnifiedPetReID(tiny_released(), hidden_channels=4).eval()
    path = tmp_path / "visibility.pth"
    payload = save_untrained_visibility_checkpoint(model, path, source_v3_checkpoint=source)
    restored, _ = load_local_visibility_checkpoint(path)
    source_rgb = torch.rand(2, 3, 13, 21) * 255
    assert torch.equal(model(source_rgb), restored(source_rgb))
    assert all(not p.requires_grad for p in restored.parameters())
    training, _ = load_local_visibility_checkpoint(path, for_training=True)
    assert training.visibility_head.training and not training.model.training
    with pytest.raises(ValueError, match="Unsupported local-path"):
        load_local_path_checkpoint(path)
    with pytest.raises(FileExistsError):
        save_untrained_visibility_checkpoint(model, path, source_v3_checkpoint=source)
    payload["model"]["model.base_model.geometry_frontend.bn.running_mean"] = torch.ones(3)
    tampered = tmp_path / "tampered.pth"
    torch.save(payload, tampered)
    with pytest.raises(ValueError, match="Frozen V3 state"):
        load_local_visibility_checkpoint(tampered)


def test_visibility_checkpoint_rejects_unknown_config(tmp_path, monkeypatch):
    import pet_id.unified_local_visibility as module
    source = tmp_path / "source.pth"
    torch.save({"fixture": True}, source)
    monkeypatch.setattr(module, "build_released_v3_safely", lambda *a, **kw: (tiny_released(), {}))
    path = tmp_path / "visibility.pth"
    payload = save_untrained_visibility_checkpoint(LocalVisibilityUnifiedPetReID(tiny_released()), path, source_v3_checkpoint=source)
    payload["model_config"]["unknown"] = True
    torch.save(payload, path)
    with pytest.raises(ValueError, match="configuration"):
        load_local_visibility_checkpoint(path)


def test_full_local_visibility_graph_is_dynamic_single_rgb_single_embedding(tmp_path):
    model = LocalVisibilityUnifiedPetReID(tiny_released(), hidden_channels=4).eval()
    for classifier in model.visibility_head.classifiers:
        classifier[-1].bias.data.fill_(2.0)
    path = tmp_path / "visibility.onnx"
    torch.onnx.export(
        LocalPathUnifiedPetReIDExport(model), (torch.rand(1, 3, 13, 19) * 255,), path,
        input_names=["rgb"], output_names=["embedding"],
        dynamic_axes={"rgb": {0: "batch", 2: "height", 3: "width"}, "embedding": {0: "batch"}},
        opset_version=20, dynamo=False, external_data=False,
    )
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    assert [item.name for item in graph.graph.input] == ["rgb"]
    assert [item.name for item in graph.graph.output] == ["embedding"]
    assert all(x.data_location != onnx.TensorProto.EXTERNAL for x in graph.graph.initializer)
    assert any("visibility_head" in node.name for node in graph.graph.node)
    assert any("nose_encoder" in node.name for node in graph.graph.node)
    assert any("refiner" in node.name for node in graph.graph.node)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for shape in ((1, 3, 13, 19), (2, 3, 7, 25), (1, 3, 2, 2)):
        rgb = torch.rand(shape) * 255
        actual = session.run(None, {"rgb": rgb.numpy()})[0]
        expected = model(rgb).detach().numpy()
        assert actual.shape == (shape[0], 512)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1, atol=1e-6)
