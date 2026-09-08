"""Geometry, gradient, and dynamic ONNX guards for graph-internal local sampling."""

import math

import numpy as np
import onnxruntime as ort
import pytest
import torch
from torch import nn

from pet_id.unified_local_path import (
    ContentBoundedRawCropper,
    LocalPathUnifiedPetReIDExport,
    load_local_path_checkpoint,
    project_rotated_box,
)


@pytest.mark.parametrize("width,height", [(2, 2), (11, 37), (37, 11), (513, 224)])
def test_all_rotated_corners_stay_inside_original_image(width, height):
    boxes = torch.tensor(
        [
            [-10.0, -20.0, 150.0, 170.0],
            [width + 10.0, height + 20.0, 15.0, 18.0],
            [width / 2, height / 2, 4.0, 5.0],
        ]
    )
    angles = torch.tensor([math.pi / 4, -math.pi / 3, 0.2])
    fitted, factor, shift = project_rotated_box(
        boxes, angles, torch.tensor(float(width)), torch.tensor(float(height))
    )
    assert torch.isfinite(fitted).all()
    assert torch.all((factor > 0) & (factor <= 1))
    assert torch.all(shift >= 0)
    for sx in (-1, 1):
        for sy in (-1, 1):
            x = (
                fitted[:, 0]
                + sx * angles.cos() * fitted[:, 2] / 2
                - sy * angles.sin() * fitted[:, 3] / 2
            )
            y = (
                fitted[:, 1]
                + sx * angles.sin() * fitted[:, 2] / 2
                + sy * angles.cos() * fitted[:, 3] / 2
            )
            assert torch.all((x >= -1e-4) & (x <= width - 1 + 1e-4))
            assert torch.all((y >= -1e-4) & (y <= height - 1 + 1e-4))


def test_valid_physical_box_is_not_moved_or_shrunk():
    box = torch.tensor([[20.0, 25.0, 10.0, 12.0]])
    fitted, factor, shift = project_rotated_box(
        box, torch.tensor([0.1]), torch.tensor(80.0), torch.tensor(60.0)
    )
    torch.testing.assert_close(fitted, box)
    torch.testing.assert_close(factor, torch.ones(1))
    torch.testing.assert_close(shift, torch.zeros(1))


def test_constant_pixels_are_preserved_even_for_oversized_rotated_roi():
    cropper = ContentBoundedRawCropper(13)
    rgb = torch.full((2, 3, 7, 19), 123.0)
    boxes = torch.tensor([[-1.0, 2.0, 0.9, 0.8], [0.5, 0.5, 0.6, 0.9]])
    angles = torch.tensor([0.7, -0.4])
    crops = cropper(rgb, boxes, angles)
    torch.testing.assert_close(
        crops, torch.full_like(crops, 123.0), rtol=1e-5, atol=1e-3
    )


def test_gradients_reach_source_pixels_and_geometry():
    cropper = ContentBoundedRawCropper(11)
    rgb = (
        torch.arange(3 * 40 * 60, dtype=torch.float32)
        .reshape(1, 3, 40, 60)
        .requires_grad_()
    )
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.3]], requires_grad=True)
    angles = torch.tensor([0.12], requires_grad=True)
    cropper(rgb, boxes, angles).square().mean().backward()
    for value in (rgb, boxes, angles):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0


class _FixedGeometry(nn.Module):
    def __init__(self):
        super().__init__()
        self.cropper = ContentBoundedRawCropper(9)
        self.register_buffer("boxes", torch.tensor([[0.4, 0.6, 0.7, 0.5]]))
        self.register_buffer("angles", torch.tensor([0.2]))

    def forward(self, rgb):
        return self.cropper(
            rgb, self.boxes.expand(rgb.shape[0], -1), self.angles.expand(rgb.shape[0])
        )


def test_dynamic_onnx_keeps_coordinate_mapping_for_new_shapes(tmp_path):
    model = _FixedGeometry().eval()
    path = tmp_path / "local_crop.onnx"
    torch.onnx.export(
        model,
        (torch.rand(1, 3, 13, 23),),
        path,
        input_names=["rgb"],
        output_names=["crop"],
        dynamic_axes={
            "rgb": {0: "batch", 2: "height", 3: "width"},
            "crop": {0: "batch"},
        },
        opset_version=20,
        dynamo=False,
    )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for shape in ((1, 3, 13, 23), (2, 3, 31, 17), (1, 3, 2, 2), (1, 3, 17, 31)):
        source = torch.rand(shape)
        with torch.inference_mode():
            expected = model(source).numpy()
        actual = session.run(None, {"rgb": source.numpy()})[0]
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


class _KeywordOnlyDiagnosticModel(nn.Module):
    def forward(self, rgb, *, return_aux=False):
        value = rgb.mean(dim=(1, 2, 3))[:, None].expand(-1, 512)
        value = torch.nn.functional.normalize(value, dim=1)
        return {"embedding": value} if return_aux else value


def test_export_boundary_exposes_only_one_rgb_input(tmp_path):
    wrapper = LocalPathUnifiedPetReIDExport(_KeywordOnlyDiagnosticModel()).eval()
    path = tmp_path / "one_rgb.onnx"
    torch.onnx.export(
        wrapper,
        (torch.ones(1, 3, 7, 11),),
        path,
        input_names=["rgb"],
        output_names=["embedding"],
        dynamic_axes={
            "rgb": {0: "batch", 2: "height", 3: "width"},
            "embedding": {0: "batch"},
        },
        opset_version=20,
        dynamo=False,
    )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [x.name for x in session.get_inputs()] == ["rgb"]
    assert [x.name for x in session.get_outputs()] == ["embedding"]
    result = session.run(None, {"rgb": np.ones((2, 3, 13, 5), dtype=np.float32)})[0]
    assert result.shape == (2, 512)
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1.0, atol=1e-6)


def test_candidate_loader_rejects_unknown_checkpoint_envelope(tmp_path):
    path = tmp_path / "not_local_path.pth"
    torch.save({"schema_version": 1, "model_type": "not_the_candidate"}, path)
    with pytest.raises(ValueError, match="Unsupported local-path"):
        load_local_path_checkpoint(path)
