"""ONNX Runtime contract tests for the image-set graph."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pet_id.reference_aware_model import (
    ReferenceAwarePetReID,
    ReferenceAwarePetReIDExport,
)
from pet_id.reference_aware_onnx_runtime import ReferenceAwareONNXRuntime
from pet_id.reference_set_model import QueryConditionedReferenceMatcher


class TinyEncoder(nn.Module):
    descriptor_dim = 4

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 4)

    def forward(self, images):
        return self.projection(images.mean(dim=(2, 3)))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ReferenceAwareONNXRuntimeTest(unittest.TestCase):
    def test_predict_and_strict_pixel_contract(self):
        torch.manual_seed(2)
        model = ReferenceAwarePetReID(
            TinyEncoder(),
            QueryConditionedReferenceMatcher(
                descriptor_dim=4,
                hidden_dim=4,
                max_references=2,
                reference_top_k=1,
            ),
        ).eval()
        wrapper = ReferenceAwarePetReIDExport(model).eval()
        query = torch.rand(1, 3, 4, 4) * 255.0
        references = torch.rand(1, 2, 3, 4, 4) * 255.0
        mask = torch.tensor([[True, False]])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = root / "model.onnx"
            torch.onnx.export(
                wrapper,
                (query, references, mask),
                graph,
                input_names=["query_rgb", "reference_rgb", "reference_mask"],
                output_names=["score"],
                dynamic_axes={
                    "query_rgb": {0: "batch"},
                    "reference_rgb": {0: "batch"},
                    "reference_mask": {0: "batch"},
                    "score": {0: "batch"},
                },
                opset_version=20,
                dynamo=False,
            )
            metadata = {
                "onnx_sha256": file_sha256(graph),
                "input_contract": {"reference_rgb": ["N", 2, 3, 4, 4]},
            }
            (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            runtime = ReferenceAwareONNXRuntime(graph)
            actual = runtime.predict(query.numpy(), references.numpy(), mask.numpy())
            with torch.inference_mode():
                expected = wrapper(query, references, mask).numpy()
            np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=0.0)
            with self.assertRaisesRegex(ValueError, "0..255"):
                runtime.predict(query.numpy() + 300.0, references.numpy(), mask.numpy())


if __name__ == "__main__":
    unittest.main()
