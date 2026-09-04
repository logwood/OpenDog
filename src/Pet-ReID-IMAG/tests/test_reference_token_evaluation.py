"""Regression tests for the independent token-matcher evaluator."""

import importlib.util
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "evaluate_reference_token_model.py"
)
SPEC = importlib.util.spec_from_file_location(
    "_reference_token_evaluation_tool",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
EVALUATION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EVALUATION
SPEC.loader.exec_module(EVALUATION)


class _AuxiliaryBaselineStub(torch.nn.Module):
    """Expose an exact singleton score with a distinct rounding path."""

    def __init__(self) -> None:
        super().__init__()
        self.return_aux_values: list[bool] = []

    def forward_encoded(
        self,
        query_descriptor: torch.Tensor,
        reference_descriptors: torch.Tensor,
        reference_mask: torch.Tensor,
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        self.return_aux_values.append(bool(kwargs.get("return_aux")))
        centroid = F.normalize(reference_descriptors.mean(dim=1), dim=1)
        baseline = torch.einsum("bd,bd->b", query_descriptor, centroid)
        baseline = baseline + torch.finfo(baseline.dtype).eps
        return {"score": baseline, "baseline_score": baseline}


def test_score_sets_uses_matchers_own_exact_baseline() -> None:
    model = _AuxiliaryBaselineStub()
    query_descriptors = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    query_tokens = query_descriptors[:, None, :]
    reference_descriptors = torch.tensor(
        [[[1.0, 0.0]], [[0.0, 1.0]]]
    )
    reference_tokens = reference_descriptors[:, :, None, :]

    learned, baseline, _gate = EVALUATION._score_sets(
        model,
        query_descriptors,
        query_tokens,
        reference_descriptors,
        reference_tokens,
        device=torch.device("cpu"),
        identity_chunk=1,
    )

    assert model.return_aux_values == [True, True]
    assert torch.equal(learned, baseline)
    assert float(baseline.diagonal().min()) > 1.0
