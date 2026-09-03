from __future__ import annotations

import numpy as np
import torch

from pet_id.monotonic_evidence_controller import (
    OUTPUT_NAMES,
    SCALAR_FEATURE_NAMES,
    MonotonicResidualController,
    choose_action,
    scalarize_evidence,
)
from tests.test_evidence_controller import _arrays


def test_scalar_preexpert_does_not_leak_mega():
    first = _arrays(np.asarray([0.1, 0.9, 0.3], dtype=np.float32), False)
    second = _arrays(np.asarray([0.9, 0.1, -0.2], dtype=np.float32), False)
    np.testing.assert_array_equal(
        scalarize_evidence(first[0], first[2]),
        scalarize_evidence(second[0], second[2]),
    )


def test_monotonic_bifor_and_unknown_logits():
    model = MonotonicResidualController().eval()
    baseline = torch.zeros(1, len(SCALAR_FEATURE_NAMES))
    stronger = baseline.clone()
    stronger[0, SCALAR_FEATURE_NAMES.index("bifor_top1")] = 0.2
    stronger[0, SCALAR_FEATURE_NAMES.index("bifor_margin")] = 0.1
    with torch.inference_mode():
        first = model(baseline)
        second = model(stronger)
    bifor = OUTPUT_NAMES.index("bifor_correct")
    unknown = OUTPUT_NAMES.index("unknown")
    assert second[0, bifor] >= first[0, bifor]
    assert second[0, unknown] <= first[0, unknown]


def test_unidentified_gallery_size_effect_is_masked():
    model = MonotonicResidualController().eval()
    small_gallery = torch.zeros(1, len(SCALAR_FEATURE_NAMES))
    large_gallery = small_gallery.clone()
    large_gallery[0, SCALAR_FEATURE_NAMES.index("log_gallery_size")] = 10.0
    with torch.inference_mode():
        first = model(small_gallery)
        second = model(large_gallery)
    torch.testing.assert_close(first, second)


def test_consult_then_accept_mega():
    probabilities = {
        "bifor_correct": 0.60,
        "mega_correct": 0.90,
        "unknown": 0.05,
        "expert_gain": 0.20,
    }
    costs = {"defer_review": 0.60, "consult_expert": 0.04}
    assert (
        choose_action(
            probabilities, expert_available=False, costs=costs
        )["action"]
        == "consult_expert"
    )
    assert (
        choose_action(
            probabilities, expert_available=True, costs=costs
        )["action"]
        == "accept_mega"
    )
