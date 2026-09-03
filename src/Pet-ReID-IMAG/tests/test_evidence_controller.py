from __future__ import annotations

import numpy as np
import torch

from pet_id.evidence_controller import (
    CONTEXT_FEATURE_NAMES,
    CANDIDATE_FEATURE_NAMES,
    EvidenceNet,
    build_evidence_arrays,
    choose_action,
)


def _arrays(mega_scores: np.ndarray, expert_available: bool):
    bifor = np.asarray([0.8, 0.4, 0.2], dtype=np.float32)
    references = np.stack((bifor - 0.02, bifor + 0.02), axis=1)
    mega_references = np.stack((mega_scores - 0.03, mega_scores + 0.03), axis=1)
    metadata = {
        "primary": {
            "descriptor": {
                "branch_available": [True, True],
                "branch_quality": [0.7, 0.8],
                "fusion_weights": [0.4, 0.6],
                "detection": {"confidence": 0.9},
                "runtime_diagnostics": {"body": {"detected": True, "score": 0.8}},
            }
        },
        "experts": {
            "megadescriptor_b224": {
                "crop_coverage": 0.6,
                "quality": {"sharpness": 0.7, "exposure": 0.8},
            }
        },
    }
    return build_evidence_arrays(
        bifor_scores=bifor,
        mega_scores=mega_scores,
        bifor_reference_scores=references,
        mega_reference_scores=mega_references,
        bifor_gallery_consistency=np.asarray([0.9, 0.8, 0.7]),
        mega_gallery_consistency=np.asarray([0.6, 0.7, 0.8]),
        metadata=metadata,
        expert_available=expert_available,
        top_candidates=2,
    )


def test_preconsultation_evidence_does_not_leak_mega_scores():
    first = _arrays(np.asarray([0.1, 0.9, 0.3], dtype=np.float32), False)
    second = _arrays(np.asarray([0.9, 0.1, -0.2], dtype=np.float32), False)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[2], second[2])


def test_evidence_schema_and_model_are_permutation_invariant():
    candidates, mask, context = _arrays(
        np.asarray([0.1, 0.9, 0.3], dtype=np.float32), True
    )
    assert candidates.shape[1] == len(CANDIDATE_FEATURE_NAMES)
    assert context.size == len(CONTEXT_FEATURE_NAMES)
    model = EvidenceNet(hidden_dim=16, dropout=0.0).eval()
    batch = torch.from_numpy(candidates[None])
    batch_mask = torch.from_numpy(mask[None])
    batch_context = torch.from_numpy(context[None])
    with torch.inference_mode():
        direct = model(batch, batch_mask, batch_context)
        reverse = model(batch.flip(1), batch_mask.flip(1), batch_context)
    torch.testing.assert_close(direct, reverse)


def test_action_choice_changes_after_expert_is_available():
    probabilities = {
        "bifor_correct": 0.55,
        "mega_correct": 0.91,
        "consult_success": 0.90,
        "recapture_correct": 0.60,
        "unknown": 0.05,
    }
    before = choose_action(
        probabilities,
        expert_available=False,
        costs={"defer_review": 0.60},
    )
    after = choose_action(
        probabilities,
        expert_available=True,
        costs={"defer_review": 0.60},
    )
    assert before["action"] == "consult_expert"
    assert after["action"] == "accept_mega"
