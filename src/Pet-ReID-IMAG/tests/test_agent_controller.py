from __future__ import annotations

import json

import pytest

from pet_id.agent_controller import (
    AGENT_DECISION_JSON_SCHEMA,
    AgentEvidenceRequest,
    InProcessAgent,
    AgentProtocolError,
    OpenAIResponsesAgent,
    ReplayFallbackAgent,
    build_openai_responses_payload,
    decide_from_identification,
)


def candidate_identification() -> dict:
    return {
        "decision": "closed_set_top1",
        "accepted": True,
        "predicted_pet_id": "e2e-wolf",
        "predicted_display_name": "E2E Wolf",
        "top1_score": 0.8150301575660706,
        "margin": 0.7161001712083817,
        "match_threshold": None,
        "minimum_margin": 0.0,
        "candidates": [
            {
                "pet_id": "e2e-wolf",
                "display_name": "E2E Wolf",
                "score": 0.8150301575660706,
                "reference_count": 2,
                "embedding": [0.1, 0.2],
            },
            {
                "pet_id": "e2e-dorl",
                "display_name": "E2E Dorl",
                "score": 0.0989299863576889,
                "reference_count": 2,
            },
        ],
        "query": {
            "filename": "query.jpg",
            "sha256": "abc123",
            "width": 1250,
            "height": 1873,
            "inference": {
                "descriptor": {
                    "runtime_diagnostics": {
                        "unified": {
                            "single_graph": True,
                            "model_type": "unified_high_resolution_pet_reid",
                            "provider": "CPUExecutionProvider",
                            "external_models": [],
                            "embedding": ["must not leave process"],
                        }
                    }
                }
            },
        },
        "model_fingerprint": "dbd4448133efec28efb770a6ce77c749b4f8b0913c8f40273420be571fe7b000",
        "gallery_snapshot": {"pets": 2, "reference_images": 4},
        "diagnostics": {"mode": "unified_single_graph", "branch_quality": None},
        "hard_case_reasons": [],
        "agent": None,
    }


def test_candidate_request_is_model_agnostic_and_does_not_leak_vectors_or_paths():
    request = AgentEvidenceRequest.from_identification(candidate_identification())
    payload = request.to_dict()
    encoded = json.dumps(payload, ensure_ascii=False)
    assert payload["primary_model"]["model_type"] == "unified_high_resolution_pet_reid"
    assert payload["candidates"][0]["pet_id"] == "e2e-wolf"
    assert "embedding" not in encoded.casefold()
    assert "prototype" not in encoded.casefold()
    assert "path" not in encoded.casefold()
    assert "candidate.1.score" in payload["evidence_catalog"]


def test_replay_fallback_can_call_the_stable_interface():
    result = decide_from_identification(
        candidate_identification(), ReplayFallbackAgent()
    )
    assert result["decision"]["action"] == "accept_primary"
    assert result["decision"]["candidate_pet_id"] == "e2e-wolf"
    assert result["decision"]["provider"] == "replay_fallback"


def test_in_process_provider_receives_only_redacted_request_and_is_validated():
    request = AgentEvidenceRequest.from_identification(candidate_identification())
    seen = {}

    def session_callback(payload):
        seen.update(payload)
        return {
            "action": "accept_primary",
            "candidate_pet_id": "e2e-wolf",
            "confidence": 0.9,
            "evidence_refs": ["score_summary.top1_score"],
            "reasons": ["session_review"],
            "tool": "none",
            "next_observation": None,
            "note": "temporary session provider",
        }

    decision = InProcessAgent(
        session_callback,
        provider="assistant_session",
    ).decide(request)
    encoded = json.dumps(seen, ensure_ascii=False).casefold()
    assert decision.provider == "assistant_session"
    assert decision.candidate_pet_id == "e2e-wolf"
    assert "embedding" not in encoded
    assert "path" not in encoded


def test_decision_validation_rejects_unknown_candidate_and_evidence():
    request = AgentEvidenceRequest.from_identification(candidate_identification())
    base = {
        "action": "accept_primary",
        "candidate_pet_id": "not-in-gallery",
        "confidence": 0.8,
        "evidence_refs": ["score_summary.top1_score"],
        "reasons": ["test"],
        "tool": "none",
        "next_observation": None,
        "note": "test",
    }
    with pytest.raises(AgentProtocolError, match="not in the candidate set"):
        from pet_id.agent_controller import AgentDecision

        AgentDecision.from_payload(base, request, provider="test")
    base["candidate_pet_id"] = "e2e-wolf"
    base["evidence_refs"] = ["invented.score"]
    with pytest.raises(AgentProtocolError, match="unavailable evidence"):
        from pet_id.agent_controller import AgentDecision

        AgentDecision.from_payload(base, request, provider="test")


def test_openai_payload_uses_strict_structured_output_and_no_storage():
    request = AgentEvidenceRequest.from_identification(candidate_identification())
    payload = build_openai_responses_payload(request, model="gpt-test")
    assert payload["store"] is False
    assert payload["text"]["format"]["type"] == "json_schema"
    assert payload["text"]["format"]["strict"] is True
    assert payload["text"]["format"]["schema"] == AGENT_DECISION_JSON_SCHEMA


def test_openai_adapter_parses_fake_responses_without_network():
    calls = []

    def fake_transport(url, payload, headers, timeout):
        calls.append((url, payload, headers, timeout))
        return {
            "id": "resp_test",
            "output_text": json.dumps(
                {
                    "action": "accept_primary",
                    "candidate_pet_id": "e2e-wolf",
                    "confidence": 0.91,
                    "evidence_refs": ["score_summary.top1_score", "score_summary.margin"],
                    "reasons": ["large_primary_margin"],
                    "tool": "none",
                    "next_observation": None,
                    "note": "structured test response",
                }
            ),
        }

    request = AgentEvidenceRequest.from_identification(candidate_identification())
    decision = OpenAIResponsesAgent(
        api_key="test-key",
        model="gpt-test",
        base_url="https://example.invalid/v1",
        transport=fake_transport,
    ).decide(request)
    assert decision.action == "accept_primary"
    assert decision.response_id == "resp_test"
    assert calls[0][0] == "https://example.invalid/v1/responses"
    assert calls[0][2]["Authorization"] == "Bearer test-key"
