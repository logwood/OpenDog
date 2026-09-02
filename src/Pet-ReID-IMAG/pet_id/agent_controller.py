"""Structured Agent Controller interface for Pet-ReID.

The identity models remain numerical specialists.  This module defines the
small, auditable boundary at which an external AI agent may reason over their
results and choose the next action.  It deliberately sends score-level and
diagnostic evidence rather than embeddings or local filesystem paths.

The OpenAI adapter uses the Responses API with a strict JSON Schema response.
The dependency is intentionally implemented with the Python standard library
so the core service can run without an LLM SDK.  A deterministic fallback is
provided for offline replay and contract tests; it is not presented as a
learned controller.
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence


AGENT_REQUEST_SCHEMA_VERSION = "pet_reid_agent_request_v1"
AGENT_DECISION_SCHEMA_VERSION = "pet_reid_agent_decision_v1"

AGENT_ACTIONS = (
    "accept_primary",
    "consult_expert",
    "request_recapture",
    "reject_unknown",
    "defer_review",
)
AGENT_TOOLS = ("none", "megadescriptor", "recapture")


AGENT_DECISION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "action": {"type": "string", "enum": list(AGENT_ACTIONS)},
        "candidate_pet_id": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence_refs": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 8,
        },
        "reasons": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 6,
        },
        "tool": {"type": "string", "enum": list(AGENT_TOOLS)},
        "next_observation": {"type": ["string", "null"]},
        "note": {"type": "string", "maxLength": 500},
    },
    "required": [
        "action",
        "candidate_pet_id",
        "confidence",
        "evidence_refs",
        "reasons",
        "tool",
        "next_observation",
        "note",
    ],
}


AGENT_SYSTEM_PROMPT = """You are the decision layer for a pet re-identification system.

The numerical model is the authority for candidate ranking. Your job is to
choose the safest next action from the supplied evidence, not to invent a new
identity or alter a model score.

Rules:
1. Use only evidence_refs present in evidence_catalog. Never cite an absent
   feature, image, breed, pose, or model output.
2. candidate_pet_id must be one of the supplied candidate IDs when accepting
   the primary result; otherwise use null.
3. If the evidence is insufficient, request recapture, consult the listed
   expert, reject_unknown, or defer_review. Do not guess.
4. The tool must agree with the action: consult_expert -> megadescriptor,
   request_recapture -> recapture, all other actions -> none.
5. Return exactly the requested JSON schema. Keep reasons concise and factual.
"""


class AgentControllerError(RuntimeError):
    """Base class for stable Agent Controller failures."""


class AgentConfigurationError(AgentControllerError):
    """The requested provider is not configured."""


class AgentProtocolError(AgentControllerError):
    """The provider returned a response outside the declared contract."""


class AgentRemoteError(AgentControllerError):
    """The remote provider could not produce a response."""


def _finite_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise AgentProtocolError(f"{field} must be numeric") from error
    if not math.isfinite(result):
        raise AgentProtocolError(f"{field} must be finite")
    return result


def _bounded_float(value: Any, field: str, minimum: float, maximum: float) -> float:
    result = _finite_float(value, field)
    if result < minimum or result > maximum:
        raise AgentProtocolError(
            f"{field} must be between {minimum:g} and {maximum:g}"
        )
    return result


def _copy_json(value: Any) -> Any:
    """Copy JSON-compatible evidence while rejecting non-JSON values."""

    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise AgentProtocolError("evidence must be JSON serializable") from error


def _redact_vectors(value: Any, *, key: str = "") -> Any:
    """Remove embeddings, prototypes, vectors, and local paths recursively."""

    lowered = key.casefold()
    blocked = (
        "embedding",
        "prototype",
        "vector",
        "feature",
        "tensor",
        "path",
    )
    if any(token in lowered for token in blocked):
        return None
    if isinstance(value, Mapping):
        return {
            str(name): _redact_vectors(item, key=str(name))
            for name, item in value.items()
            if not any(token in str(name).casefold() for token in blocked)
        }
    if isinstance(value, (list, tuple)):
        return [_redact_vectors(item, key=key) for item in value]
    return value


def _model_info_from_identification(result: Mapping[str, Any]) -> dict[str, Any]:
    query = result.get("query")
    query = query if isinstance(query, Mapping) else {}
    inference = query.get("inference")
    inference = inference if isinstance(inference, Mapping) else {}
    descriptor = inference.get("descriptor")
    descriptor = descriptor if isinstance(descriptor, Mapping) else {}
    diagnostics = descriptor.get("runtime_diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
    unified = diagnostics.get("unified")
    unified = unified if isinstance(unified, Mapping) else {}
    return {
        "backend": result.get("decision"),
        "model_fingerprint": result.get("model_fingerprint"),
        "model_type": unified.get("model_type"),
        "single_graph": unified.get("single_graph"),
        "raw_spatial_input": unified.get("raw_spatial_input"),
        "provider": unified.get("provider"),
        "external_models": unified.get("external_models"),
        "embedding_dim": 512,
    }


@dataclass(frozen=True)
class AgentCandidate:
    pet_id: str
    display_name: str | None
    rank: int
    score: float
    reference_count: int | None
    expert_scores: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pet_id": self.pet_id,
            "display_name": self.display_name,
            "rank": self.rank,
            "score": self.score,
            "reference_count": self.reference_count,
            "expert_scores": self.expert_scores,
        }


@dataclass(frozen=True)
class AgentEvidenceRequest:
    """Stable request sent to a controller provider."""

    request_id: str
    primary_model: dict[str, Any]
    query: dict[str, Any]
    gallery: dict[str, Any]
    candidates: tuple[AgentCandidate, ...]
    score_summary: dict[str, Any]
    diagnostics: dict[str, Any]
    evidence_catalog: tuple[str, ...]
    allowed_actions: tuple[str, ...] = AGENT_ACTIONS
    max_tool_calls: int = 1
    tool_calls_used: int = 0
    tool_costs: dict[str, float] | None = None
    prior_decisions: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise AgentProtocolError("request_id cannot be blank")
        if not self.candidates:
            raise AgentProtocolError("at least one candidate is required")
        unknown_actions = set(self.allowed_actions) - set(AGENT_ACTIONS)
        if unknown_actions:
            raise AgentProtocolError(
                f"unsupported actions: {sorted(unknown_actions)}"
            )
        if self.max_tool_calls < 0 or self.tool_calls_used < 0:
            raise AgentProtocolError("tool-call budgets cannot be negative")
        if self.tool_calls_used > self.max_tool_calls:
            raise AgentProtocolError("tool_calls_used exceeds max_tool_calls")

    @property
    def candidate_ids(self) -> frozenset[str]:
        return frozenset(candidate.pet_id for candidate in self.candidates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": AGENT_REQUEST_SCHEMA_VERSION,
            "request_id": self.request_id,
            "primary_model": _copy_json(_redact_vectors(self.primary_model)),
            "query": _copy_json(_redact_vectors(self.query)),
            "gallery": _copy_json(_redact_vectors(self.gallery)),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "score_summary": _copy_json(_redact_vectors(self.score_summary)),
            "diagnostics": _copy_json(_redact_vectors(self.diagnostics)),
            "evidence_catalog": list(self.evidence_catalog),
            "allowed_actions": list(self.allowed_actions),
            "budget": {
                "max_tool_calls": self.max_tool_calls,
                "tool_calls_used": self.tool_calls_used,
                "tool_costs": _copy_json(self.tool_costs or {}),
            },
            "prior_decisions": _copy_json(list(self.prior_decisions)),
        }

    @classmethod
    def from_identification(
        cls,
        result: Mapping[str, Any],
        *,
        request_id: str | None = None,
        allowed_actions: Sequence[str] = AGENT_ACTIONS,
        max_tool_calls: int = 1,
        tool_calls_used: int = 0,
        tool_costs: Mapping[str, float] | None = None,
        prior_decisions: Sequence[Mapping[str, Any]] = (),
    ) -> "AgentEvidenceRequest":
        """Convert V3, V4, or legacy identify output without leaking vectors."""

        raw_candidates = result.get("candidates")
        if not isinstance(raw_candidates, Sequence) or isinstance(
            raw_candidates, (str, bytes)
        ):
            raise AgentProtocolError("identification result has no candidates")
        candidates: list[AgentCandidate] = []
        for index, raw in enumerate(raw_candidates, start=1):
            if not isinstance(raw, Mapping):
                raise AgentProtocolError("candidate rows must be objects")
            pet_id = str(raw.get("pet_id") or "").strip()
            if not pet_id:
                raise AgentProtocolError("candidate pet_id cannot be blank")
            expert_scores = raw.get("expert_scores")
            normalized_expert_scores = None
            if isinstance(expert_scores, Mapping):
                normalized_expert_scores = {
                    str(name): _finite_float(score, f"expert_scores.{name}")
                    for name, score in expert_scores.items()
                }
            reference_count = raw.get("reference_count")
            normalized_reference_count = (
                int(reference_count) if reference_count is not None else None
            )
            candidates.append(
                AgentCandidate(
                    pet_id=pet_id,
                    display_name=(
                        str(raw["display_name"])
                        if raw.get("display_name") is not None
                        else None
                    ),
                    rank=index,
                    score=_finite_float(raw.get("score"), f"candidates[{index}].score"),
                    reference_count=normalized_reference_count,
                    expert_scores=normalized_expert_scores,
                )
            )

        query = result.get("query")
        query = query if isinstance(query, Mapping) else {}
        inference = query.get("inference")
        inference = inference if isinstance(inference, Mapping) else {}
        diagnostics = result.get("diagnostics")
        diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
        hard_case_reasons = result.get("hard_case_reasons")
        if not isinstance(hard_case_reasons, Sequence) or isinstance(
            hard_case_reasons, (str, bytes)
        ):
            hard_case_reasons = []
        score_summary = {
            "top1_score": _finite_float(result.get("top1_score"), "top1_score"),
            "margin": (
                None
                if result.get("margin") is None
                else _finite_float(result.get("margin"), "margin")
            ),
            "accepted_by_primary": bool(result.get("accepted")),
            "decision": result.get("decision"),
            "match_threshold": result.get("match_threshold"),
            "minimum_margin": result.get("minimum_margin"),
        }
        query_payload = {
            "filename": query.get("filename"),
            "sha256": query.get("sha256"),
            "width": query.get("width"),
            "height": query.get("height"),
            "inference": _redact_vectors(inference),
        }
        gallery_payload = {
            "model_fingerprint": result.get("model_fingerprint"),
            "snapshot": result.get("gallery_snapshot"),
        }
        diagnostics_payload = {
            "runtime": _redact_vectors(diagnostics),
            "hard_case_reasons": [str(item) for item in hard_case_reasons],
            "agent_existing": _redact_vectors(result.get("agent")),
        }
        catalog = [
            "primary_model.model_fingerprint",
            "primary_model.model_type",
            "query.width",
            "query.height",
            "score_summary.top1_score",
            "score_summary.margin",
            "score_summary.accepted_by_primary",
            "gallery.snapshot",
            "diagnostics.hard_case_reasons",
        ]
        for index, candidate in enumerate(candidates, start=1):
            catalog.extend(
                (
                    f"candidate.{index}.score",
                    f"candidate.{index}.reference_count",
                )
            )
            for expert_name in sorted(candidate.expert_scores or {}):
                catalog.append(f"candidate.{index}.expert_scores.{expert_name}")
        return cls(
            request_id=request_id or f"agent-{uuid.uuid4().hex}",
            primary_model=_model_info_from_identification(result),
            query=query_payload,
            gallery=gallery_payload,
            candidates=tuple(candidates),
            score_summary=score_summary,
            diagnostics=diagnostics_payload,
            evidence_catalog=tuple(catalog),
            allowed_actions=tuple(allowed_actions),
            max_tool_calls=max_tool_calls,
            tool_calls_used=tool_calls_used,
            tool_costs=(
                {str(name): _bounded_float(value, f"tool_costs.{name}", 0.0, 1000.0)
                 for name, value in tool_costs.items()}
                if tool_costs is not None
                else {"megadescriptor": 0.04, "recapture": 0.0}
            ),
            prior_decisions=tuple(_copy_json(dict(item)) for item in prior_decisions),
        )


@dataclass(frozen=True)
class AgentDecision:
    action: str
    candidate_pet_id: str | None
    confidence: float
    evidence_refs: tuple[str, ...]
    reasons: tuple[str, ...]
    tool: str
    next_observation: str | None
    note: str
    provider: str = "unknown"
    response_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": AGENT_DECISION_SCHEMA_VERSION,
            "action": self.action,
            "candidate_pet_id": self.candidate_pet_id,
            "confidence": self.confidence,
            "evidence_refs": list(self.evidence_refs),
            "reasons": list(self.reasons),
            "tool": self.tool,
            "next_observation": self.next_observation,
            "note": self.note,
            "provider": self.provider,
            "response_id": self.response_id,
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        request: AgentEvidenceRequest,
        *,
        provider: str,
        response_id: str | None = None,
    ) -> "AgentDecision":
        expected = set(AGENT_DECISION_JSON_SCHEMA["properties"])
        unknown = set(payload) - expected
        missing = expected - set(payload)
        if unknown or missing:
            raise AgentProtocolError(
                f"decision fields mismatch; missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        action = str(payload["action"])
        if action not in request.allowed_actions:
            raise AgentProtocolError(f"action is not allowed: {action}")
        candidate = payload["candidate_pet_id"]
        if candidate is not None:
            candidate = str(candidate)
            if candidate not in request.candidate_ids:
                raise AgentProtocolError(
                    f"candidate_pet_id is not in the candidate set: {candidate}"
                )
        confidence = _bounded_float(payload["confidence"], "confidence", 0.0, 1.0)
        raw_evidence_refs = payload["evidence_refs"]
        if not isinstance(raw_evidence_refs, Sequence) or isinstance(
            raw_evidence_refs, (str, bytes)
        ):
            raise AgentProtocolError("evidence_refs must be an array")
        if len(raw_evidence_refs) > 8:
            raise AgentProtocolError("evidence_refs cannot contain more than 8 items")
        evidence_refs = tuple(str(item) for item in raw_evidence_refs)
        unknown_refs = set(evidence_refs) - set(request.evidence_catalog)
        if unknown_refs:
            raise AgentProtocolError(
                f"decision cited unavailable evidence: {sorted(unknown_refs)}"
            )
        raw_reasons = payload["reasons"]
        if not isinstance(raw_reasons, Sequence) or isinstance(
            raw_reasons, (str, bytes)
        ):
            raise AgentProtocolError("reasons must be an array")
        if len(raw_reasons) > 6:
            raise AgentProtocolError("reasons cannot contain more than 6 items")
        reasons = tuple(str(item).strip() for item in raw_reasons)
        if any(not item for item in reasons):
            raise AgentProtocolError("reasons cannot contain blank strings")
        tool = str(payload["tool"])
        if tool not in AGENT_TOOLS:
            raise AgentProtocolError(f"unsupported tool: {tool}")
        expected_tool = {
            "consult_expert": "megadescriptor",
            "request_recapture": "recapture",
        }.get(action, "none")
        if tool != expected_tool:
            raise AgentProtocolError(
                f"tool {tool!r} does not match action {action!r}; "
                f"expected {expected_tool!r}"
            )
        if tool != "none" and request.tool_calls_used >= request.max_tool_calls:
            raise AgentProtocolError("the requested tool-call budget is exhausted")
        next_observation = payload["next_observation"]
        if next_observation is not None:
            next_observation = str(next_observation)
        note = str(payload["note"])
        if len(note) > 500:
            raise AgentProtocolError("note exceeds 500 characters")
        return cls(
            action=action,
            candidate_pet_id=candidate,
            confidence=confidence,
            evidence_refs=evidence_refs,
            reasons=reasons,
            tool=tool,
            next_observation=next_observation,
            note=note,
            provider=provider,
            response_id=response_id,
        )


class AgentController(Protocol):
    def decide(self, request: AgentEvidenceRequest) -> AgentDecision: ...


class ReplayFallbackAgent:
    """Deterministic controller used only for local contract/replay testing."""

    def decide(self, request: AgentEvidenceRequest) -> AgentDecision:
        top = request.candidates[0]
        summary = request.score_summary
        hard_cases = set(request.diagnostics.get("hard_case_reasons", ()))
        if bool(summary.get("accepted_by_primary")) and not hard_cases:
            action = "accept_primary"
            candidate = top.pet_id
            tool = "none"
            next_observation = None
            reasons = ("primary_model_accepted",)
        elif "unknown" in " ".join(sorted(hard_cases)).casefold():
            action = "reject_unknown"
            candidate = None
            tool = "none"
            next_observation = None
            reasons = ("primary_model_marked_unknown",)
        else:
            action = "defer_review"
            candidate = None
            tool = "none"
            next_observation = "human_review"
            reasons = ("evidence_requires_review",)
        score = float(summary["top1_score"])
        confidence = max(0.0, min(1.0, (score + 1.0) * 0.5))
        refs = ["score_summary.top1_score"]
        if summary.get("margin") is not None:
            refs.append("score_summary.margin")
        return AgentDecision(
            action=action,
            candidate_pet_id=candidate,
            confidence=confidence,
            evidence_refs=tuple(refs),
            reasons=reasons,
            tool=tool,
            next_observation=next_observation,
            note="offline replay fallback; no remote model was called",
            provider="replay_fallback",
        )


def build_openai_responses_payload(
    request: AgentEvidenceRequest,
    *,
    model: str,
    max_output_tokens: int = 500,
) -> dict[str, Any]:
    """Build an auditable Responses API request without making a network call."""

    if not model.strip():
        raise AgentConfigurationError("model cannot be blank")
    if max_output_tokens < 1:
        raise AgentConfigurationError("max_output_tokens must be positive")
    return {
        "model": model,
        "store": False,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": AGENT_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": json.dumps(
                            request.to_dict(),
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    }
                ],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "pet_reid_agent_decision",
                "strict": True,
                "schema": AGENT_DECISION_JSON_SCHEMA,
            }
        },
        "max_output_tokens": max_output_tokens,
    }


def _extract_response_text(response: Mapping[str, Any]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text
    output = response.get("output")
    if isinstance(output, Sequence) and not isinstance(output, (str, bytes)):
        fragments: list[str] = []
        for item in output:
            if not isinstance(item, Mapping):
                continue
            content = item.get("content")
            if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
                continue
            for part in content:
                if not isinstance(part, Mapping):
                    continue
                text = part.get("text")
                if isinstance(text, str):
                    fragments.append(text)
        if fragments:
            return "".join(fragments)
    raise AgentProtocolError("Responses API returned no output text")


class OpenAIResponsesAgent:
    """Call an OpenAI-compatible Responses endpoint with strict JSON output."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
        max_output_tokens: int = 500,
        transport: Callable[[str, Mapping[str, Any], Mapping[str, str], float], Mapping[str, Any]]
        | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model or os.environ.get("AGENT_MODEL") or "gpt-5"
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        self.timeout_seconds = float(timeout_seconds)
        self.max_output_tokens = int(max_output_tokens)
        self.transport = transport or self._post_json

    def decide(self, request: AgentEvidenceRequest) -> AgentDecision:
        if not self.api_key:
            raise AgentConfigurationError(
                "OPENAI_API_KEY is not configured; use ReplayFallbackAgent for offline replay"
            )
        payload = build_openai_responses_payload(
            request,
            model=self.model,
            max_output_tokens=self.max_output_tokens,
        )
        response = self.transport(
            self._responses_url(),
            payload,
            {"Authorization": f"Bearer {self.api_key}"},
            self.timeout_seconds,
        )
        try:
            raw_payload = json.loads(_extract_response_text(response))
        except json.JSONDecodeError as error:
            raise AgentProtocolError("provider output was not valid JSON") from error
        if not isinstance(raw_payload, Mapping):
            raise AgentProtocolError("provider JSON output must be an object")
        return AgentDecision.from_payload(
            raw_payload,
            request,
            provider="openai_responses",
            response_id=(
                str(response["id"]) if response.get("id") is not None else None
            ),
        )

    def _responses_url(self) -> str:
        base = self.base_url.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return f"{base}/responses"

    def _post_json(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                **dict(headers),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                value = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise AgentRemoteError(
                f"agent provider returned HTTP {error.code}"
            ) from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise AgentRemoteError("agent provider request failed") from error
        except json.JSONDecodeError as error:
            raise AgentRemoteError("agent provider returned invalid JSON") from error
        if not isinstance(value, Mapping):
            raise AgentRemoteError("agent provider response must be an object")
        return value


def decide_from_identification(
    identification: Mapping[str, Any],
    controller: AgentController,
    **request_kwargs: Any,
) -> dict[str, Any]:
    """Convenience bridge used by replay tools and future API integration."""

    request = AgentEvidenceRequest.from_identification(
        identification,
        **request_kwargs,
    )
    decision = controller.decide(request)
    return {
        "request": request.to_dict(),
        "decision": decision.to_dict(),
    }


__all__ = [
    "AGENT_ACTIONS",
    "AGENT_DECISION_JSON_SCHEMA",
    "AGENT_DECISION_SCHEMA_VERSION",
    "AGENT_REQUEST_SCHEMA_VERSION",
    "AGENT_SYSTEM_PROMPT",
    "AgentCandidate",
    "AgentConfigurationError",
    "AgentController",
    "AgentControllerError",
    "AgentDecision",
    "AgentEvidenceRequest",
    "AgentProtocolError",
    "AgentRemoteError",
    "OpenAIResponsesAgent",
    "ReplayFallbackAgent",
    "build_openai_responses_payload",
    "decide_from_identification",
]
