# Pet-ReID LLM Agent Interface (contract v1)

> Here `v1` versions the serialized Agent request/response contract. It is not
> a model generation or a deployment role.

## Purpose

This interface puts an AI decision layer around the latest frozen identity
models without asking the AI to replace numerical embedding retrieval. The
current latest base candidate is UnifiedPetReID V4 High Resolution. V4 emits a
single L2-normalized 512-D embedding and does not expose separate nose, face,
or body embeddings. The interface therefore describes evidence generically as
`primary_model`, candidate scores, margins, Gallery state, and runtime
diagnostics.

The same request can also be built from older BIFOR/Mega identify responses.
Expert-specific scores are optional. The Agent must never receive embeddings,
prototypes, tensors, or local filesystem paths.

## Request contract

`AgentEvidenceRequest.to_dict()` returns `pet_reid_agent_request_v1`:

- `primary_model`: model fingerprint, model type, provider, graph contract;
- `query`: content hash, dimensions, filename, and sanitized inference metadata;
- `gallery`: model fingerprint and Gallery snapshot;
- `candidates`: ranked `pet_id`, display name, score, reference count, optional
  expert score map;
- `score_summary`: top-1 score, margin, primary acceptance, and decision mode;
- `diagnostics`: runtime diagnostics, hard-case reasons, and any existing Agent
  result;
- `evidence_catalog`: the only evidence references the AI may cite;
- `allowed_actions` and a tool-call budget.

The adapter removes keys containing `embedding`, `prototype`, `vector`,
`feature`, `tensor`, or `path` before serializing the request. This is a data
boundary, not a prompt suggestion.

## Decision contract

The Agent returns exactly one of:

- `accept_primary`
- `consult_expert`
- `request_recapture`
- `reject_unknown`
- `defer_review`

The response also contains a candidate ID (only when it is present in the
request), calibrated-looking confidence in `[0,1]`, evidence references,
short reasons, the matching tool (`megadescriptor`, `recapture`, or `none`),
and a next-observation hint. The local validator rejects unknown candidates,
unknown evidence references, mismatched tools, exhausted tool budgets, and
malformed values.

## Provider boundary

`OpenAIResponsesAgent` sends the request to `/v1/responses` using strict JSON
Schema output and `store=false`. The implementation uses `urllib`, so the core
service does not require the OpenAI SDK. `OPENAI_API_KEY` is required for a
real call; `AGENT_MODEL` defaults to `gpt-5` if no model is provided.

`ReplayFallbackAgent` is deterministic and exists only for offline replay and
contract tests. It must not be described as a learned or production policy.

For a temporary session-level provider, `InProcessAgent` accepts a host callback
that receives the already-redacted request mapping and returns decision JSON.
This lets an orchestrator (for example, the current assistant session) inspect
one request at a time while the local validator still enforces candidate,
evidence, action, and budget constraints. The current ChatGPT/Codex session is
not exposed as a local HTTP server; this adapter is an in-process experiment
hook, not a production model provider.

## Latest V4 replay

The interface was replayed against the completed Unified V4 CPU smoke report:

```powershell
$env:PYTHONPATH = "$PWD\src\Pet-ReID-IMAG"
& D:\CondaData\envs\torch312\python.exe `
  src\Pet-ReID-IMAG\tools\call_agent_controller.py `
  --input-json artifacts\runs\live-stack-e2e\final-20260902-unified-v4-cpu\live-stack-smoke.json `
  --provider replay
```

This checks the interface and candidate grounding only. It is not an LLM
quality evaluation. The current environment has no `OPENAI_API_KEY`, so no
remote call was made during this implementation.

## Integration rule

Do not silently call the remote Agent from `/v1/identify`. First record the
structured request and decision in an explicit Agent route or an offline
replay. Once a provider is configured, tool execution should remain local:
the Agent proposes `consult_expert` or `request_recapture`, and a local
executor performs that action and returns a new request for the next turn.

The 110 V2 controller test identities are already spent and must not be used
to tune this prompt or provider. New identities are required for an Agent
quality claim.

## Files

- `src/Pet-ReID-IMAG/pet_id/agent_controller.py`
- `src/Pet-ReID-IMAG/tools/call_agent_controller.py`
- `src/Pet-ReID-IMAG/tests/test_agent_controller.py`
