"""Call or replay the structured Pet-ReID Agent Controller.

Examples:

    # Offline contract replay against a latest Unified V4 smoke report.
    python tools/call_agent_controller.py \
        --input-json ../../artifacts/runs/live-stack-e2e/final-20260902-unified-v4-cpu/live-stack-smoke.json \
        --provider replay

    # Real Responses API call after explicitly configuring a key.
    $env:OPENAI_API_KEY = "..."
    python tools/call_agent_controller.py --input-json identify.json --provider openai
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pet_id.agent_controller import (
    AgentEvidenceRequest,
    OpenAIResponsesAgent,
    ReplayFallbackAgent,
)


def _load_identification(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("identification"), dict):
        payload = payload["identification"]
    if not isinstance(payload, dict):
        raise ValueError("input JSON must be an identify result or smoke report")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--provider", choices=("replay", "openai"), default="replay")
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--request-id")
    parser.add_argument("--max-tool-calls", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    identification = _load_identification(args.input_json.expanduser().resolve())
    request = AgentEvidenceRequest.from_identification(
        identification,
        request_id=args.request_id,
        max_tool_calls=args.max_tool_calls,
    )
    controller = (
        ReplayFallbackAgent()
        if args.provider == "replay"
        else OpenAIResponsesAgent(model=args.model)
    )
    decision = controller.decide(request)
    result = {"request": request.to_dict(), "decision": decision.to_dict()}
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
