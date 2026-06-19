#!/usr/bin/env python3
"""Minimal no-server repro for reasoning-only EmptyModelResponseError.

This demonstrates the parser/client boundary behind:

    EmptyModelResponseError(
        "Model returned reasoning but no content and did not call any tools"
    )

It does not call vLLM, create sandboxes, or load a dataset. It builds two
completion token streams with the Nemotron-3 renderer:

1. a normal completion: reasoning closes with ``</think>`` and is followed by a
   ``bash`` tool call;
2. a truncated completion: the same prefix, capped exactly at ``</think>``.

The truncated case parses as non-empty reasoning with no content/tool calls,
which triggers the current RendererClient empty-response classification.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, is_dataclass
from typing import Any

from transformers import AutoTokenizer

from renderers import Nemotron3Renderer, ToolSpec
from verifiers.clients.renderer_client import RendererClient
from verifiers.errors import EmptyModelResponseError


BASH_TOOL: ToolSpec = {
    "name": "bash",
    "description": "Run a shell command.",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Command to execute.",
            }
        },
        "required": ["command"],
    },
}

REASONING = (
    "We need to find the relevant file. "
    "Let's search for NdarrayMixin and the view code.\n\n"
    "Use grep to locate. We'll need to edit file.\n\n"
    "Let's first explore repository structure."
)

TOOL_CALL = """\

<tool_call>
<function=bash>
<parameter=command>
find /testbed -type f -name "*.py" | grep -E "(table|core)" | head -20
</parameter>
</function>
</tool_call>"""


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return jsonable(asdict(value))
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list | tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    return repr(value)


def make_native_response(
    renderer: Nemotron3Renderer,
    completion_ids: list[int],
    *,
    finish_reason: str,
) -> dict[str, Any]:
    parsed = renderer.parse_response(completion_ids, tools=[BASH_TOOL])
    return {
        "request_id": "minimal-repro",
        "prompt_ids": [],
        "completion_ids": completion_ids,
        "completion_logprobs": [],
        "content": parsed.content,
        "reasoning_content": parsed.reasoning_content,
        "tool_calls": parsed.tool_calls,
        "finish_reason": finish_reason,
        "routed_experts": None,
        "multi_modal_data": None,
        "prompt_attribution": None,
    }


async def classify(response: dict[str, Any]) -> str:
    try:
        # ``raise_from_native_response`` does not use ``self``. Calling the
        # real method keeps this repro pinned to the production condition.
        await RendererClient.raise_from_native_response(None, response)  # type: ignore[arg-type]
    except EmptyModelResponseError as exc:
        return f"{type(exc).__name__}: {exc}"
    return "valid"


def summarize(name: str, response: dict[str, Any], classification: str) -> dict[str, Any]:
    return {
        "case": name,
        "classification": classification,
        "finish_reason": response["finish_reason"],
        "content": response["content"],
        "reasoning_content": response["reasoning_content"],
        "tool_calls": jsonable(response["tool_calls"]),
        "summary": {
            "content_len": len(response["content"] or ""),
            "reasoning_len": len(response["reasoning_content"] or ""),
            "tool_call_count": len(response["tool_calls"] or []),
            "completion_len": len(response["completion_ids"]),
        },
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        help="Tokenizer/model name used to instantiate the Nemotron-3 renderer.",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    renderer = Nemotron3Renderer(tokenizer)

    full_completion_text = REASONING + "\n</think>" + TOOL_CALL
    full_completion_ids = tokenizer.encode(
        full_completion_text,
        add_special_tokens=False,
    )

    think_end_id = tokenizer.convert_tokens_to_ids("</think>")
    think_end_index = full_completion_ids.index(think_end_id)
    truncated_completion_ids = full_completion_ids[: think_end_index + 1]

    cases = {
        "full_reasoning_then_tool_call": make_native_response(
            renderer,
            full_completion_ids,
            finish_reason="stop",
        ),
        "truncated_at_think_end": make_native_response(
            renderer,
            truncated_completion_ids,
            finish_reason="length",
        ),
    }

    results = [
        summarize(name, response, await classify(response))
        for name, response in cases.items()
    ]
    print(json.dumps(results, indent=2, ensure_ascii=False))

    truncated = results[1]
    if not truncated["classification"].startswith("EmptyModelResponseError:"):
        raise SystemExit("Expected truncated_at_think_end to raise EmptyModelResponseError")


if __name__ == "__main__":
    asyncio.run(main())
