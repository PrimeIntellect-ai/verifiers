#!/usr/bin/env python3
"""Reproduce renderer-client EmptyModelResponseError from a saved rollout row.

This is a debugging helper for reasoning-model responses that are truncated
at a thinking boundary. It replays a saved rollout prompt through
RendererClient, finds the first ``</think>`` token in a longer response, then
runs a capped request that ends at that boundary. When the cap lands before
post-thinking content or tool calls, RendererClient raises:

    EmptyModelResponseError(
        "Model returned reasoning but no content and did not call any tools"
    )
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Literal

from transformers import AutoTokenizer

from verifiers.clients import resolve_client
from verifiers.errors import Error
from verifiers.types import (
    AssistantMessage,
    ClientConfig,
    Message,
    SystemMessage,
    TextMessage,
    Tool,
    ToolCall,
    ToolMessage,
    UserMessage,
)

Variant = Literal["prompt_only", "prompt_plus_completion"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Saved rollout JSONL")
    parser.add_argument("--output", required=True, help="JSONL output path")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible /v1 URL")
    parser.add_argument(
        "--model",
        default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    )
    parser.add_argument("--example-id", required=True)
    parser.add_argument("--rollout-id")
    parser.add_argument(
        "--variant",
        choices=("prompt_only", "prompt_plus_completion"),
        default="prompt_only",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--discover-max-tokens", type=int, default=512)
    parser.add_argument(
        "--cap-tokens",
        type=int,
        help="Skip discovery and use this cap for the repro request.",
    )
    parser.add_argument("--api-key-var", default="VLLM_API_KEY")
    parser.add_argument(
        "--dump-dir",
        help=(
            "Sets VF_RENDERER_RESPONSE_DUMP_DIR so RendererClient writes the "
            "raw native empty-like response before raising."
        ),
    )
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list | tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump(mode="json"))
    return repr(value)


def response_summary(response: dict[str, Any] | None) -> dict[str, Any]:
    if response is None:
        return {"response_is_none": True}
    return {
        "finish_reason": response.get("finish_reason"),
        "content_len": len(response.get("content") or ""),
        "reasoning_len": len(response.get("reasoning_content") or ""),
        "tool_call_count": len(response.get("tool_calls") or []),
        "completion_len": len(response.get("completion_ids") or []),
    }


def load_rollout_row(
    path: Path, *, example_id: str, rollout_id: str | None
) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if str(row.get("example_id")) != example_id:
                continue
            if rollout_id is not None and str(row.get("rollout_id")) != rollout_id:
                continue
            return row
    raise ValueError(
        f"No rollout row matched example_id={example_id!r}, rollout_id={rollout_id!r}"
    )


def parse_message(raw: dict[str, Any]) -> Message:
    role = raw.get("role")
    if role == "system":
        return SystemMessage(**raw)
    if role == "user":
        return UserMessage(**raw)
    if role == "assistant":
        data = dict(raw)
        if data.get("tool_calls"):
            tool_calls = []
            for tool_call in data["tool_calls"]:
                if isinstance(tool_call, str):
                    tool_call = json.loads(tool_call)
                tool_calls.append(ToolCall(**tool_call))
            data["tool_calls"] = tool_calls
        return AssistantMessage(**data)
    if role == "tool":
        return ToolMessage(**raw)
    if role == "text":
        return TextMessage(**raw)
    raise ValueError(f"Unsupported message role: {role!r}")


def parse_tools(raw_tools: list[dict[str, Any]] | None) -> list[Tool] | None:
    if not raw_tools:
        return None
    return [Tool(**tool) for tool in raw_tools]


def messages_for_variant(row: dict[str, Any], variant: Variant) -> list[Message]:
    raw_messages = list(row.get("prompt") or [])
    if variant == "prompt_plus_completion":
        raw_messages.extend(row.get("completion") or [])
    return [parse_message(message) for message in raw_messages]


async def renderer_request(
    *,
    client,
    row: dict[str, Any],
    model: str,
    max_tokens: int,
    seed: int,
    variant: Variant,
) -> dict[str, Any]:
    messages = messages_for_variant(row, variant)
    tools = parse_tools(row.get("tool_defs"))
    native_prompt, extra_kwargs = await client.to_native_prompt(messages)
    native_tools = await client.to_native_tools(tools)
    state = {
        "trajectory_id": row.get("rollout_id"),
        "example_id": row.get("example_id"),
        "replay_variant": variant,
    }
    response = await client.get_native_response(
        native_prompt,
        model,
        {"max_tokens": max_tokens, "seed": seed},
        native_tools,
        **extra_kwargs,
        state=state,
    )
    record: dict[str, Any] = {
        "example_id": row.get("example_id"),
        "rollout_id": row.get("rollout_id"),
        "policy_version": row.get("policy_version"),
        "variant": variant,
        "max_tokens": max_tokens,
        "seed": seed,
        "response_summary": response_summary(response),
        "native_response": jsonable(response),
    }
    try:
        await client.raise_from_native_response(response)
        record["classification"] = "valid"
    except Error as exc:
        record["classification"] = exc.__class__.__name__
        record["validation_error"] = repr(exc)
    return record


def first_think_end_cap(record: dict[str, Any], model: str) -> int:
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
    think_end_id = tokenizer.convert_tokens_to_ids("</think>")
    completion_ids = record["native_response"]["completion_ids"]
    for index, token_id in enumerate(completion_ids):
        if token_id == think_end_id:
            return index + 1
    raise ValueError("Discovery response did not contain a </think> token")


async def main() -> None:
    args = parse_args()
    if args.dump_dir:
        os.environ["VF_RENDERER_RESPONSE_DUMP_DIR"] = args.dump_dir

    row = load_rollout_row(
        Path(args.input),
        example_id=args.example_id,
        rollout_id=args.rollout_id,
    )
    client = resolve_client(
        ClientConfig(
            client_type="renderer",
            api_base_url=args.base_url,
            api_key_var=args.api_key_var,
            timeout=3600.0,
        )
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    try:
        if args.cap_tokens is None:
            discovery = await renderer_request(
                client=client,
                row=row,
                model=args.model,
                max_tokens=args.discover_max_tokens,
                seed=args.seed,
                variant=args.variant,
            )
            discovery["phase"] = "discover"
            cap_tokens = first_think_end_cap(discovery, args.model)
            discovery["suggested_cap_tokens"] = cap_tokens
            records.append(discovery)
        else:
            cap_tokens = args.cap_tokens

        capped = await renderer_request(
            client=client,
            row=row,
            model=args.model,
            max_tokens=cap_tokens,
            seed=args.seed,
            variant=args.variant,
        )
        capped["phase"] = "capped_repro"
        records.append(capped)
    finally:
        await client.close()

    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = [
        {
            "phase": record["phase"],
            "classification": record["classification"],
            "max_tokens": record["max_tokens"],
            "response_summary": record["response_summary"],
            "suggested_cap_tokens": record.get("suggested_cap_tokens"),
            "validation_error": record.get("validation_error"),
        }
        for record in records
    ]
    print(json.dumps({"output": str(output), "summary": summary}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
