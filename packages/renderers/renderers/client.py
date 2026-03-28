"""Renderer-based verifiers client — replaces the TITO/MITO token client.

All tokenization happens client-side via a Renderer:
    messages → Renderer.render_ids() → token IDs → vLLM /v1/completions → completion tokens
    completion tokens → Renderer.parse_response() → structured message back to verifiers

Verifiers just sends standard chat messages. This client intercepts them,
renders to tokens, calls vLLM's completions endpoint, and returns a structured
Response that verifiers can work with.
"""

from __future__ import annotations

import base64
from typing import Any

import httpx
import numpy as np
from openai import AsyncOpenAI

from renderers.base import Renderer


async def completions_request(
    client: AsyncOpenAI,
    renderer: Renderer,
    messages: list[dict[str, Any]],
    model: str,
    tools: list[dict[str, Any]] | None = None,
    **sampling_args: Any,
) -> dict[str, Any]:
    """Render messages to tokens, call vLLM /v1/completions, return parsed result.

    Returns a dict with: prompt_ids, completion_ids, completion_logprobs,
    content, reasoning_content, tool_calls, finish_reason, usage, routed_experts.
    """
    prompt_ids = renderer.render_ids(messages, tools=tools, add_generation_prompt=True)

    # Build completions body
    body: dict[str, Any] = {
        "model": model,
        "prompt": prompt_ids,
        "logprobs": 1,
        "return_token_ids": True,
        "add_special_tokens": False,
        "skip_special_tokens": False,
        "stop_token_ids": renderer.get_stop_token_ids(),
    }

    # Map sampling args
    for key in ["temperature", "top_p", "seed", "n"]:
        if key in sampling_args:
            body[key] = sampling_args[key]

    max_tokens = sampling_args.get("max_completion_tokens") or sampling_args.get(
        "max_tokens"
    )
    if max_tokens is not None:
        body["max_tokens"] = max_tokens

    extra_body = sampling_args.get("extra_body", {})
    for key in ["repetition_penalty", "min_tokens", "min_p", "top_k"]:
        if key in extra_body:
            body[key] = extra_body[key]

    # Strip /v1 from base_url for raw post
    base_url = str(client.base_url).rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]

    async with httpx.AsyncClient(base_url=base_url, timeout=600.0) as http:
        resp = await http.post("/v1/completions", json=body)

    if resp.status_code != 200:
        raise RuntimeError(f"vLLM returned {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    choice = data.get("choices", [{}])[0]

    completion_ids = choice.get("token_ids") or []
    parsed = renderer.parse_response(completion_ids)

    # Extract logprobs
    logprobs_data = choice.get("logprobs") or {}
    token_logprobs = logprobs_data.get("token_logprobs") or []
    completion_logprobs = [lp if lp is not None else 0.0 for lp in token_logprobs]

    # Extract routed experts
    routed_experts = None
    raw_re = choice.get("routed_experts")
    if isinstance(raw_re, dict) and "data" in raw_re and "shape" in raw_re:
        routed_experts = (
            np.frombuffer(base64.b85decode(raw_re["data"]), dtype=np.int32)
            .reshape(raw_re["shape"])
            .tolist()
        )

    return {
        "prompt_ids": list(prompt_ids),
        "completion_ids": list(completion_ids),
        "completion_logprobs": completion_logprobs,
        "content": parsed.content,
        "reasoning_content": parsed.reasoning_content,
        "tool_calls": parsed.tool_calls,
        "finish_reason": choice.get("finish_reason"),
        "usage": data.get("usage"),
        "routed_experts": routed_experts,
    }
