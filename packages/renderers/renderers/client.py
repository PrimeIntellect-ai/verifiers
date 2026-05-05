"""Renderer-based verifiers client.

All tokenization happens client-side via a Renderer:
    messages → Renderer.render_ids() → token IDs → backend transport → completion tokens
    completion tokens → Renderer.parse_response() → structured message back to verifiers

When a RendererPool is passed instead of a single Renderer, the sync tokenization
and parsing work is offloaded to threads for parallel execution across rollouts.
HuggingFace fast tokenizers release the GIL during Rust encoding, so threads
achieve real parallelism.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any, Literal, cast

import numpy as np
from openai import AsyncOpenAI, BadRequestError

from renderers.base import Message, Renderer, RendererPool, ToolSpec

RendererTransport = Literal["prime_vllm_generate", "dynamo_chat_nvext"]

# Logs the (per-message length, total prompt length) on every vLLM /generate
# call at DEBUG, and the same plus the response text on a 4xx at WARNING.
# Lets us post-mortem overlong-prompt rejections without re-running.
_request_logger = logging.getLogger("verifiers.renderer_client.completions_request")


async def _run_pooled(pool: RendererPool, fn):
    """Check out a renderer, run *fn(renderer)* in a thread, return result."""

    def _work():
        with pool.checkout() as r:
            return fn(r)

    return await asyncio.to_thread(_work)


async def completions_request(
    client: AsyncOpenAI,
    renderer: Renderer | RendererPool,
    messages: list[Message],
    model: str,
    tools: list[ToolSpec] | None = None,
    prompt_ids: list[int] | None = None,
    transport: RendererTransport = "prime_vllm_generate",
    **sampling_args: Any,
) -> dict[str, Any]:
    """Render messages to tokens, call a token-in/token-out transport.

    Returns a dict with: prompt_ids, completion_ids, completion_logprobs,
    content, reasoning_content, tool_calls, finish_reason, usage, routed_experts.
    """
    if tools and not getattr(renderer, "supports_tools", True):
        raise ValueError(
            f"{type(renderer).__name__} does not support tools. "
            "Choose a model-specific renderer instead of the default fallback."
        )

    pool = renderer if isinstance(renderer, RendererPool) else None

    # -- Prepare: tokenize prompt --
    def _prepare(r: Renderer):
        prepared_prompt_ids = (
            list(prompt_ids)
            if prompt_ids is not None
            else r.render_ids(messages, tools=tools, add_generation_prompt=True)
        )
        stop_token_ids = r.get_stop_token_ids()
        return prepared_prompt_ids, stop_token_ids

    if pool is not None:
        prompt_ids, stop_token_ids = await _run_pooled(pool, _prepare)
    else:
        prompt_ids, stop_token_ids = _prepare(renderer)

    # -- Build request body --
    generate_keys = (
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "seed",
        "n",
        "repetition_penalty",
        "min_tokens",
        "prompt_logprobs",
        "priority",
        "cache_salt",
    )
    body: dict[str, Any] = {
        "model": model,
        "prompt_token_ids": prompt_ids,
        "stop_token_ids": stop_token_ids,
    }
    for key in generate_keys:
        if key in sampling_args and sampling_args[key] is not None:
            body[key] = sampling_args[key]

    max_tokens = sampling_args.get("max_completion_tokens") or sampling_args.get(
        "max_tokens"
    )
    if max_tokens is not None:
        body["max_tokens"] = max_tokens

    extra_body = sampling_args.get("extra_body") or {}
    for key in generate_keys:
        if key not in body and key in extra_body and extra_body[key] is not None:
            body[key] = extra_body[key]

    path = "/generate"
    if transport == "dynamo_chat_nvext":
        nvext: dict[str, Any] = {
            "token_data": prompt_ids,
            "extra_fields": ["completion_token_ids"],
        }
        priority = sampling_args.get("priority", extra_body.get("priority"))
        if priority is not None:
            nvext["agent_hints"] = {"priority": priority}

        body = {
            "model": model,
            "messages": [{"role": "user", "content": "(token-in mode)"}],
            "stream": False,
            "logprobs": True,
            "stop_token_ids": stop_token_ids,
            "nvext": nvext,
        }
        if max_tokens is not None:
            body["max_completion_tokens"] = max_tokens
        for key in (
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "seed",
            "n",
            "repetition_penalty",
            "min_tokens",
        ):
            value = sampling_args.get(key, extra_body.get(key))
            if value is not None:
                body[key] = value
        path = "/chat/completions"
    elif transport != "prime_vllm_generate":
        raise ValueError(f"Unsupported renderer transport: {transport}")

    extra_headers = sampling_args.get("extra_headers") or {}

    _request_logger.debug(
        "renderer transport=%s prompt_len=%d messages=%d max_tokens=%s stop_ids=%d",
        transport,
        len(prompt_ids),
        len(messages),
        max_tokens,
        len(stop_token_ids),
    )
    post_kwargs: dict[str, Any] = {
        "cast_to": cast(Any, dict[str, Any]),
        "body": body,
    }
    if extra_headers:
        post_kwargs["options"] = cast(Any, {"headers": extra_headers})
    try:
        data = await client.post(path, **post_kwargs)
    except BadRequestError as exc:
        _log_overlong_prompt_diagnostic(
            prompt_ids=prompt_ids,
            messages=messages,
            max_tokens=max_tokens,
            exc=exc,
        )
        raise

    choices = data.get("choices") or [{}]
    choice = choices[0]
    if transport == "dynamo_chat_nvext":
        completion_ids = (
            choice.get("token_ids")
            or choice.get("nvext", {}).get("completion_token_ids")
            or data.get("nvext", {}).get("completion_token_ids")
            or []
        )
        completion_logprobs = [
            float(item.get("logprob") or 0.0)
            for item in (choice.get("logprobs") or {}).get("content") or []
            if isinstance(item, dict)
        ]
        raw_re = (
            choice.get("routed_experts")
            or choice.get("nvext", {}).get("routed_experts")
            or data.get("nvext", {}).get("routed_experts")
        )
        response_id = data.get("id") or data.get("request_id") or ""
    else:
        completion_ids = choice.get("token_ids") or []
        completion_logprobs = [
            float(lp) if lp is not None else 0.0 for lp in choice.get("logprobs") or []
        ]
        raw_re = choice.get("routed_experts")
        response_id = data.get("id") or ""

    if pool is not None:
        parsed = await _run_pooled(pool, lambda r: r.parse_response(completion_ids))
    else:
        parsed = renderer.parse_response(completion_ids)

    # Renderer parsing is responsible for client-side tool detection. If a
    # transport reports a plain stop but parsed tokens contain tool calls, expose
    # the chat-style finish reason so OpenAI-compatible agent loops continue.
    finish_reason = choice.get("finish_reason")
    if parsed.tool_calls and finish_reason == "stop":
        finish_reason = "tool_calls"

    routed_experts = None
    if isinstance(raw_re, dict) and "data" in raw_re and "shape" in raw_re:
        routed_experts = (
            np.frombuffer(base64.b85decode(raw_re["data"]), dtype=np.int32)
            .reshape(raw_re["shape"])
            .tolist()
        )

    return {
        "id": response_id,
        "created": data.get("created") or 0,
        "model": data.get("model") or "",
        "prompt_ids": list(prompt_ids),
        "completion_ids": list(completion_ids),
        "completion_logprobs": completion_logprobs,
        "content": parsed.content,
        "reasoning_content": parsed.reasoning_content,
        "tool_calls": parsed.tool_calls,
        "finish_reason": finish_reason,
        "usage": data.get("usage"),
        "routed_experts": routed_experts,
    }


def _log_overlong_prompt_diagnostic(
    *,
    prompt_ids: list[int],
    messages: list[Any],
    max_tokens: int | None,
    exc: BadRequestError,
) -> None:
    """Log a structured snapshot when vLLM rejects with 4xx — usually overlong.

    Captures total prompt length, per-message role + character count, and
    the first chunk of the response body. Lets us post-mortem WHICH rollout
    blew the context window without rerunning.
    """
    body_text = ""
    response = getattr(exc, "response", None)
    if response is not None:
        body_text = (response.text or "")[:500].replace("\n", " ")
    msg_summary = []
    for i, m in enumerate(messages):
        role = m.get("role", "?")
        content = m.get("content")
        if isinstance(content, str):
            content_len = len(content)
        elif isinstance(content, list):
            content_len = sum(
                len(p.get("text", "")) if isinstance(p, dict) else 0 for p in content
            )
        else:
            content_len = 0
        tool_calls = m.get("tool_calls")
        tc_count = len(tool_calls) if tool_calls else 0
        msg_summary.append(f"[{i}]{role}(c={content_len},tc={tc_count})")
    _request_logger.warning(
        "vllm 4xx prompt_len=%d messages=%d max_tokens=%s per_msg=%s response_body=%s",
        len(prompt_ids),
        len(messages),
        max_tokens,
        " ".join(msg_summary),
        body_text,
    )
