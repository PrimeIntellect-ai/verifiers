"""Renderer-based generate client for vLLM 0.20's /inference/v1/generate.

    messages → Renderer.render_ids() → token IDs → POST /inference/v1/generate
    → completion tokens → Renderer.parse_response() → structured message

When a RendererPool is passed instead of a single Renderer, the sync tokenization
and parsing work is offloaded to threads for parallel execution across rollouts.
HuggingFace fast tokenizers release the GIL during Rust encoding, so threads
achieve real parallelism.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any, cast

import numpy as np
from openai import AsyncOpenAI, BadRequestError

from renderers.base import Message, Renderer, RendererPool, ToolSpec

_request_logger = logging.getLogger("renderers.client")


async def _run_pooled(pool: RendererPool, fn):
    def _work():
        with pool.checkout() as r:
            return fn(r)

    return await asyncio.to_thread(_work)


async def generate(
    *,
    client: AsyncOpenAI,
    renderer: Renderer | RendererPool,
    messages: list[Message],
    model: str,
    prompt_ids: list[int] | None = None,
    tools: list[ToolSpec] | None = None,
    sampling_params: dict[str, Any] | None = None,
    cache_salt: str | None = None,
    priority: int | None = None,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Tokenize messages, call vLLM /inference/v1/generate, parse the response.

    ``sampling_params`` is forwarded to vLLM verbatim. Two fields are always
    set by us and override caller values: ``stop_token_ids`` (from the
    renderer) and ``logprobs=1`` (we always emit completion_logprobs). Pass
    ``prompt_ids`` to skip rendering and use a prebuilt token sequence.

    Returns a dict with: request_id, prompt_ids, completion_ids,
    completion_logprobs, content, reasoning_content, tool_calls,
    finish_reason, routed_experts.
    """
    if tools and not getattr(renderer, "supports_tools", True):
        raise ValueError(
            f"{type(renderer).__name__} does not support tools. "
            "Choose a model-specific renderer instead of the default fallback."
        )

    pool = renderer if isinstance(renderer, RendererPool) else None

    def _prepare(r: Renderer):
        ids = (
            list(prompt_ids)
            if prompt_ids is not None
            else r.render_ids(messages, tools=tools, add_generation_prompt=True)
        )
        return ids, r.get_stop_token_ids()

    if pool is not None:
        prompt_ids, stop_token_ids = await _run_pooled(pool, _prepare)
    else:
        prompt_ids, stop_token_ids = _prepare(renderer)

    sp: dict[str, Any] = dict(sampling_params or {})
    sp["stop_token_ids"] = stop_token_ids
    sp["logprobs"] = 1
    sp.setdefault("skip_special_tokens", False)

    body: dict[str, Any] = {
        "model": model,
        "token_ids": prompt_ids,
        "sampling_params": sp,
    }
    if cache_salt is not None:
        body["cache_salt"] = cache_salt
    if priority is not None:
        body["priority"] = priority

    # /inference/v1/generate is mounted at the server root, not under /v1
    # like the OpenAI-compatible endpoints. Build an absolute URL so the
    # AsyncOpenAI client doesn't prepend its automatic /v1.
    base = str(client.base_url).rstrip("/").removesuffix("/v1")
    endpoint = f"{base}/inference/v1/generate"
    _request_logger.debug(
        "POST %s prompt_len=%d max_tokens=%s",
        endpoint,
        len(prompt_ids),
        sp.get("max_tokens"),
    )
    post_kwargs: dict[str, Any] = {
        "cast_to": cast(Any, dict[str, Any]),
        "body": body,
    }
    if extra_headers:
        post_kwargs["options"] = cast(Any, {"headers": extra_headers})
    try:
        data = await client.post(endpoint, **post_kwargs)
    except BadRequestError as exc:
        _log_overlong_prompt_diagnostic(
            prompt_ids=prompt_ids,
            messages=messages,
            max_tokens=sp.get("max_tokens"),
            exc=exc,
        )
        raise

    choice = (data.get("choices") or [{}])[0]
    completion_ids = choice.get("token_ids") or []

    if pool is not None:
        parsed = await _run_pooled(pool, lambda r: r.parse_response(completion_ids))
    else:
        parsed = renderer.parse_response(completion_ids)

    # ChatCompletionLogProbs flatten: {"content": [{"logprob": ...}, ...]}
    raw_logprobs = choice.get("logprobs") or {}
    content_lp = raw_logprobs.get("content") if isinstance(raw_logprobs, dict) else None
    completion_logprobs = [float(c.get("logprob") or 0.0) for c in content_lp or []]

    routed_experts = None
    raw_re = choice.get("routed_experts")
    if isinstance(raw_re, dict) and "data" in raw_re and "shape" in raw_re:
        routed_experts = (
            np.frombuffer(base64.b85decode(raw_re["data"]), dtype=np.int32)
            .reshape(raw_re["shape"])
            .tolist()
        )

    # /inference/v1/generate returns finish_reason in {"stop","length",...} —
    # never "tool_calls" (a chat-completions concept). Promote stop→tool_calls
    # when we extracted tool calls client-side, so OpenAI-compatible agent
    # loops continue past the tool turn instead of treating the response as
    # final.
    finish_reason = choice.get("finish_reason")
    if parsed.tool_calls and finish_reason == "stop":
        finish_reason = "tool_calls"

    return {
        "request_id": data.get("request_id") or "",
        "prompt_ids": list(prompt_ids),
        "completion_ids": list(completion_ids),
        "completion_logprobs": completion_logprobs,
        "content": parsed.content,
        "reasoning_content": parsed.reasoning_content,
        "tool_calls": parsed.tool_calls,
        "finish_reason": finish_reason,
        "routed_experts": routed_experts,
    }


def _log_overlong_prompt_diagnostic(
    *,
    prompt_ids: list[int],
    messages: list[Message],
    max_tokens: int | None,
    exc: BadRequestError,
) -> None:
    """Log a structured snapshot when vLLM rejects with 4xx — usually overlong.

    Captures total prompt length, per-message role + character count, and
    the first chunk of the response body.
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
