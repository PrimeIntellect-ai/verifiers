"""Renderer-based verifiers client.

All tokenization happens client-side via a Renderer:
    messages → Renderer.render_ids() → token IDs → vLLM /v1/generate → completion tokens
    completion tokens → Renderer.parse_response() → structured message back to verifiers

Verifiers just sends standard chat messages. This client intercepts them,
renders to tokens, calls the token-in generate endpoint, and returns a structured
Response that verifiers can work with.
"""

from __future__ import annotations

import base64
from typing import Any, cast

import numpy as np
from openai import AsyncOpenAI

from renderers.base import Message, Renderer, ToolSpec


async def completions_request(
    client: AsyncOpenAI,
    renderer: Renderer,
    messages: list[Message],
    model: str,
    tools: list[ToolSpec] | None = None,
    **sampling_args: Any,
) -> dict[str, Any]:
    """Render messages to tokens, call vLLM /v1/generate, return parsed result.

    Returns a dict with: prompt_ids, completion_ids, completion_logprobs,
    content, reasoning_content, tool_calls, finish_reason, usage, routed_experts.
    """
    if tools and not getattr(renderer, "supports_tools", True):
        raise ValueError(
            f"{type(renderer).__name__} does not support tools. "
            "Choose a model-specific renderer instead of the default fallback."
        )

    prompt_ids = renderer.render_ids(messages, tools=tools, add_generation_prompt=True)
    images = _extract_images(messages)

    body: dict[str, Any] = {
        "model": model,
        "prompt_token_ids": prompt_ids,
        "stop_token_ids": renderer.get_stop_token_ids(),
    }
    if images:
        body["images"] = images

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

    data = await client.post("/generate", cast_to=cast(Any, dict[str, Any]), body=body)
    choice = data.get("choices", [{}])[0]

    completion_ids = choice.get("token_ids") or []
    parsed = renderer.parse_response(completion_ids)

    completion_logprobs = [
        float(lp) if lp is not None else 0.0 for lp in choice.get("logprobs") or []
    ]

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


def _extract_images(messages: list[Message]) -> list[dict[str, str]]:
    from io import BytesIO
    from pathlib import Path

    images: list[dict[str, str]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if item.get("type") == "image_url":
                url = (item.get("image_url") or {}).get("url", "")
                if url.startswith("data:image"):
                    header, b64_data = url.split(",", 1)
                    media_type = header.split(";")[0].split(":")[1]
                    images.append({"data": b64_data, "media_type": media_type})
                elif url.startswith("file://"):
                    raw = Path(url.removeprefix("file://")).read_bytes()
                    images.append(
                        {
                            "data": base64.b64encode(raw).decode("ascii"),
                            "media_type": "image/png",
                        }
                    )
            elif item.get("type") == "image":
                image = item.get("image")
                if image is not None and hasattr(image, "save"):
                    buffer = BytesIO()
                    image.save(buffer, format="PNG")
                    images.append(
                        {
                            "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
                            "media_type": "image/png",
                        }
                    )
    return images
