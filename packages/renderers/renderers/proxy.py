"""Rendering proxy — intercepts chat completions, renders to tokens, forwards to /v1/generate.

Text-only: Renderer tokenizes → /v1/generate with token IDs.
VLM: Renderer tokenizes (with image placeholders) + raw images extracted →
/v1/generate with token IDs + images. vLLM processes images and generates.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from renderers.base import Renderer

logger = logging.getLogger("rendering.proxy")

UPSTREAM_BASE_URL_HEADER = "X-Renderer-Upstream-Base-URL"
HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


class RenderingProxy:
    def __init__(
        self,
        renderer: Renderer,
        vllm_base_url: str | None = "http://localhost:8000",
        processor=None,
    ):
        self._renderer = renderer
        self._processor = processor
        self._default_vllm_base_url = _normalize_upstream_base_url(vllm_base_url) if vllm_base_url else None
        self._client = httpx.AsyncClient(timeout=600.0)
        self._app = Starlette(
            routes=[
                Route("/v1/chat/completions", self._handle_chat_completions, methods=["POST"]),
                Route("/v1/models", self._proxy_passthrough, methods=["GET"]),
                Route("/health", self._health, methods=["GET"]),
            ]
        )

    @property
    def app(self) -> Starlette:
        return self._app

    async def close(self) -> None:
        await self._client.aclose()

    async def _health(self, request: Request):
        return JSONResponse({"status": "ok"})

    async def _proxy_passthrough(self, request: Request):
        upstream_base_url = self._get_upstream_base_url(request)
        resp = await self._client.request(
            method=request.method,
            url=f"{upstream_base_url}{request.url.path}",
            content=await request.body(),
            headers=_forward_headers(request.headers),
        )
        return JSONResponse(json.loads(resp.content), status_code=resp.status_code)

    async def _handle_chat_completions(self, request: Request):
        body = await request.json()
        messages = body.get("messages", [])
        tools = body.get("tools")
        model = body.get("model")
        if tools and not getattr(self._renderer, "supports_tools", True):
            return JSONResponse(
                {
                    "error": (
                        f"{type(self._renderer).__name__} does not support tools. "
                        "Choose a model-specific renderer instead of the default fallback."
                    )
                },
                status_code=400,
            )

        # Renderer tokenizes (handles image placeholders for VLM)
        prompt_ids = self._renderer.render_ids(messages, tools=tools, add_generation_prompt=True)

        # Extract raw images from messages (for VLM)
        images = _extract_images(messages)

        # Build /v1/generate request
        generate_body: dict[str, Any] = {
            "model": model,
            "prompt_token_ids": prompt_ids,
            "max_tokens": body.get("max_completion_tokens") or body.get("max_tokens", 4096),
            "temperature": body.get("temperature", 1.0),
            "top_p": body.get("top_p", 1.0),
            "stop_token_ids": self._renderer.get_stop_token_ids(),
        }
        if images:
            generate_body["images"] = images

        for key in ["seed", "n", "repetition_penalty", "min_tokens", "min_p", "top_k"]:
            extra = body.get("extra_body", {})
            if key in extra:
                generate_body[key] = extra[key]
            elif key in body:
                generate_body[key] = body[key]

        try:
            upstream_base_url = self._get_upstream_base_url(request)
            resp = await self._client.post(
                f"{upstream_base_url}/v1/generate",
                json=generate_body,
                headers=_forward_headers(request.headers),
            )
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        except Exception as e:
            logger.error(f"Proxy → vLLM request failed: {e}")
            return JSONResponse({"error": str(e)}, status_code=502)

        if resp.status_code != 200:
            logger.error(f"vLLM returned {resp.status_code}: {resp.text[:200]}")
            try:
                error_body = json.loads(resp.content)
            except (json.JSONDecodeError, ValueError):
                error_body = {"error": resp.text[:500]}
            return JSONResponse(error_body, status_code=resp.status_code)

        gen_resp = resp.json()
        return JSONResponse(self._to_chat_response(gen_resp, prompt_ids))

    def _get_upstream_base_url(self, request: Request) -> str:
        header_base_url = request.headers.get(UPSTREAM_BASE_URL_HEADER)
        if header_base_url:
            return _normalize_upstream_base_url(header_base_url)
        if self._default_vllm_base_url:
            return self._default_vllm_base_url
        raise ValueError(
            f"Missing {UPSTREAM_BASE_URL_HEADER} and no default upstream base URL is configured."
        )

    def _to_chat_response(self, gen_resp: dict, prompt_ids: list[int]) -> dict:
        choice = gen_resp.get("choices", [{}])[0]
        completion_ids = choice.get("token_ids", [])
        parsed = self._renderer.parse_response(completion_ids)

        tool_calls = None
        if parsed.tool_calls:
            tool_calls = [
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": (
                            json.dumps(tc["function"]["arguments"])
                            if isinstance(tc["function"]["arguments"], dict)
                            else tc["function"]["arguments"]
                        ),
                    },
                }
                for i, tc in enumerate(parsed.tool_calls)
            ]

        logprobs = choice.get("logprobs", [])
        chat_logprobs = (
            {"content": [{"token": "", "logprob": lp} for lp in logprobs]} if logprobs else None
        )

        return {
            "id": gen_resp.get("id", ""),
            "object": "chat.completion",
            "created": 0,
            "model": gen_resp.get("model", ""),
            "prompt_token_ids": gen_resp.get("prompt_token_ids", prompt_ids),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": parsed.content,
                        "reasoning_content": parsed.reasoning_content,
                        "tool_calls": tool_calls,
                    },
                    "finish_reason": choice.get("finish_reason", "stop"),
                    "logprobs": chat_logprobs,
                    "token_ids": completion_ids,
                    "routed_experts": choice.get("routed_experts"),
                }
            ],
            "usage": gen_resp.get("usage", {}),
        }


def _extract_images(messages: list[dict]) -> list[dict]:
    """Extract base64 image data from messages, in document order."""
    from io import BytesIO

    images = []
    for msg in messages:
        content = msg.get("content")
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
                    from pathlib import Path

                    raw = Path(url.removeprefix("file://")).read_bytes()
                    images.append({"data": base64.b64encode(raw).decode("ascii"), "media_type": "image/png"})
            elif item.get("type") == "image":
                img_obj = item.get("image")
                if img_obj and hasattr(img_obj, "save"):
                    buf = BytesIO()
                    img_obj.save(buf, format="PNG")
                    images.append({"data": base64.b64encode(buf.getvalue()).decode("ascii"), "media_type": "image/png"})
    return images


def _normalize_upstream_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3]
    return normalized


def _forward_headers(headers: Any) -> dict[str, str]:
    forwarded = {}
    for key, value in headers.items():
        lower = key.lower()
        if lower in HOP_BY_HOP_HEADERS or lower == UPSTREAM_BASE_URL_HEADER.lower():
            continue
        forwarded[key] = value
    return forwarded
