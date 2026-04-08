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


class RenderingProxy:
    def __init__(
        self,
        renderer: Renderer,
        vllm_base_url: str = "http://localhost:8000",
        processor=None,
    ):
        self._renderer = renderer
        self._processor = processor
        self._vllm_base_url = vllm_base_url.rstrip("/")
        base = self._vllm_base_url
        if base.endswith("/v1"):
            base = base[:-3]
        self._client = httpx.AsyncClient(base_url=base, timeout=600.0)
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
        resp = await self._client.request(
            method=request.method,
            url=str(request.url.path),
            content=await request.body(),
            headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
        )
        return JSONResponse(json.loads(resp.content), status_code=resp.status_code)

    async def _handle_chat_completions(self, request: Request):
        body = await request.json()
        messages = body.get("messages", [])
        tools = body.get("tools")
        model = body.get("model")

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
            resp = await self._client.post("/v1/generate", json=generate_body)
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
