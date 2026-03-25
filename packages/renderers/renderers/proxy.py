"""Rendering proxy — intercepts OpenAI chat completions requests, renders
messages to tokens via the Renderer, and forwards as /v1/completions.

Runs as a lightweight async HTTP server between the verifiers client and vLLM.
Verifiers sends standard chat messages → proxy renders to tokens → vLLM generates.

Usage:
    proxy = RenderingProxy(renderer, vllm_base_url="http://localhost:8000")
    await proxy.start(port=8100)
    # verifiers talks to http://localhost:8100/v1/chat/completions
    # proxy renders messages → forwards tokens to http://localhost:8000/v1/completions
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from renderers.base import Renderer


class RenderingProxy:
    def __init__(self, renderer: Renderer, vllm_base_url: str = "http://localhost:8000"):
        self._renderer = renderer
        self._vllm_base_url = vllm_base_url.rstrip("/")
        # Strip trailing /v1 — we add the full path ourselves
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

    async def _health(self, request: Request):
        return JSONResponse({"status": "ok"})

    async def _proxy_passthrough(self, request: Request):
        """Forward request to vLLM unchanged."""
        resp = await self._client.request(
            method=request.method,
            url=str(request.url.path),
            content=await request.body(),
            headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
        )
        return JSONResponse(json.loads(resp.content), status_code=resp.status_code)

    async def _handle_chat_completions(self, request: Request):
        """Intercept chat completions: render messages → forward as /v1/completions."""
        body = await request.json()

        messages = body.get("messages", [])
        tools = body.get("tools")
        model = body.get("model")

        # Render messages to token IDs
        prompt_ids = self._renderer.render_ids(
            messages,
            tools=tools,
            add_generation_prompt=True,
        )

        # Build /v1/completions request
        completions_body: dict[str, Any] = {
            "model": model,
            "prompt": prompt_ids,
            "max_tokens": body.get("max_completion_tokens") or body.get("max_tokens", 4096),
            "temperature": body.get("temperature", 1.0),
            "top_p": body.get("top_p", 1.0),
            "logprobs": 1,
            "return_token_ids": True,
            "add_special_tokens": False,
            "skip_special_tokens": False,
            "stop_token_ids": self._renderer.get_stop_token_ids(),
        }

        # Forward optional params
        for key in ["seed", "n", "repetition_penalty", "min_tokens", "min_p", "top_k"]:
            extra = body.get("extra_body", {})
            if key in extra:
                completions_body[key] = extra[key]
            elif key in body:
                completions_body[key] = body[key]

        # Call vLLM /v1/completions
        import logging

        logger = logging.getLogger("rendering.proxy")
        try:
            resp = await self._client.post("/v1/completions", json=completions_body)
        except Exception as e:
            logger.error(f"Proxy → vLLM request failed: {e}")
            return JSONResponse({"error": str(e)}, status_code=502)

        if resp.status_code != 200:
            logger.error(f"vLLM returned {resp.status_code}: {resp.text[:200]}")
            return JSONResponse(json.loads(resp.content), status_code=resp.status_code)

        completions_resp = resp.json()

        # Convert completions response → chat completions response format
        # so verifiers' standard OpenAIChatCompletionsClient can parse it
        chat_resp = self._completions_to_chat(completions_resp, prompt_ids)
        return JSONResponse(chat_resp)

    def _completions_to_chat(self, completions_resp: dict, prompt_ids: list[int]) -> dict:
        """Convert /v1/completions response to /v1/chat/completions format."""
        choice = completions_resp.get("choices", [{}])[0]
        completion_ids = choice.get("token_ids", [])

        # Parse generated tokens back into structured message
        parsed = self._renderer.parse_response(completion_ids)

        # Build tool_calls in OAI format
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

        # Build logprobs in chat format
        logprobs_data = choice.get("logprobs", {})
        token_logprobs = logprobs_data.get("token_logprobs", []) if logprobs_data else []
        chat_logprobs = (
            {"content": [{"token": "", "logprob": lp if lp is not None else 0.0} for lp in token_logprobs]}
            if token_logprobs
            else None
        )

        return {
            "id": completions_resp.get("id", ""),
            "object": "chat.completion",
            "created": completions_resp.get("created", 0),
            "model": completions_resp.get("model", ""),
            "prompt_token_ids": prompt_ids,
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
                }
            ],
            "usage": completions_resp.get("usage", {}),
        }
