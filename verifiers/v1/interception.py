"""The per-rollout interception server.

Every rollout runs an harness program whose OpenAI-style calls are caught here:
this small localhost server routes each `POST /v1/chat/completions` to our
`Client`, records a `Turn`, and returns the result in OpenAI shape. We inject
`OPENAI_BASE_URL`/`OPENAI_API_KEY` so the program's SDK talks to us. Chat
completions only, no streaming — only the model endpoint is intercepted (tools
and the user simulator are handled out-of-band for now).
"""

import logging
import secrets
from collections.abc import Awaitable, Callable

from aiohttp import web

from verifiers.v1.clients import RolloutContext
from verifiers.v1.trace import Trace, Turn
from verifiers.v1.types import (
    AssistantMessage,
    Message,
    Messages,
    Response,
    SystemMessage,
    Tool,
    ToolCall,
    ToolMessage,
    UserMessage,
)

logger = logging.getLogger(__name__)


def parse_message(raw: dict) -> Message:
    """An OpenAI request message dict -> a typed Message."""
    role = raw.get("role")
    content = raw.get("content")
    if isinstance(content, list):  # multimodal parts -> joined text
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    content = content or ""
    if role == "system":
        return SystemMessage(content=content)
    if role == "tool":
        return ToolMessage(tool_call_id=raw.get("tool_call_id", ""), content=content)
    if role == "assistant":
        calls = [
            ToolCall(
                id=c["id"],
                name=c["function"]["name"],
                arguments=c["function"]["arguments"],
            )
            for c in (raw.get("tool_calls") or [])
        ] or None
        return AssistantMessage(content=raw.get("content"), tool_calls=calls)
    return UserMessage(content=content)


def parse_tools(raw: list[dict] | None) -> list[Tool] | None:
    if not raw:
        return None
    return [
        Tool(
            name=t["function"]["name"],
            description=t["function"].get("description", ""),
            parameters=t["function"].get("parameters", {}),
            strict=t["function"].get("strict"),
        )
        for t in raw
        if t.get("type", "function") == "function"
    ]


def serialize_completion(response: Response, model: str) -> dict:
    """A `Response` -> an OpenAI chat.completion dict the program's SDK expects."""
    message: dict = {"role": "assistant", "content": response.message.content}
    if response.message.tool_calls:
        message["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": c.arguments},
            }
            for c in response.message.tool_calls
        ]
    usage = (
        {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        if response.usage
        else None
    )
    return {
        "id": response.id or "vf-intercept",
        "object": "chat.completion",
        "created": response.created,
        "model": response.model or model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": response.finish_reason or "stop",
            }
        ],
        "usage": usage,
    }


# The interception server only ever proxies one rollout's own harness requests, so
# aiohttp's default 1 MiB body cap is an artificial bottleneck — a large tool result
# (e.g. a `cat` of a big file) trips it and the harness gets a 413. Allow large bodies; the
# upstream provider and the model's context window are the real limits, this is just a
# host-OOM backstop.
_MAX_REQUEST_BODY = 1024**3  # 1 GiB (aiohttp's default is 1 MiB)


class InterceptionServer:
    """A localhost server that proxies one rollout's chat-completions calls."""

    def __init__(
        self,
        ctx: RolloutContext,
        trace: Trace,
        stops: list[Callable[[Trace], Awaitable[bool]]] | None = None,
        max_turns: int | None = None,
    ) -> None:
        self.ctx = ctx
        self.trace = trace
        self.stops = stops or []
        self.max_turns = max_turns
        self.secret = secrets.token_urlsafe(16)
        self.port = 0
        self.runner: web.AppRunner | None = None

    async def __aenter__(self) -> "InterceptionServer":
        app = web.Application(client_max_size=_MAX_REQUEST_BODY)
        app.router.add_post("/v1/chat/completions", self.handle_chat)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]  # actual ephemeral port
        logger.debug("interception up: id=%s port=%d", self.trace.id, self.port)
        return self

    async def __aexit__(self, *exc) -> None:
        if self.runner is not None:
            await self.runner.cleanup()

    async def handle_chat(self, request: web.Request) -> web.Response:
        if request.headers.get("Authorization") != f"Bearer {self.secret}":
            logger.warning("interception: unauthorized request id=%s", self.trace.id)
            return web.json_response({"error": "unauthorized"}, status=401)
        turn = len(self.trace.trajectory) + 1  # 1-based: the turn being handled
        # The framework's max_turns cap refuses turns past it, halting any harness (same
        # mechanism as a @stop) — a turn limit is the framework's job, not the harness's.
        if self.max_turns is not None and len(self.trace.trajectory) >= self.max_turns:
            self.trace.stop("max_turns")
            logger.debug("max_turns %d reached: id=%s", self.max_turns, self.trace.id)
            return web.json_response(
                {"error": "rollout stopped: max_turns"}, status=400
            )
        # A @stop firing here refuses the turn before it is served, which halts the
        # harness (its model call errors out); Harness.run treats that exit as clean.
        for stop in self.stops:
            if await stop(self.trace):
                self.trace.stop(stop.__name__)
                logger.debug(
                    "stop %r fired: id=%s turn=%d", stop.__name__, self.trace.id, turn
                )
                return web.json_response(
                    {"error": f"rollout stopped: {stop.__name__}"}, status=400
                )
        body = await request.json()
        prompt: Messages = [parse_message(m) for m in body.get("messages", [])]
        tools = parse_tools(body.get("tools"))
        logger.debug(
            "chat request: id=%s turn=%d model=%s messages=%d tools=%d",
            self.trace.id,
            turn,
            self.ctx.model,
            len(prompt),
            len(tools or []),
        )
        try:
            response = await self.ctx.client.get_response(
                prompt, self.ctx.model, self.ctx.sampling, tools
            )
        except Exception as e:  # surface to the program as an API error
            logger.warning("model call failed: id=%s %s", self.trace.id, e)
            return web.json_response({"error": str(e)}, status=502)
        completion_tokens = response.usage.completion_tokens if response.usage else None
        logger.debug(
            "chat response: id=%s finish=%s tool_calls=%d completion_tokens=%s",
            self.trace.id,
            response.finish_reason,
            len(response.message.tool_calls or []),
            completion_tokens,
        )
        self.trace.trajectory.append(
            Turn(prompt=prompt, response=response, tokens=response.tokens)
        )  # branches are derived from the trajectory (see Trace.branches)
        return web.json_response(serialize_completion(response, self.ctx.model))
