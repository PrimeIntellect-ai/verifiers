"""The per-rollout interception server.

Every rollout runs an agent program whose OpenAI-style calls are caught here:
this small localhost server routes each `POST /v1/chat/completions` to our
`Client`, records a `Turn`, and returns the result in OpenAI shape. We inject
`OPENAI_BASE_URL`/`OPENAI_API_KEY` so the program's SDK talks to us. Chat
completions only, no streaming — only the model endpoint is intercepted (tools
and the user simulator are handled out-of-band for now).
"""

import secrets
from collections.abc import Awaitable, Callable

from aiohttp import web

from verifiers.nano.context import RolloutContext
from verifiers.nano.transcript import Transcript, Turn
from verifiers.nano.types import (
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


class InterceptionServer:
    """A localhost server that proxies one rollout's chat-completions calls."""

    def __init__(
        self,
        ctx: RolloutContext,
        transcript: Transcript,
        stops: list[Callable[[Transcript], Awaitable[bool]]] | None = None,
    ) -> None:
        self.ctx = ctx
        self.transcript = transcript
        self.stops = stops or []
        self.secret = secrets.token_urlsafe(16)
        self.port = 0
        self.runner: web.AppRunner | None = None

    async def __aenter__(self) -> "InterceptionServer":
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self.handle_chat)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]  # actual ephemeral port
        return self

    async def __aexit__(self, *exc) -> None:
        if self.runner is not None:
            await self.runner.cleanup()

    async def handle_chat(self, request: web.Request) -> web.Response:
        if request.headers.get("Authorization") != f"Bearer {self.secret}":
            return web.json_response({"error": "unauthorized"}, status=401)
        # A @stop firing here refuses the turn before it is served, which halts the
        # agent (its model call errors out); Agent.run treats that exit as clean.
        for stop in self.stops:
            if await stop(self.transcript):
                self.transcript.stop(stop.__name__)
                return web.json_response(
                    {"error": f"rollout stopped: {stop.__name__}"}, status=400
                )
        body = await request.json()
        prompt: Messages = [parse_message(m) for m in body.get("messages", [])]
        tools = parse_tools(body.get("tools"))
        try:
            response = await self.ctx.client.get_response(
                prompt, self.ctx.model, self.ctx.sampling, tools
            )
        except Exception as e:  # surface to the program as an API error
            return web.json_response({"error": str(e)}, status=502)
        self.transcript.trajectory.append(Turn(prompt=prompt, response=response))
        self.transcript.messages = [*prompt, response.message]  # full conversation
        return web.json_response(serialize_completion(response, self.ctx.model))
