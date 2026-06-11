"""The interception server: harness chat-completions, caught and proxied.

Every rollout runs an harness program whose OpenAI-style calls are caught here: a small
localhost server routes each `POST /v1/chat/completions` to our `Client`, records the turn
into the trace's message graph, and returns the result in OpenAI shape. We inject
`OPENAI_BASE_URL`/`OPENAI_API_KEY` so the program's SDK talks to us. Chat completions only,
no streaming.

One server multiplexes many rollouts: each rollout registers a `RolloutSession` under its
own secret (the bearer token the harness already sends), and the server routes by that
secret to the right session. So N rollouts need one server (and, behind a remote runtime,
one tunnel) per pool member rather than one each — see `interception.pool`.

When a rollout sets a user simulator (see `verifiers.v1.user`), the session also drives it:
after each model turn it injects the simulator's reply as a user turn and re-prompts the
model, so a multi-turn exchange plays out within one program request, transparently to the
harness. Tools are handled out-of-band (run by the harness).
"""

import logging
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from aiohttp import web

from verifiers.v1.clients import RolloutContext
from verifiers.v1 import graph
from verifiers.v1.errors import OverlongPromptError
from verifiers.v1.trace import Trace
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
    content_to_parts,
)

if TYPE_CHECKING:
    from verifiers.v1.user import Respond

logger = logging.getLogger(__name__)


def _content_text(content) -> str:
    """Flatten content to text — for roles that never carry images (assistant, tool)."""
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def parse_message(raw: dict) -> Message:
    """An OpenAI request message dict -> a typed Message. User/system bodies keep their
    image parts (multimodal ingress); assistant/tool bodies flatten to text."""
    role = raw.get("role")
    content = raw.get("content")
    if role == "system":
        return SystemMessage(content=content_to_parts(content))
    if role == "tool":
        return ToolMessage(
            tool_call_id=raw.get("tool_call_id", ""), content=_content_text(content)
        )
    if role == "assistant":
        calls = [
            ToolCall(
                id=c["id"],
                name=c["function"]["name"],
                arguments=c["function"]["arguments"],
            )
            for c in (raw.get("tool_calls") or [])
        ] or None
        return AssistantMessage(
            content=_content_text(content) or None, tool_calls=calls
        )
    return UserMessage(content=content_to_parts(content))


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


# Each session proxies one rollout's own harness requests, so aiohttp's default 1 MiB body
# cap is an artificial bottleneck — a large tool result (e.g. a `cat` of a big file) trips it
# and the harness gets a 413. Allow large bodies; the upstream provider and the model's
# context window are the real limits, this is just a host-OOM backstop.
_MAX_REQUEST_BODY = 1024**3  # 1 GiB (aiohttp's default is 1 MiB)


@dataclass(frozen=True)
class RolloutLimits:
    """Per-rollout framework limits (None = no cap), checked before each turn is served.
    The first limit reached refuses the turn — halting any harness, the same mechanism as
    a @stop — and becomes the trace's stop condition. Each caps a trace computed property:
    `max_turns` -> num_turns, `max_input_tokens` -> prompt_len, `max_output_tokens` ->
    completion_len, `max_total_tokens` -> total_tokens. Token caps are soft by one turn:
    they're checked between turns, so the turn that crosses a cap still completes."""

    max_turns: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_total_tokens: int | None = None

    def reached(self, trace: Trace) -> str | None:
        """The name of the first limit `trace` has reached, or None if within all caps."""
        if self.max_turns is not None and trace.num_turns >= self.max_turns:
            return "max_turns"
        if (
            self.max_input_tokens is not None
            and trace.prompt_len >= self.max_input_tokens
        ):
            return "max_input_tokens"
        if (
            self.max_output_tokens is not None
            and trace.completion_len >= self.max_output_tokens
        ):
            return "max_output_tokens"
        if (
            self.max_total_tokens is not None
            and trace.total_tokens >= self.max_total_tokens
        ):
            return "max_total_tokens"
        return None


@dataclass
class RolloutSession:
    """One rollout's interception state, served by a (possibly shared) `InterceptionServer`
    and keyed there by its secret. Holds everything a single rollout's chat-completions need:
    the client/model context, the trace to record turns onto, the framework limits + `@stop`s
    checked before each turn, and (optionally) a user simulator the rollout sets before the
    harness runs."""

    ctx: RolloutContext
    trace: Trace
    stops: list[Callable[[Trace], Awaitable[bool]]] = field(default_factory=list)
    limits: RolloutLimits = field(default_factory=RolloutLimits)
    user: "Respond | None" = None
    """A user simulator the rollout sets before the harness runs (see `verifiers.v1.user`).
    When set, each model turn with no tool call is followed by the simulator's reply,
    injected as a user turn, and the model is re-prompted — all within one program request,
    transparently to the harness."""

    async def refused(self) -> str | None:
        """The framework's limits (turns / token budget) and `@stop` checks, run before each
        model call. Sets the stop condition and returns its name, else None. A refused first
        call halts the harness (its model call errors out); Harness.run treats it as clean."""
        if (limit := self.limits.reached(self.trace)) is not None:
            self.trace.stop(limit)
            logger.debug("limit %r reached: id=%s", limit, self.trace.id)
            return limit
        for stop in self.stops:
            if await stop(self.trace):
                self.trace.stop(stop.__name__)
                logger.debug("stop %r fired: id=%s", stop.__name__, self.trace.id)
                return stop.__name__
        return None


class InterceptionServer:
    """A localhost server that proxies chat-completions for one or more rollouts. Each
    rollout `register`s a `RolloutSession` and gets back a secret; `handle_chat` routes every
    request to the session whose secret matches the request's bearer token. A single server
    can multiplex many rollouts (the basis for `interception.pool`); used 1:1 it's just a
    server with one session."""

    def __init__(self) -> None:
        self.sessions: dict[str, RolloutSession] = {}
        self.port = 0
        self.runner: web.AppRunner | None = None

    def register(self, session: RolloutSession) -> str:
        """Add a session under a fresh secret (the bearer token the harness must send) and
        return it."""
        secret = secrets.token_urlsafe(16)
        self.sessions[secret] = session
        return secret

    def unregister(self, secret: str) -> None:
        self.sessions.pop(secret, None)

    async def __aenter__(self) -> "InterceptionServer":
        app = web.Application(client_max_size=_MAX_REQUEST_BODY)
        app.router.add_post("/v1/chat/completions", self.handle_chat)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]  # actual ephemeral port
        logger.debug("interception up: port=%d", self.port)
        return self

    async def __aexit__(self, *exc) -> None:
        if self.runner is not None:
            await self.runner.cleanup()

    async def handle_chat(self, request: web.Request) -> web.Response:
        auth = request.headers.get("Authorization", "")
        session = self.sessions.get(auth.removeprefix("Bearer "))
        if session is None:
            logger.warning("interception: unauthorized request")
            return web.json_response({"error": "unauthorized"}, status=401)
        body = await request.json()
        prompt: Messages = [parse_message(m) for m in body.get("messages", [])]
        tools = parse_tools(body.get("tools"))
        # A user simulator turns one program request into a multi-turn exchange: after each
        # model turn the simulator's reply is injected as a user turn and the model is
        # re-prompted, so a whole game plays out here and only the final assistant message
        # returns to the (simulator-unaware) program. Without a simulator the loop runs once.
        last: Response | None = None
        while True:
            refused = await session.refused()
            if refused is not None:
                # Refuse the first model call to halt the harness; once a simulated
                # conversation is under way, just end it and return the last turn cleanly.
                if last is None:
                    return web.json_response(
                        {"error": f"rollout stopped: {refused}"}, status=400
                    )
                return web.json_response(serialize_completion(last, session.ctx.model))
            try:
                response = await session.ctx.client.get_response(
                    prompt, session.ctx.model, session.ctx.sampling, tools
                )
            except OverlongPromptError:
                # An overlong prompt is a budget limit, not a crash: end the rollout cleanly
                # as a truncation — return the last turn if there is one, else refuse to halt
                # the harness (same shape as `refused` above).
                session.trace.stop("context_length")
                logger.debug("prompt too long: id=%s", session.trace.id)
                if last is None:
                    return web.json_response(
                        {"error": "rollout stopped: context_length"}, status=400
                    )
                return web.json_response(serialize_completion(last, session.ctx.model))
            except Exception as e:  # surface to the program as an API error
                logger.warning("model call failed: id=%s %s", session.trace.id, e)
                return web.json_response({"error": str(e)}, status=502)
            graph.add_turn(session.trace, prompt, response)  # one node per new message;
            # branches fall out of walking the graph (see Trace.branches / verifiers.v1.graph)
            last = response
            # Hand back to the program when the model wants a tool (the program runs it) or
            # when there's no user simulator to keep the conversation going.
            if response.message.tool_calls or session.user is None:
                return web.json_response(
                    serialize_completion(response, session.ctx.model)
                )
            user_messages, done = await session.user(response.message.content or "")
            if done:
                session.trace.stop("user_completed")
                return web.json_response(
                    serialize_completion(response, session.ctx.model)
                )
            prompt = [*prompt, response.message, *user_messages]
