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

import contextlib
import logging
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from aiohttp import web

from verifiers.v1.clients import RolloutContext
from verifiers.v1.dialects import DIALECTS, Dialect
from verifiers.v1 import graph
from verifiers.v1.errors import OverlongPromptError
from verifiers.v1.trace import Trace
from verifiers.v1.types import Messages

if TYPE_CHECKING:
    from verifiers.v1.user import Respond

logger = logging.getLogger(__name__)


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
    """A localhost server that proxies model calls for one or more rollouts. It serves every
    registered dialect's routes (see `dialects.DIALECTS`), so a request's wire format is resolved
    from the endpoint it arrived on — not declared by the harness. Each rollout `register`s a
    `RolloutSession` and gets back a secret; each request is routed to the session whose secret
    matches its bearer token. A single server can multiplex many rollouts (the basis for
    `interception.pool`); used 1:1 it's just a server with one session."""

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

    def _handler_for(self, dialect: Dialect):
        """Bind a route's dialect to the request handler — the route the SDK posts to is what
        selects the wire format (see `dialects.DIALECTS`)."""

        async def handler(request: web.Request) -> web.StreamResponse:
            return await self.handle_request(request, dialect)

        return handler

    def _aux_handler_for(self, dialect: Dialect, route: str):
        async def handler(request: web.Request) -> web.Response:
            return await self.handle_aux(request, dialect, route)

        return handler

    async def __aenter__(self) -> "InterceptionServer":
        app = web.Application(client_max_size=_MAX_REQUEST_BODY)
        for dialect in DIALECTS:
            for route in dialect.routes:
                app.router.add_post(route, self._handler_for(dialect))
            for aux in dialect.aux_routes:
                app.router.add_post(aux, self._aux_handler_for(dialect, aux))
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

    async def handle_request(
        self, request: web.Request, dialect: Dialect
    ) -> web.StreamResponse:
        session = self.sessions.get(dialect.secret(request.headers))
        if session is None:
            logger.warning("interception: unauthorized request")
            return web.json_response(dialect.error_body("unauthorized"), status=401)
        body = await request.json()
        # `body` is forwarded to the model 1:1 (the proxy mutates only model + sampling), so no
        # provider field is lost. `prompt` is the dialect's typed parse, kept only to build the
        # trace (the renderer re-derives its own from the body it's handed). A user simulator
        # extends both each turn (`dialect.extend` for the wire, `prompt` for the trace).
        prompt: Messages
        prompt, _ = dialect.parse_request(body)
        if dialect.streaming(body):
            return await self._stream(request, session, dialect, body, prompt)
        # A user simulator turns one program request into a multi-turn exchange: after each
        # model turn the simulator's reply is injected as a user turn and the model is
        # re-prompted, so a whole game plays out here and only the final assistant message
        # returns to the (simulator-unaware) program. Without a simulator the loop runs once.
        completion: dict | None = (
            None  # the latest turn's response, returned to the program
        )
        while True:
            refused = await session.refused()
            if refused is not None:
                # Refuse the first model call to halt the harness; once a simulated
                # conversation is under way, just end it and return the last turn cleanly.
                if completion is None:
                    return web.json_response(
                        dialect.error_body(f"rollout stopped: {refused}"), status=400
                    )
                return web.json_response(completion)
            try:
                response = await session.ctx.client.get_response(
                    dialect,
                    body,
                    session.ctx.model,
                    session.ctx.sampling,
                    request.headers,
                )
            except OverlongPromptError:
                # An overlong prompt is a budget limit, not a crash: end the rollout cleanly
                # as a truncation — return the last turn if there is one, else refuse to halt
                # the harness (same shape as `refused` above).
                session.trace.stop("context_length")
                logger.debug("prompt too long: id=%s", session.trace.id)
                if completion is None:
                    return web.json_response(
                        dialect.error_body("rollout stopped: context_length"),
                        status=400,
                    )
                return web.json_response(completion)
            except Exception as e:  # surface to the program as an API error
                logger.warning("model call failed: id=%s %s", session.trace.id, e)
                return web.json_response(dialect.error_body(str(e)), status=502)
            # `Response.raw` is the wire response handed to the program 1:1 — the provider's
            # verbatim bytes (proxy) or the client's serialized completion (renderer).
            completion = response.raw
            graph.add_turn(session.trace, prompt, response)  # one node per new message;
            # branches fall out of walking the graph (see Trace.branches / verifiers.v1.graph)
            # Hand back to the program when the model wants a tool (the program runs it) or
            # when there's no user simulator to keep the conversation going.
            if response.message.tool_calls or session.user is None:
                return web.json_response(completion)
            user_messages, done = await session.user(response.message.content or "")
            if done:
                session.trace.stop("user_completed")
                return web.json_response(completion)
            # Inject the model turn + the simulator's user turn(s): into the wire request for the
            # next model call (`dialect.extend`, which keeps the model turn verbatim so reasoning
            # survives) and into the typed prompt for the trace.
            body = dialect.extend(body, completion, user_messages)
            prompt = [*prompt, response.message, *user_messages]

    async def _stream(
        self,
        request: web.Request,
        session: RolloutSession,
        dialect: Dialect,
        body: dict,
        prompt: Messages,
    ) -> web.StreamResponse:
        """A streamed (SSE) model turn: relay the provider's stream through to the program,
        accumulating the bytes to record the turn on the trace. Single-shot — a streamed turn
        never drives a user simulator (the only client that streams is the eval relay)."""
        refused = await session.refused()
        if refused is not None:
            return web.json_response(
                dialect.error_body(f"rollout stopped: {refused}"), status=400
            )
        try:
            reply = await session.ctx.client.relay(
                dialect,
                body,
                session.ctx.model,
                session.ctx.sampling,
                request.headers,
            )
        except OverlongPromptError:
            session.trace.stop("context_length")
            logger.debug("prompt too long: id=%s", session.trace.id)
            return web.json_response(
                dialect.error_body("rollout stopped: context_length"), status=400
            )
        except Exception as e:  # surface to the program as an API error
            logger.warning("model call failed: id=%s %s", session.trace.id, e)
            return web.json_response(dialect.error_body(str(e)), status=502)
        buffer = bytearray()
        async for chunk in reply.chunks:
            buffer += chunk
        graph.add_turn(session.trace, prompt, dialect.parse_stream(bytes(buffer)))
        resp = web.StreamResponse()
        resp.content_type = reply.content_type.split(";")[0].strip()
        await resp.prepare(request)
        with contextlib.suppress(ConnectionResetError):
            await resp.write(bytes(buffer))
            await resp.write_eof()
        return resp

    async def handle_aux(
        self, request: web.Request, dialect: Dialect, route: str
    ) -> web.Response:
        """A non-model-turn side request (an `aux_route`, e.g. Anthropic's `count_tokens`):
        relayed verbatim to the provider, never recorded on the trace."""
        session = self.sessions.get(dialect.secret(request.headers))
        if session is None:
            return web.json_response(dialect.error_body("unauthorized"), status=401)
        try:
            result = await session.ctx.client.relay_aux(
                dialect, route, await request.json()
            )
        except Exception as e:
            logger.warning("aux call failed: id=%s %s", session.trace.id, e)
            return web.json_response(dialect.error_body(str(e)), status=502)
        return web.json_response(result)
