"""The interception server: harness chat-completions, caught and proxied.

Every rollout runs an harness program whose OpenAI-style calls are caught here: a small
localhost server routes each `POST /v1/chat/completions` to our `Client`, records the turn
into the trace's message graph, and returns the result in OpenAI shape. We inject
`OPENAI_BASE_URL`/`OPENAI_API_KEY` so the program's SDK talks to us. Both non-streaming and
SSE requests are supported.

One server multiplexes many rollouts: each rollout registers a `RolloutSession` under its
own secret (the bearer token the harness already sends), and the server routes by that
secret to the right session. So N rollouts need one server (and, behind a remote runtime,
one tunnel) per pool member rather than one each — see `interception.pool`.

The server is a pure model boundary: one request, one turn — refusal checks (limits,
`@stop`s), the model call, the graph commit, retry atomicity. A run's user exchange
lives a layer up, between harness segments (see `verifiers.v1.rollout`); nothing
conversational happens here. Tools are handled out-of-band (run by the harness).
"""

import asyncio
import contextlib
import hashlib
import json
import logging
import secrets
import time
import traceback
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

from aiohttp import web
from aiohttp.payload import BufferedReaderPayload
from pydantic import TypeAdapter, ValidationError
from pydantic_core import PydanticSerializationError, from_json, to_json

from verifiers.v1 import graph
from verifiers.v1.dialects import DIALECTS, Dialect
from verifiers.v1.dialects.base import is_sse_done_event
from verifiers.v1.errors import (
    OverlongPromptError,
    ProviderError,
    RolloutError,
    TaskError,
)
from verifiers.v1.interception.base import BaseInterceptionConfig, Interception, Slot
from verifiers.v1.interception.tunnel import (
    PrimeTunnelConfig,
    Tunnel,
    TunnelConfig,
    make_tunnel,
)
from verifiers.v1.session import (
    ReplayResponse,
    RequestKey,
    RolloutSession,
    StreamReplay,
)
from verifiers.v1.trace import Error, ModelCall, TimeSpan
from verifiers.v1.types import (
    FinishReason,
    Messages,
    Response,
    Tool,
    Usage,
)

logger = logging.getLogger(__name__)


# Each session proxies one rollout's own harness requests, so aiohttp's default 1 MiB body
# cap is an artificial bottleneck — a large tool result (e.g. a `cat` of a big file) trips it
# and the harness gets a 413. Allow large bodies; the upstream provider and the model's
# context window are the real limits, this is just a host-OOM backstop.
_MAX_REQUEST_BODY = 1024**3  # 1 GiB (aiohttp's default is 1 MiB)
_KEEPALIVE_INTERVAL_SECONDS = 3
_STREAM_QUEUE_MAXSIZE = 16
# The Bash harness uses the OpenAI SDK's 600-second read timeout. Keep the replay through that
# full failure-detection window plus the SDK's bounded retry delay, while still expiring entries
# during long-running sessions.
_REPLAY_TTL_SECONDS = 660
# blake2b saturates ~1.7 GB/s, so a body up to this size hashes inline in well under a
# millisecond; a larger one (bodies may reach `_MAX_REQUEST_BODY`) is hashed off the event
# loop instead — see `_request_digest`.
_HASH_INLINE_MAX = 1024**2  # 1 MiB


def _body_digest(raw: bytes) -> bytes:
    return hashlib.blake2b(raw, digest_size=16).digest()


async def _request_digest(raw: bytes) -> bytes:
    """Digest a request body for the retry-replay guard. Hash a small body inline; offload a
    large one to a thread so it does not stall every multiplexed rollout on the event loop
    (blake2b releases the GIL, so the thread runs the hash off the loop)."""
    if len(raw) <= _HASH_INLINE_MAX:
        return _body_digest(raw)
    return await asyncio.to_thread(_body_digest, raw)


def _completion_response(completion: ReplayResponse) -> web.Response:
    """Serialize a model's JSON-native response without an intermediate string."""
    if isinstance(completion, StreamReplay):
        body = BufferedReaderPayload(
            completion.path.open("rb"),
            content_type=completion.content_type,
        )
        body.headers.pop("Content-Disposition", None)
        return web.Response(body=body)
    try:
        body = to_json(completion, inf_nan_mode="constants")
    except PydanticSerializationError:
        return web.json_response(completion)
    return web.Response(body=body, content_type="application/json", charset="utf-8")


def _expire_replay(session: RolloutSession, request_key: RequestKey) -> None:
    session.replay_expirations.pop(request_key, None)
    completion = session.replays.pop(request_key, None)
    if isinstance(completion, StreamReplay):
        completion.path.unlink(missing_ok=True)


def _retain_replay(
    session: RolloutSession, request_key: RequestKey, completion: ReplayResponse
) -> None:
    if expiration := session.replay_expirations.pop(request_key, None):
        expiration.cancel()
    session.replays[request_key] = completion
    session.replay_expirations[request_key] = asyncio.get_running_loop().call_later(
        _REPLAY_TTL_SECONDS, _expire_replay, session, request_key
    )


async def _queue_chunks(
    chunks: AsyncIterator[bytes],
    queue: asyncio.Queue[bytes | None],
    ready: asyncio.Event,
) -> None:
    try:
        async for chunk in chunks:
            await queue.put(chunk)
            ready.set()
    finally:
        await queue.put(None)
        ready.set()


class InterceptionServerConfig(BaseInterceptionConfig):
    """A single interception server shared by every rollout, reached (when any consumer is
    remote) via its `tunnel` — the shape that supports a bring-your-own endpoint
    (`tunnel.type custom`)."""

    type: Literal["server"] = "server"
    tunnel: TunnelConfig = PrimeTunnelConfig()
    """How remote consumers reach the server: `prime` (a framework-minted prime_tunnel) or
    `custom` (a pre-started tunnel / reverse proxy / direct bind you provide)."""


class InterceptionServer(Interception):
    """A server that proxies model calls for one or more rollouts — and is itself the
    single-server `Interception` (the pools compose several of these). When a consumer
    needs a public URL, it mints the configured tunnel and binds where that tunnel says;
    otherwise it stays on host loopback."""

    def __init__(
        self,
        config: InterceptionServerConfig | None = None,
        requires_tunnel: bool = False,
    ) -> None:
        super().__init__()
        self.sessions: dict[str, RolloutSession] = {}
        self.config = config or InterceptionServerConfig()
        self.tunnel: Tunnel | None = (
            make_tunnel(self.config.tunnel) if requires_tunnel else None
        )
        self.host = "127.0.0.1"
        self.port = 0
        self.base_url = ""  # set by `start`
        self.runner: web.AppRunner | None = None

    @property
    def load(self) -> int:
        """Rollouts currently registered — what the pools balance on."""
        return len(self.sessions)

    def register(self, session: RolloutSession) -> str:
        """Add a session under a fresh secret (the bearer token the harness must send) and
        return it."""
        secret = secrets.token_urlsafe(16)
        self.sessions[secret] = session
        return secret

    def unregister(self, secret: str) -> None:
        session = self.sessions.pop(secret, None)
        if session is not None:
            # The rollout concluded; its trace is sealed. Cancel straggler handlers
            # (aiohttp keeps them alive past client death) so a slow upstream call
            # can't commit a late turn onto the concluded trace.
            session.release()

    @asynccontextmanager
    async def acquire(self, session: RolloutSession) -> AsyncIterator[Slot]:
        secret = self.register(session)
        try:
            yield self.base_url, secret
        finally:
            self.unregister(secret)

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

    async def start(self) -> None:
        app = web.Application(client_max_size=_MAX_REQUEST_BODY)
        for dialect in DIALECTS:
            for route in dialect.routes:
                app.router.add_post(route, self._handler_for(dialect))
            for aux in dialect.aux_routes:
                app.router.add_post(aux, self._aux_handler_for(dialect, aux))
        # The shared-state back-channel (see `verifiers.v1.state`): a rollout's tool servers
        # GET/PUT their `self.state` here, keyed by the same bearer secret as the model routes.
        app.router.add_get("/state", self.handle_state_get)
        app.router.add_put("/state", self.handle_state_put)
        # A launched tool server fetches its rollout's task here to run `setup_task` — the task
        # is never passed via env, only over this channel, keyed by the same bearer secret.
        app.router.add_get("/task", self.handle_task_get)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        self.stack.push_async_callback(self.runner.cleanup)
        # Without a tunnel, local URL translation reaches an ephemeral loopback port.
        # Otherwise the tunnel determines the bind address and publishes it.
        if self.tunnel is None:
            self.host, bind_port = "127.0.0.1", 0
        else:
            self.host, bind_port = self.tunnel.bind_host, self.tunnel.bind_port
        site = web.TCPSite(self.runner, self.host, bind_port)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]  # actual bound port
        logger.info("interception up: url=http://%s:%d", self.host, self.port)
        self.stack.callback(
            logger.info, "interception down: url=http://%s:%d", self.host, self.port
        )
        if self.tunnel is None:
            self.base_url = f"http://127.0.0.1:{self.port}"
        else:
            self.base_url = await self.stack.enter_async_context(
                self.tunnel.expose(self.port)
            )

    def _fail(
        self, session: RolloutSession, dialect: Dialect, error: RolloutError
    ) -> web.Response:
        """Stash a model-turn-adjacent failure (a `@stop` raising) so the rollout
        re-raises it as the real cause, and report it to the harness as an HTTP error."""
        session.error = error
        logger.warning(
            "rollout %s failed: %s: %s", session.trace.id, type(error).__name__, error
        )
        return web.json_response(
            dialect.error_body(str(error)),
            status=getattr(error, "status_code", 400),
        )

    def record_call(
        self,
        session: RolloutSession,
        dialect: Dialect,
        request: dict | None,
        started: float,
        *,
        ended: float | None = None,
        node: int | None = None,
        finish_reason: "FinishReason" = None,
        usage: "Usage | None" = None,
        error: BaseException | None = None,
    ) -> None:
        """Append one provider exchange to the trace's per-call records (`Trace.calls`):
        the model + effective settings that went upstream, timing, and — when the call
        committed no turn — the error, coupled to the exchange that raised it. Called
        once per real exchange; replayed/coalesced SDK retries never reach it."""
        if (
            session.released
        ):  # the trace is sealed — a straggler exchange isn't recorded
            return
        sampling = None
        if request is not None:
            try:
                sampling = dialect.parse_sampling(request)
            except ValidationError:
                # A malformed harness knob must not kill recording (this runs in the
                # exchange's `finally`); the provider rejects the request on its own.
                logger.warning(
                    "unrecordable call settings: id=%s", session.trace.id, exc_info=True
                )
        session.trace.calls.append(
            ModelCall(
                node=node,
                model=request.get("model") if request is not None else None,
                sampling=sampling,
                endpoint=dialect.upstream_path,
                finish_reason=finish_reason,
                usage=usage,
                time=TimeSpan(
                    start=started,
                    end=ended if ended is not None else time.time(),
                ),
                error=None
                if error is None
                else Error(
                    type=type(error).__name__,
                    message=str(error),
                    status_code=getattr(error, "status_code", None),
                    # Provider errors already carry the actionable upstream diagnostic.
                    # Format from the exception object: the record is written in a
                    # `finally`, where the ambient exception state is already cleared.
                    traceback=None
                    if isinstance(error, ProviderError)
                    else "".join(traceback.format_exception(error)),
                ),
            )
        )

    async def handle_request(
        self, request: web.Request, dialect: Dialect
    ) -> web.StreamResponse:
        session = self.sessions.get(dialect.secret(request.headers))
        if session is None:
            logger.warning("interception: unauthorized request")
            return web.json_response(dialect.error_body("unauthorized"), status=401)
        session.adopt(asyncio.current_task())
        if session.terminated.is_set():
            return web.json_response(
                dialect.error_body("rollout terminated"), status=400
            )
        raw = await request.read()
        try:
            body = from_json(raw)
        except ValueError:
            body = json.loads(raw)
        request_id = request.headers.get("idempotency-key")
        request_key = (
            (request.path, await _request_digest(raw), request_id)
            if request_id is not None
            else None
        )
        # Keep `read()` for aiohttp's size guard, then release its cache and our local
        # alias after parsing so the wire body does not survive model inference.
        request._read_bytes = None
        del raw
        streaming = dialect.streaming(body)
        logger.debug(
            "intercept %s: id=%s stream=%s",
            request.path,
            session.trace.id,
            streaming,
        )
        # Graph atomicity under explicitly identified retries. An Idempotency-Key identifies
        # every attempt of one logical request; unkeyed byte-identical calls are independent
        # samples. Two retry cases are resolved without re-sampling:
        #   1. the first attempt already finished -> replay the recorded response;
        #   2. the first attempt is still computing (a slow turn) -> await it and return its
        #      result, so a slow turn is safe without an inflated client timeout.
        # A failed attempt caches nothing and re-runs normally.
        if (
            request_key is not None
            and (completion := session.replays.get(request_key)) is not None
        ):
            _retain_replay(session, request_key, completion)
            logger.debug("intercept replay: id=%s (retried request)", session.trace.id)
            return _completion_response(completion)

        async def coalesced(
            inflight: "asyncio.Future[ReplayResponse | None]",
        ) -> web.Response:
            # Await the first attempt instead of re-sampling. None means it produced no servable
            # response (it errored/refused), so let the SDK retry afresh.
            logger.debug(
                "intercept coalesce: id=%s (retry of in-flight turn)", session.trace.id
            )
            completion = await asyncio.shield(inflight)
            if completion is None:
                if session.terminated.is_set():
                    return web.json_response(
                        dialect.error_body("rollout terminated"), status=400
                    )
                return web.json_response(
                    dialect.error_body("upstream attempt failed"), status=503
                )
            return _completion_response(completion)

        replay: tuple[RequestKey, asyncio.Future[ReplayResponse | None]] | None = None
        if request_key is not None:
            if (inflight := session.inflight.get(request_key)) is not None:
                return await coalesced(inflight)
            fut: asyncio.Future[ReplayResponse | None] = (
                asyncio.get_running_loop().create_future()
            )
            session.inflight[request_key] = fut
            replay = request_key, fut

        def serve(response: Response) -> web.Response:
            # Record the served turn and hand it to any coalesced retry, so a retried
            # byte-identical request replays instead of re-sampling and forking the graph.
            # `Response.raw` is the full native provider object (or the renderer's synthesized
            # completion) that the server serializes back to the program.
            assert response.raw is not None
            if replay is not None:
                request_key, fut = replay
                _retain_replay(session, request_key, response.raw)
                if not fut.done():
                    fut.set_result(response.raw)
            return _completion_response(response.raw)

        try:
            if session.released:
                return web.json_response(
                    dialect.error_body("rollout concluded"), status=409
                )
            if isinstance(session.error, TaskError):
                return self._fail(session, dialect, session.error)
            try:
                refused = await session.refused()
            except RolloutError as e:
                return self._fail(session, dialect, e)
            except Exception as e:  # noqa: BLE001 - task hook boundary
                return self._fail(
                    session,
                    dialect,
                    TaskError(f"@stop failed: {type(e).__name__}: {e}"),
                )
            if refused is not None:
                return web.json_response(
                    dialect.error_body(f"rollout stopped: {refused}"), status=400
                )
            try:
                request_outcome = await session.run_intercepts("request", body, dialect)
            except RolloutError as e:
                return self._fail(session, dialect, e)
            if request_outcome.termination is not None:
                session.signal_termination(request_outcome.termination)
                return web.json_response(
                    dialect.error_body(
                        f"rollout terminated: {request_outcome.termination[1].reason}"
                    ),
                    status=400,
                )

            if dialect.streaming(body) != streaming:
                return self._fail(
                    session,
                    dialect,
                    TaskError("@intercept cannot change request streaming mode"),
                )

            # The typed prompt and tools are derived after request rewrites, so both
            # the model and trace see the same tool result.
            prompt: Messages
            try:
                prompt, tools = dialect.parse_request(body)
            except Exception as e:
                if not request_outcome.rewritten:
                    raise
                return self._fail(
                    session,
                    dialect,
                    TaskError(
                        "@intercept produced an invalid request: "
                        f"{type(e).__name__}: {e}"
                    ),
                )
            response_intercepts = session.has_response_intercepts
            if response_intercepts:
                previous = body.get("previous_response_id")
                if previous in session.rewritten_response_ids:
                    return self._fail(
                        session,
                        dialect,
                        TaskError(
                            "@intercept cannot continue a rewritten response by "
                            "previous_response_id; replay its output in the request"
                        ),
                    )
                if (
                    request.path == "/v1/responses"
                    and body.get("conversation") is not None
                ):
                    return self._fail(
                        session,
                        dialect,
                        TaskError(
                            "@intercept requires stateless Responses requests; conversation "
                            "state would retain the provider's original response"
                        ),
                    )
                if request.path == "/v1/chat/completions" and body.get("n", 1) != 1:
                    return self._fail(
                        session,
                        dialect,
                        TaskError(
                            "@intercept requires exactly one Chat Completions choice"
                        ),
                    )

            if streaming:
                return await self._stream(
                    request,
                    session,
                    dialect,
                    body,
                    prompt,
                    replay=replay,
                    tools=tools,
                    intercept_response=response_intercepts,
                )
            turn = graph.prepare_turn(session.trace, prompt)
            session.error = None
            upstream_request: dict | None = None
            call_response: Response | None = None
            node: int | None = None
            error: BaseException | None = None
            started = time.time()
            provider_ended: float | None = None
            try:
                try:
                    # What actually goes upstream: the native body with the rollout's model +
                    # sampling imposed — recorded raw on the trace, per call.
                    upstream_request = dialect.apply_overrides(
                        body, session.ctx.model, session.ctx.sampling
                    )
                    call_response = await session.ctx.client.get_response(
                        dialect,
                        body,
                        session.ctx.model,
                        session.ctx.sampling,
                        headers=request.headers,
                        session_id=session.trace.id,
                        turn=turn,
                    )
                    provider_ended = time.time()
                    logger.debug(
                        "intercept turn: id=%s tools=%d",
                        session.trace.id,
                        len(call_response.message.tool_calls or []),
                    )
                    if session.released:  # concluded while sampling — seal holds
                        return web.json_response(
                            dialect.error_body("rollout concluded"), status=409
                        )
                    assert call_response.raw is not None
                    response_outcome = await session.run_intercepts(
                        "response",
                        call_response.raw,
                        dialect,
                        prompt,
                    )
                    if session.released:
                        return web.json_response(
                            dialect.error_body("rollout concluded"), status=409
                        )
                    if response_outcome.termination is not None:
                        # Commit the sampled terminal action, but never serve it.
                        node = turn.commit(call_response, tools)
                        session.signal_termination(response_outcome.termination)
                        return web.json_response(
                            dialect.error_body(
                                "rollout terminated: "
                                f"{response_outcome.termination[1].reason}"
                            ),
                            status=400,
                        )
                    if response_outcome.rewritten:
                        try:
                            rewritten_response = dialect.parse_response(
                                dialect.validate_response(call_response.raw)
                            )
                            rewritten_response.tokens = call_response.tokens
                            rewritten_response.raw = call_response.raw
                        except Exception as e:
                            raise TaskError(
                                "@intercept produced an invalid response: "
                                f"{type(e).__name__}: {e}"
                            ) from e
                        if rewritten_response.id:
                            session.rewritten_response_ids.add(rewritten_response.id)
                        call_response = rewritten_response
                    # The harness-visible message is canonical for transcripts and
                    # scorers; the node records that interception replaced it.
                    node = turn.commit(call_response, tools, response_outcome.rewritten)
                except OverlongPromptError as e:
                    # An overlong prompt is a budget limit, not a crash: end the rollout
                    # cleanly as a truncation — refuse the call to halt the harness (same
                    # shape as `refused` above).
                    error = e
                    session.trace.stop("context_length")
                    logger.debug("prompt too long: id=%s", session.trace.id)
                    return web.json_response(
                        dialect.error_body("rollout stopped: context_length"),
                        status=400,
                    )
                except RolloutError as e:
                    # Stash the real cause; the rollout re-raises it after the harness returns.
                    # Provider errors carry their status; deterministic task errors are 400s.
                    error = e
                    session.error = e
                    logger.warning(
                        "model call failed: id=%s %s: %s",
                        session.trace.id,
                        type(e).__name__,
                        e,
                    )
                    return web.json_response(
                        dialect.error_body(str(e)),
                        status=getattr(e, "status_code", 400),
                    )
                except Exception as e:  # noqa: BLE001 - API boundary
                    error = e
                    logger.warning(
                        "model call failed: id=%s %s: %s",
                        session.trace.id,
                        type(e).__name__,
                        e,
                    )
                    return web.json_response(dialect.error_body(str(e)), status=502)
                except BaseException as e:
                    # A cancelled exchange (harness disconnect, shutdown) is still
                    # recorded, coupled to its cancellation.
                    error = e
                    raise
            finally:
                # The turn's one per-exchange record: settings, timing, outcome, and
                # the error that ended it (if any).
                self.record_call(
                    session,
                    dialect,
                    upstream_request,
                    started,
                    ended=provider_ended,
                    node=node,
                    finish_reason=call_response.finish_reason
                    if call_response
                    else None,
                    usage=call_response.usage if call_response else None,
                    error=error,
                )
            return serve(call_response)
        finally:
            # Free the in-flight slot and unblock any coalesced retry; None signals "no servable
            # response" (an error/refuse return above), so the waiter surfaces a retryable error.
            # Only clear our own entry — never one a later owner may have installed.
            if replay is not None:
                request_key, fut = replay
                if session.inflight.get(request_key) is fut:
                    session.inflight.pop(request_key, None)
                if not fut.done():
                    fut.set_result(None)
            if session.terminated.is_set():
                session.termination_complete.set()

    async def _stream(
        self,
        request: web.Request,
        session: RolloutSession,
        dialect: Dialect,
        body: dict,
        prompt: Messages,
        replay: tuple[RequestKey, "asyncio.Future[ReplayResponse | None]"] | None,
        tools: list[Tool] | None = None,
        intercept_response: bool = False,
    ) -> web.StreamResponse:
        """Relay one SSE turn, delaying response interception and retaining keyed replay."""
        if session.released:
            return web.json_response(
                dialect.error_body("rollout concluded"), status=409
            )

        session.error = None
        upstream_request: dict | None = None
        reply = None
        response: Response | None = None
        node: int | None = None
        error: BaseException | None = None
        turn = graph.prepare_turn(session.trace, prompt)
        started = time.time()
        buffer = (
            NamedTemporaryFile(prefix="vf-stream-replay-", delete=False)  # noqa: SIM115
            if replay is not None or intercept_response
            else None
        )
        replay_path = Path(buffer.name) if buffer is not None else None
        published = False
        provider_ended: float | None = None
        try:
            try:
                upstream_request = dialect.apply_overrides(
                    body, session.ctx.model, session.ctx.sampling
                )
                reply = await session.ctx.client.relay(
                    dialect,
                    body,
                    session.ctx.model,
                    session.ctx.sampling,
                    headers=request.headers,
                    session_id=session.trace.id,
                )
            except OverlongPromptError as e:
                error = e
                session.trace.stop("context_length")
                logger.debug("prompt too long: id=%s", session.trace.id)
                return web.json_response(
                    dialect.error_body("rollout stopped: context_length"), status=400
                )
            except RolloutError as e:
                error = e
                return self._fail(session, dialect, e)
            except Exception as e:  # noqa: BLE001 - API boundary
                error = e
                logger.warning("model call failed: id=%s %s", session.trace.id, e)
                return web.json_response(dialect.error_body(str(e)), status=502)

            resp = web.StreamResponse(
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
            resp.content_type = reply.content_type.split(";")[0].strip()
            parser = dialect.stream_parser()
            parser_error: Exception | None = None
            deferred: list[bytes] = []
            connected = True

            async def write(chunk: bytes) -> None:
                nonlocal connected
                if not connected:
                    return
                try:
                    await resp.write(chunk)
                except ConnectionResetError:
                    connected = False
                    if replay is None:
                        raise

            async def keepalive() -> None:
                await write(b": keepalive\n")

            async def finish_live() -> None:
                for event in deferred:
                    await write(event)
                if connected:
                    if replay is None:
                        await resp.write_eof()
                    else:
                        with contextlib.suppress(ConnectionResetError):
                            await resp.write_eof()

            def feed(chunk: bytes) -> None:
                nonlocal parser_error
                if parser_error is not None:
                    return
                try:
                    if parser.on_done is not None and is_sse_done_event(chunk):
                        parser.on_done()
                    parser.feed(chunk)
                except Exception as e:  # noqa: BLE001 - defer parser failure
                    parser_error = e

            queue: asyncio.Queue[bytes | None] = asyncio.Queue(
                maxsize=_STREAM_QUEUE_MAXSIZE
            )
            ready = asyncio.Event()
            # Timeouts cancel only this readiness wait, never the pending queue read.
            producer = asyncio.create_task(_queue_chunks(reply.chunks, queue, ready))
            try:
                try:
                    await resp.prepare(request)
                except ConnectionResetError:
                    connected = False
                    if replay is None:
                        raise
                while True:
                    try:
                        async with asyncio.timeout(_KEEPALIVE_INTERVAL_SECONDS):
                            await ready.wait()
                    except TimeoutError:
                        await keepalive()
                        continue
                    chunk = queue.get_nowait()
                    if queue.empty():
                        ready.clear()
                    if chunk is None:
                        await producer
                        break
                    if not any(
                        line.startswith(b"data:") for line in chunk.splitlines()
                    ):
                        # Some clients JSON-decode comment-only events as empty payloads.
                        await keepalive()
                        continue
                    # Retain model events for keyed replay, and delay them when response
                    # interception must classify the complete turn before delivery.
                    if buffer is not None:
                        buffer.write(chunk)
                    if not intercept_response and (
                        deferred or dialect.is_terminal_event(chunk)
                    ):
                        feed(chunk)
                        deferred.append(chunk)
                        continue
                    if not intercept_response:
                        await write(chunk)
                    feed(chunk)
            finally:
                producer.cancel()
                if queue.full():
                    queue.get_nowait()
                await asyncio.gather(producer, return_exceptions=True)
                try:
                    await reply.close()
                finally:
                    provider_ended = time.time()

            try:
                if parser_error is not None:
                    raise parser_error
                response = parser.finish()
                if intercept_response and response.raw is None:
                    raise ValueError("stream parser returned no native response")
            except Exception as e:
                if not intercept_response:
                    await finish_live()
                    raise
                failure = ProviderError(
                    f"malformed upstream stream: {type(e).__name__}: {e}"
                )
                error = session.error = failure
                logger.warning(
                    "stream parsing failed: id=%s %s", session.trace.id, failure
                )
                if request.transport is not None:
                    request.transport.abort()
                return resp

            if intercept_response:
                assert response.raw is not None
                interception = asyncio.create_task(
                    session.run_intercepts("response", response.raw, dialect, prompt)
                )
                try:
                    while not interception.done():
                        await asyncio.wait(
                            {interception}, timeout=_KEEPALIVE_INTERVAL_SECONDS
                        )
                        if not interception.done():
                            await keepalive()
                    outcome = await interception
                except RolloutError as e:
                    error = session.error = e
                    logger.warning(
                        "stream interception failed: id=%s %s", session.trace.id, e
                    )
                    if request.transport is not None:
                        request.transport.abort()
                    return resp
                finally:
                    if not interception.done():
                        interception.cancel()
                    await asyncio.gather(interception, return_exceptions=True)

                if outcome.termination is not None:
                    await keepalive()
                    node = turn.commit(response, tools)
                    session.signal_termination(outcome.termination)
                    if request.transport is not None:
                        request.transport.abort()
                    return resp

                if outcome.rewritten:
                    try:
                        assert buffer is not None
                        rewritten_response = dialect.parse_response(
                            dialect.validate_response(response.raw)
                        )
                        rewritten_response.tokens = response.tokens
                        rewritten_response.raw = response.raw
                        buffer.seek(0)
                        buffer.truncate()
                        for event in dialect.stream_events(response.raw):
                            buffer.write(event)
                    except Exception as e:  # noqa: BLE001 - interception boundary
                        failure = TaskError(
                            "@intercept produced an invalid response: "
                            f"{type(e).__name__}: {e}"
                        )
                        error = session.error = failure
                        logger.warning(
                            "stream interception failed: id=%s %s",
                            session.trace.id,
                            failure,
                        )
                        if request.transport is not None:
                            request.transport.abort()
                        return resp
                    if rewritten_response.id:
                        session.rewritten_response_ids.add(rewritten_response.id)
                    response = rewritten_response

                await keepalive()
            if session.released:
                if intercept_response:
                    if request.transport is not None:
                        request.transport.abort()
                else:
                    await finish_live()
                return resp

            try:
                stream_replay = None
                if replay is not None:
                    assert buffer is not None and replay_path is not None
                    buffer.flush()
                    if not intercept_response:
                        buffer.close()
                    stream_replay = StreamReplay(replay_path, resp.content_type)
                node = turn.commit(
                    response,
                    tools,
                    rewritten=intercept_response and outcome.rewritten,
                )
                if replay is not None:
                    request_key, fut = replay
                    assert stream_replay is not None
                    _retain_replay(session, request_key, stream_replay)
                    published = True
                    if not fut.done():
                        fut.set_result(stream_replay)
                logger.debug("intercept stream turn: id=%s", session.trace.id)
            except BaseException:
                if node is None:
                    # A client must not observe a successful terminal event for a turn that was
                    # never committed. Propagating the error closes this prepared response.
                    resp.force_close()
                elif not intercept_response:
                    # The turn is already durable; don't strand the live client if replay cache
                    # publication fails afterward.
                    await finish_live()
                raise
            else:
                if not intercept_response:
                    await finish_live()

            if not intercept_response:
                return resp
            assert buffer is not None
            with (
                contextlib.suppress(ConnectionResetError)
                if replay is not None
                else contextlib.nullcontext()
            ):
                buffer.seek(0)
                while chunk := buffer.read(64 * 1024):
                    await resp.write(chunk)
                await resp.write_eof()
            return resp
        except BaseException as e:
            if node is None:
                error = e
            if intercept_response and request.transport is not None:
                request.transport.abort()
            raise
        finally:
            self.record_call(
                session,
                dialect,
                upstream_request,
                started,
                ended=provider_ended,
                node=node,
                finish_reason=response.finish_reason if response is not None else None,
                usage=response.usage if response is not None else None,
                error=error,
            )
            if buffer is not None:
                buffer.close()
            if replay_path is not None and not published:
                replay_path.unlink(missing_ok=True)

    async def handle_aux(
        self, request: web.Request, dialect: Dialect, route: str
    ) -> web.Response:
        """A non-model-turn side request (an `aux_route`, e.g. Anthropic's `count_tokens`):
        relayed as native JSON, never recorded on the trace."""
        session = self.sessions.get(dialect.secret(request.headers))
        if session is None:
            return web.json_response(dialect.error_body("unauthorized"), status=401)
        session.adopt(asyncio.current_task())
        logger.debug("intercept aux %s: id=%s", route, session.trace.id)
        try:
            result = await session.ctx.client.relay_aux(
                dialect, route, await request.json(), headers=request.headers
            )
        except RolloutError as e:
            # An aux call isn't a model turn, so don't clobber a pending turn error.
            session.error = session.error or e
            logger.warning(
                "aux call failed: id=%s %s: %s",
                session.trace.id,
                type(e).__name__,
                e,
            )
            return web.json_response(
                dialect.error_body(str(e)), status=getattr(e, "status_code", 502)
            )
        except Exception as e:  # noqa: BLE001 - auxiliary API boundary
            logger.warning("aux call failed: id=%s %s", session.trace.id, e)
            return web.json_response(dialect.error_body(str(e)), status=502)
        return web.json_response(result)

    def _session_for(self, request: web.Request) -> RolloutSession | None:
        """The session a state request belongs to, by its `Authorization: Bearer <secret>` — the
        same per-rollout secret the model routes use (dialect-independent, so parsed directly)."""
        auth = request.headers.get("Authorization", "")
        secret = auth[len("Bearer ") :] if auth.startswith("Bearer ") else ""
        session = self.sessions.get(secret)
        if session is not None:  # state writes must not land on a sealed trace either
            session.adopt(asyncio.current_task())
        return session

    async def handle_state_get(self, request: web.Request) -> web.Response:
        """Hand a rollout's tool server the current shared `trace.state` (it pulls before each
        `@vf.tool` call, so it sees writes from the other servers)."""
        session = self._session_for(request)
        if session is None:
            return web.json_response({"error": "unauthorized"}, status=401)
        logger.debug("intercept GET /state: id=%s", session.trace.id)
        state = session.trace.state
        return web.Response(
            # TypeAdapter emits UTF-8 bytes directly, avoiding a JSON str copy in aiohttp.
            body=TypeAdapter(type(state)).dump_json(state),
            content_type="application/json",
            charset="utf-8",
        )

    async def handle_task_get(self, request: web.Request) -> web.Response:
        """Hand a launched tool server the rollout's task (class ref + JSON) so it can run
        `setup_task` for this rollout — keyed by the same bearer secret as the state channel."""
        session = self._session_for(request)
        if session is None:
            return web.json_response({"error": "unauthorized"}, status=401)
        logger.debug("intercept GET /task: id=%s", session.trace.id)
        task = session.trace.task.data
        return web.json_response(
            {
                "cls": f"{type(task).__module__}:{type(task).__qualname__}",
                "task": task.model_dump_json(),
            }
        )

    async def handle_state_put(self, request: web.Request) -> web.Response:
        """Replace a rollout's shared `trace.state` with a server's pushed copy (validated into the
        trace's `State` type). Last write wins per call. A task ends the trajectory from state via
        its own `@stop` (run in `RolloutSession.refused` before each model call)."""
        session = self._session_for(request)
        if session is None:
            return web.json_response({"error": "unauthorized"}, status=401)
        logger.debug("intercept PUT /state: id=%s", session.trace.id)
        state_cls = type(session.trace.state)
        raw = await request.read()
        try:
            new_state = state_cls.model_validate_json(raw)
        except ValidationError as e:
            # Reject malformed, over-nested, or mismatched state before it enters the shared channel.
            logger.warning("state PUT rejected: id=%s %s", session.trace.id, e)
            return web.json_response(
                {"error": f"invalid state PUT for {state_cls.__name__}: {e}"},
                status=400,
            )
        if session.released:  # the trace is sealed — a straggler write must not land
            return web.json_response({"error": "rollout concluded"}, status=409)
        session.trace.state = new_state
        return web.json_response({"ok": True})
