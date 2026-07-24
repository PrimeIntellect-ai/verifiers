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
from tempfile import SpooledTemporaryFile
from typing import Literal

from aiohttp import web
from pydantic import TypeAdapter, ValidationError
from pydantic_core import PydanticSerializationError, from_json, to_json

from verifiers.v1.dialects import DIALECTS, Dialect
from verifiers.v1.dialects.base import is_sse_done_event
from verifiers.v1 import graph
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
    AssistantMessage,
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
_STREAM_MEMORY_BUFFER = 4 * 1024**2
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
        return web.Response(body=completion.body, content_type=completion.content_type)
    try:
        body = to_json(completion, inf_nan_mode="constants")
    except PydanticSerializationError:
        return web.json_response(completion)
    return web.Response(body=body, content_type="application/json", charset="utf-8")


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
        self.port = self.runner.addresses[0][1]  # actual bound port
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
            status=getattr(error, "status_code", 502),
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
        request_key = (request.path, await _request_digest(raw))
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
        # Preserve the zero-overhead live relay when no policy is installed. An
        # intercepted stream is buffered below so retries can replay the classified result.
        if streaming and not session.intercepts:
            prompt, tools = dialect.parse_request(body)
            return await self._stream(request, session, dialect, body, prompt, tools)

        # Graph atomicity under retries. Claim ownership before running any handler:
        # a concurrent retry then awaits this exact attempt instead of running a judge
        # twice, re-sampling, or committing a phantom branch.
        if session.last_request == request_key and session.last_response is not None:
            logger.debug("intercept replay: id=%s (retried request)", session.trace.id)
            return _completion_response(session.last_response)

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

        if (inflight := session.inflight.get(request_key)) is not None:
            return await coalesced(inflight)
        fut: asyncio.Future[ReplayResponse | None] = (
            asyncio.get_running_loop().create_future()
        )
        session.inflight[request_key] = fut

        try:
            if session.released:
                return web.json_response(
                    dialect.error_body("rollout concluded"), status=409
                )
            try:
                request_outcome = await session.run_intercepts("request", body, dialect)
            except RolloutError as e:
                return self._fail(session, dialect, e)
            if request_outcome.termination is not None:
                session.signal_termination(request_outcome.termination)
                return web.json_response(
                    dialect.error_body(
                        "rollout terminated: "
                        f"{request_outcome.termination.result.reason}"
                    ),
                    status=400,
                )

            # The typed prompt and tools are derived after request rewrites, so both
            # the model and trace see the same tool result.
            prompt: Messages
            prompt, tools = dialect.parse_request(body)
            if session.has_response_intercepts:
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

            if dialect.streaming(body):
                return await self._buffered_stream(
                    request,
                    session,
                    dialect,
                    body,
                    prompt,
                    request_key,
                    fut,
                    tools,
                )

            try:
                refused = await session.refused()
            except RolloutError as e:
                return self._fail(session, dialect, e)
            except Exception as e:
                return self._fail(
                    session,
                    dialect,
                    TaskError(f"@stop failed: {type(e).__name__}: {e}"),
                )
            if refused is not None:
                # Refuse the model call to halt the harness (it sees an HTTP error;
                # `Harness.run` treats a stopped rollout as the clean exit it is).
                return web.json_response(
                    dialect.error_body(f"rollout stopped: {refused}"),
                    status=400,
                )
            turn = graph.prepare_turn(session.trace, prompt)
            session.error = None
            upstream_request: dict | None = None
            call_response: Response | None = None
            delivered_message: AssistantMessage | None = None
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
                    if response_outcome.termination is not None:
                        # Commit the sampled violating action, but never serve it.
                        node = turn.commit(call_response, tools)
                        session.signal_termination(response_outcome.termination)
                        return web.json_response(
                            dialect.error_body(
                                "rollout terminated: "
                                f"{response_outcome.termination.result.reason}"
                            ),
                            status=400,
                        )
                    if response_outcome.rewritten:
                        try:
                            delivered_message = dialect.parse_response(
                                dialect.validate_response(call_response.raw)
                            ).message
                        except Exception as e:
                            raise TaskError(
                                "@intercept produced an invalid response: "
                                f"{type(e).__name__}: {e}"
                            ) from e
                        if call_response.id:
                            session.rewritten_response_ids.add(call_response.id)
                    # One node per new message; branches fall out of walking the
                    # graph. Commit the original typed sample, not the rewritten
                    # wire copy returned to the harness.
                    node = turn.commit(call_response, tools)
                    if delivered_message is not None:
                        session.trace.nodes[node].delivered_message = delivered_message
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
                    # Relay the provider's status so the harness SDK retries 5xx/429 and not 4xx.
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
                        status=getattr(e, "status_code", 502),
                    )
                except Exception as e:  # surface to the program as an API error
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
            raw_response = call_response.raw
            assert raw_response is not None
            session.last_request = request_key
            session.last_response = raw_response
            if not fut.done():
                fut.set_result(raw_response)
            return _completion_response(raw_response)
        finally:
            # Free the in-flight slot and unblock any coalesced retry; None signals "no servable
            # response" (an error/refuse return above), so the waiter surfaces a retryable error.
            # Only clear our own entry — never one a later owner may have installed.
            if session.inflight.get(request_key) is fut:
                session.inflight.pop(request_key, None)
            if not fut.done():
                fut.set_result(None)
            if session.terminated.is_set():
                session.termination_complete.set()

    async def _buffered_stream(
        self,
        request: web.Request,
        session: RolloutSession,
        dialect: Dialect,
        body: dict,
        prompt: Messages,
        request_key: RequestKey,
        fut: "asyncio.Future[ReplayResponse | None]",
        tools: list[Tool] | None = None,
    ) -> web.StreamResponse:
        """Classify a complete SSE turn before atomically committing and serving it."""
        try:
            refused = await session.refused()
        except RolloutError as e:
            return self._fail(session, dialect, e)
        except Exception as e:
            return self._fail(
                session, dialect, TaskError(f"@stop failed: {type(e).__name__}: {e}")
            )
        if refused is not None:
            return web.json_response(
                dialect.error_body(f"rollout stopped: {refused}"), status=400
            )

        session.error = None
        upstream_request: dict | None = None
        reply = None
        response: Response | None = None
        delivered_message: AssistantMessage | None = None
        node: int | None = None
        error: BaseException | None = None
        turn = graph.prepare_turn(session.trace, prompt)
        started = time.time()
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
                error = session.error = e
                logger.warning(
                    "model call failed: id=%s %s: %s",
                    session.trace.id,
                    type(e).__name__,
                    e,
                )
                return web.json_response(
                    dialect.error_body(str(e)),
                    status=getattr(e, "status_code", 502),
                )
            except Exception as e:
                error = e
                logger.warning("model call failed: id=%s %s", session.trace.id, e)
                return web.json_response(dialect.error_body(str(e)), status=502)

            resp = web.StreamResponse(
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
            resp.content_type = reply.content_type.split(";")[0].strip()
            parser = dialect.stream_parser()
            queue: asyncio.Queue[bytes | None] = asyncio.Queue(
                maxsize=_STREAM_QUEUE_MAXSIZE
            )
            ready = asyncio.Event()
            producer = asyncio.create_task(_queue_chunks(reply.chunks, queue, ready))
            parser_error: Exception | None = None

            async def keepalive() -> None:
                await resp.write(b": keepalive\n")

            with SpooledTemporaryFile(max_size=_STREAM_MEMORY_BUFFER) as buffer:
                try:
                    await resp.prepare(request)
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
                        buffer.write(chunk)
                        if parser_error is None:
                            try:
                                if parser.on_done is not None and is_sse_done_event(
                                    chunk
                                ):
                                    parser.on_done()
                                parser.feed(chunk)
                            except Exception as e:
                                parser_error = e
                except ConnectionResetError as e:
                    error = e
                    return resp
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
                    if response.raw is None:
                        raise ValueError("stream parser returned no native response")
                except Exception as e:
                    failure = ProviderError(
                        f"malformed upstream stream: {type(e).__name__}: {e}"
                    )
                    error = session.error = failure
                    logger.warning(
                        "stream parsing failed: id=%s %s",
                        session.trace.id,
                        failure,
                    )
                    if request.transport is not None:
                        request.transport.abort()
                    return resp
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
                        "stream interception failed: id=%s %s",
                        session.trace.id,
                        e,
                    )
                    if request.transport is not None:
                        request.transport.abort()
                    return resp
                finally:
                    if not interception.done():
                        interception.cancel()
                    await asyncio.gather(interception, return_exceptions=True)

                if outcome.termination is not None:
                    try:
                        await keepalive()
                    except ConnectionResetError as e:
                        error = e
                        return resp
                    node = turn.commit(response, tools)
                    session.signal_termination(outcome.termination)
                    if request.transport is not None:
                        request.transport.abort()
                    return resp

                if outcome.rewritten:
                    try:
                        delivered_message = dialect.parse_response(
                            dialect.validate_response(response.raw)
                        ).message
                        payload = b"".join(dialect.stream_events(response.raw))
                    except Exception as e:
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
                    if response.id:
                        session.rewritten_response_ids.add(response.id)
                else:
                    buffer.seek(0)
                    payload = buffer.read()

            # Probe connectivity, then publish replay ownership before any terminal
            # event reaches the client.
            try:
                await keepalive()
            except ConnectionResetError as e:
                error = e
                return resp
            node = turn.commit(response, tools)
            if delivered_message is not None:
                session.trace.nodes[node].delivered_message = delivered_message
            replay = StreamReplay(payload, resp.content_type)
            session.last_request = request_key
            session.last_response = replay
            if not fut.done():
                fut.set_result(replay)
            logger.debug("intercept stream turn: id=%s", session.trace.id)
            with contextlib.suppress(ConnectionResetError):
                for offset in range(0, len(payload), 64 * 1024):
                    await resp.write(payload[offset : offset + 64 * 1024])
                await resp.write_eof()
            return resp
        except BaseException as e:
            if node is None:
                error = e
            if request.transport is not None:
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

    async def _stream(
        self,
        request: web.Request,
        session: RolloutSession,
        dialect: Dialect,
        body: dict,
        prompt: Messages,
        tools: list[Tool] | None = None,
    ) -> web.StreamResponse:
        """A streamed (SSE) model turn: relay the provider's stream through to the program,
        incrementally assembling the response to record on the trace (the only client that
        streams is the eval relay)."""
        if session.released:  # concluded while this request queued — seal holds
            return web.json_response(
                dialect.error_body("rollout concluded"), status=409
            )
        try:
            refused = await session.refused()
        except RolloutError as e:
            return self._fail(session, dialect, e)
        except Exception as e:
            return self._fail(
                session, dialect, TaskError(f"@stop failed: {type(e).__name__}: {e}")
            )
        if refused is not None:
            return web.json_response(
                dialect.error_body(f"rollout stopped: {refused}"), status=400
            )
        session.error = None
        upstream_request: dict | None = None
        reply = None
        response: Response | None = None
        node: int | None = None
        error: BaseException | None = None
        turn = graph.prepare_turn(session.trace, prompt)
        started = time.time()
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
                session.error = e
                logger.warning(
                    "model call failed: id=%s %s: %s",
                    session.trace.id,
                    type(e).__name__,
                    e,
                )
                return web.json_response(
                    dialect.error_body(str(e)), status=getattr(e, "status_code", 502)
                )
            except Exception as e:  # surface to the program as an API error
                error = e
                logger.warning("model call failed: id=%s %s", session.trace.id, e)
                return web.json_response(dialect.error_body(str(e)), status=502)
            resp = web.StreamResponse(
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
            resp.content_type = reply.content_type.split(";")[0].strip()
            # Parse complete events as they relay, avoiding a full-stream byte copy.
            parser = dialect.stream_parser()
            feed_event = parser.feed
            on_done = parser.on_done
            # One bounded producer avoids per-event tasks; keepalive timeouts only cancel readiness waits.
            queue: asyncio.Queue[bytes | None] = asyncio.Queue(
                maxsize=_STREAM_QUEUE_MAXSIZE
            )
            ready = asyncio.Event()
            producer = asyncio.create_task(_queue_chunks(reply.chunks, queue, ready))
            parser_error: Exception | None = None
            # SSE events from the turn-ending one onward (the terminal event and any trailing
            # `[DONE]`), withheld until the turn is committed: a client that ends its turn on the
            # terminal event (e.g. codex on `response.completed`) would otherwise reach scoring
            # with the turn still unrecorded.
            deferred: list[bytes] = []
            try:
                await resp.prepare(request)
                while True:
                    try:
                        async with asyncio.timeout(_KEEPALIVE_INTERVAL_SECONDS):
                            await ready.wait()
                    except TimeoutError:
                        # Don't terminate an empty event; some SSE clients try to JSON-decode it.
                        await resp.write(b": keepalive\n")
                        continue
                    chunk = queue.get_nowait()
                    if queue.empty():
                        ready.clear()
                    if chunk is None:
                        await producer
                        break
                    # We send our own keepalive above. Some clients treat a complete
                    # comment-only event from upstream as an empty JSON payload.
                    if not any(
                        line.startswith(b"data:") for line in chunk.splitlines()
                    ):
                        await resp.write(b": keepalive\n")
                        continue
                    if deferred or dialect.is_terminal_event(chunk):
                        if parser_error is None:
                            try:
                                if on_done is not None and is_sse_done_event(chunk):
                                    on_done()
                                feed_event(chunk)
                            except Exception as e:
                                parser_error = e
                        # forwarded after the turn is committed, below
                        deferred.append(chunk)
                        continue
                    await resp.write(chunk)
                    if parser_error is None:
                        try:
                            feed_event(chunk)
                        except Exception as e:
                            parser_error = e
            except ConnectionResetError as e:
                # The harness went away mid-stream; the provider exchange still happened.
                error = e
                return resp
            finally:
                producer.cancel()
                # Let a canceled producer enqueue EOF while unwinding.
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
                if not session.released:  # concluded mid-stream — seal holds
                    node = turn.commit(response, tools)
                    logger.debug("intercept stream turn: id=%s", session.trace.id)
            finally:
                # Release the withheld events only now — after the commit — then close.
                with contextlib.suppress(ConnectionResetError):
                    for event in deferred:
                        await resp.write(event)
                    await resp.write_eof()
            return resp
        except BaseException as e:
            # Anything that propagates (a mid-relay upstream failure, a parser or commit
            # error, a cancellation) ends a real exchange; couple it to the record unless
            # the turn already committed (then only post-commit delivery failed).
            if node is None:
                error = e
            raise
        finally:
            # The turn's one per-exchange record: settings, timing, outcome, and the
            # error that ended it (if any).
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
        except Exception as e:
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
