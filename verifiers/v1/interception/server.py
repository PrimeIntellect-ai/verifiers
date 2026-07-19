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

When a rollout sets a user simulator (see `verifiers.v1.mcp.user`), the session also drives it:
after each model turn it injects the simulator's reply as a user turn and re-prompts the
model, so a multi-turn exchange plays out within one program request, transparently to the
harness. When the row carries no prompt (`TaskData.prompt is None`), the simulator also
opens the conversation: its first turn is seeded before the model is ever called. Tools are
handled out-of-band (run by the harness).
"""

import asyncio
import contextlib
import hashlib
import json
import logging
import secrets
import tempfile
import time
import traceback
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
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
    UserError,
    model_error,
)
from verifiers.v1.interception.base import BaseInterceptionConfig, Interception, Slot
from verifiers.v1.interception.tunnel import (
    PrimeTunnelConfig,
    Tunnel,
    TunnelConfig,
    make_tunnel,
)
from verifiers.v1.session import RolloutSession
from verifiers.v1.trace import Error, ModelCall, TimeSpan
from verifiers.v1.types import FinishReason, Messages, Response, Tool, Usage

logger = logging.getLogger(__name__)


# Each session proxies one rollout's own harness requests, so aiohttp's default 1 MiB body
# cap is an artificial bottleneck — a large tool result (e.g. a `cat` of a big file) trips it
# and the harness gets a 413. Allow large bodies; the upstream provider and the model's
# context window are the real limits, this is just a host-OOM backstop.
_MAX_REQUEST_BODY = 1024**3  # 1 GiB (aiohttp's default is 1 MiB)
# Atomic stream delivery needs replay storage: keep the common case in memory, spill larger
# completions to disk, and reject a broken/unbounded provider before it can exhaust the host.
_STREAM_SPOOL_MEMORY_BYTES = 1024**2
_MAX_STREAM_RESPONSE_BYTES = 64 * 1024**2
_KEEPALIVE_INTERVAL_SECONDS = 3
# blake2b saturates ~1.7 GB/s, so a body up to this size hashes inline in well under a
# millisecond; a larger one (bodies may reach `_MAX_REQUEST_BODY`) is hashed off the event
# loop instead — see `_request_digest`.
_HASH_INLINE_MAX = 1024**2  # 1 MiB


@dataclass
class _StreamReplay:
    """One validated native SSE response retained for exact SDK retries."""

    request: bytes
    content_type: str
    events: tempfile.SpooledTemporaryFile[bytes]
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def write(self, response: web.StreamResponse, start: int = 0) -> None:
        # A retry can arrive while the first client is still receiving the terminal bytes.
        # Serialize readers over the shared spool; cached retries start from byte zero while
        # the owner resumes at the withheld terminal event after committing.
        async with self.lock:
            self.events.seek(start)
            while event := self.events.read(64 * 1024):
                await response.write(event)

    async def close(self) -> None:
        async with self.lock:
            self.events.close()


def _body_digest(raw: bytes) -> bytes:
    return hashlib.blake2b(raw, digest_size=16).digest()


async def _request_digest(raw: bytes) -> bytes:
    """Digest a request body for the retry-replay guard. Hash a small body inline; offload a
    large one to a thread so it does not stall every multiplexed rollout on the event loop
    (blake2b releases the GIL, so the thread runs the hash off the loop)."""
    if len(raw) <= _HASH_INLINE_MAX:
        return _body_digest(raw)
    return await asyncio.to_thread(_body_digest, raw)


def _completion_response(completion: dict | None) -> web.Response:
    """Serialize a model's JSON-native response without an intermediate string."""
    try:
        body = to_json(completion, inf_nan_mode="constants")
    except PydanticSerializationError:
        return web.json_response(completion)
    return web.Response(body=body, content_type="application/json", charset="utf-8")


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
    single-server `Interception` (the pools compose several of these). With
    `requires_tunnel` (some consumer is off the host network) it mints its configured
    tunnel; on `start` it then binds where the tunnel says (`bind_host`/`bind_port`) and
    sets `base_url` — the one URL every consumer reaches it at — to the tunnel's public
    URL. Without, every consumer is on the host network: it binds loopback, tunnel-free."""

    def __init__(
        self,
        config: InterceptionServerConfig | None = None,
        requires_tunnel: bool = False,
    ) -> None:
        super().__init__()
        self.sessions: dict[str, RolloutSession] = {}
        self.requests: dict[str, set[asyncio.Task]] = {}
        self.stream_requests: dict[
            str, dict[bytes, asyncio.Future[_StreamReplay | None]]
        ] = {}
        self.stream_replays: dict[str, _StreamReplay] = {}
        self.replay_cleanup: set[asyncio.Task[None]] = set()
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
        self.requests[secret] = set()
        self.stream_requests[secret] = {}
        return secret

    async def cancel(self, secret: str) -> None:
        """Close model admission, then cancel every handler the slot already accepted."""
        requests = tuple(self.requests.pop(secret, set()))
        for request in requests:
            request.cancel()
        await asyncio.gather(*requests, return_exceptions=True)

    async def release(self, secret: str) -> None:
        await self.cancel(secret)
        self.stream_requests.pop(secret, None)
        replay = self.stream_replays.pop(secret, None)
        if replay is not None:
            await replay.close()
        self.sessions.pop(secret, None)

    def _retire_replay(self, secret: str, replay: _StreamReplay) -> None:
        """Close an obsolete spool after handlers that could have captured it finish."""
        current = asyncio.current_task()
        readers = tuple(
            request
            for request in self.requests.get(secret, ())
            if request is not current
        )

        async def close() -> None:
            await asyncio.gather(*readers, return_exceptions=True)
            await replay.close()

        cleanup = asyncio.create_task(close())
        self.replay_cleanup.add(cleanup)
        cleanup.add_done_callback(self.replay_cleanup.discard)

    async def stop(self) -> None:
        await super().stop()
        await asyncio.gather(*self.replay_cleanup, return_exceptions=True)

    @asynccontextmanager
    async def acquire(self, session: RolloutSession) -> AsyncIterator[Slot]:
        secret = self.register(session)
        try:
            yield Slot(self.base_url, secret, lambda: self.cancel(secret))
        finally:
            await self.release(secret)

    def _handler_for(self, dialect: Dialect):
        """Bind a route's dialect to the request handler — the route the SDK posts to is what
        selects the wire format (see `dialects.DIALECTS`)."""

        async def handler(request: web.Request) -> web.StreamResponse:
            secret = dialect.secret(request.headers)
            requests = self.requests.get(secret)
            if requests is None:
                if secret in self.sessions:
                    return web.json_response(
                        dialect.error_body("rollout stopped"), status=400
                    )
                return await self.handle_request(request, dialect)
            task = asyncio.current_task()
            assert task is not None
            requests.add(task)
            try:
                return await self.handle_request(request, dialect)
            finally:
                requests.discard(task)

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
        # The shared-state back-channel (see `verifiers.v1.state`): a rollout's tool/user servers
        # GET/PUT their `self.state` here, keyed by the same bearer secret as the model routes.
        app.router.add_get("/state", self.handle_state_get)
        app.router.add_put("/state", self.handle_state_put)
        # A launched tool/user server fetches its rollout's task here to run `setup_task` — the task
        # is never passed via env, only over this channel, keyed by the same bearer secret.
        app.router.add_get("/task", self.handle_task_get)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        self.stack.push_async_callback(self.runner.cleanup)
        # No tunnel → every consumer shares the host network: bind loopback on any ephemeral
        # port. Otherwise the tunnel says where to bind for it to reach the port, and
        # `expose` publishes it.
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
        """Stash a model-turn-adjacent failure (a `@stop` or user simulator raising) so the rollout
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
        node: int | None = None,
        finish_reason: "FinishReason" = None,
        usage: "Usage | None" = None,
        error: BaseException | None = None,
    ) -> None:
        """Append one provider exchange to the trace's per-call records (`Trace.calls`):
        the model + effective settings that went upstream, timing, and — when the call
        committed no turn — the error, coupled to the exchange that raised it. Called
        once per real exchange; replayed/coalesced SDK retries never reach it."""
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
                time=TimeSpan(start=started, end=time.time()),
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

    async def _replay_stream(
        self,
        request: web.Request,
        pending: _StreamReplay | asyncio.Future[_StreamReplay | None],
    ) -> web.StreamResponse:
        """Serve a completed stream cache, or wait on the owner of an identical request."""
        content_type = (
            pending.content_type
            if isinstance(pending, _StreamReplay)
            else "text/event-stream"
        )
        resp = web.StreamResponse(
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
        resp.content_type = content_type
        try:
            await resp.prepare(request)
            if isinstance(pending, asyncio.Future):
                while True:
                    try:
                        replay = await asyncio.wait_for(
                            asyncio.shield(pending), _KEEPALIVE_INTERVAL_SECONDS
                        )
                        break
                    except TimeoutError:
                        await resp.write(b": keepalive\n\n")
            else:
                replay = pending
            if replay is None:
                if request.transport is not None:
                    request.transport.close()
                return resp
            await replay.write(resp)
            await resp.write_eof()
        except ConnectionError:
            pass
        return resp

    async def handle_request(
        self, request: web.Request, dialect: Dialect
    ) -> web.StreamResponse:
        secret = dialect.secret(request.headers)
        session = self.sessions.get(secret)
        if session is None:
            logger.warning("interception: unauthorized request")
            return web.json_response(dialect.error_body("unauthorized"), status=401)
        raw = await request.read()
        try:
            body = from_json(raw)
        except ValueError:
            body = json.loads(raw)
        req_hash = await _request_digest(raw)
        # Keep `read()` for aiohttp's size guard, then release its cache and our local
        # alias after parsing so the wire body does not survive model inference.
        request._read_bytes = None
        del raw
        streaming = dialect.streaming(body)
        # Graph atomicity under retries. The harness SDK retries a transient failure by
        # re-sending the byte-identical request; sampling it again would commit a second turn and
        # fork the graph into a dead-end branch. Two cases, both resolved without re-sampling:
        #   1. the first attempt already finished -> replay the recorded response;
        #   2. the first attempt is still computing (a slow turn) -> await it and return its
        #      result, so a slow turn is safe without an inflated client timeout.
        # A growing conversation never repeats a body, so these only ever match a real retry; a
        # failed attempt caches nothing and re-runs normally.
        if (
            streaming
            and (replay := self.stream_replays.get(secret)) is not None
            and replay.request == req_hash
        ):
            logger.debug("intercept stream replay: id=%s", session.trace.id)
            return await self._replay_stream(request, replay)
        if (
            streaming
            and (streams := self.stream_requests.get(secret)) is not None
            and (pending := streams.get(req_hash)) is not None
        ):
            logger.debug("intercept stream coalesce: id=%s", session.trace.id)
            return await self._replay_stream(request, pending)
        if session.last_request == req_hash and session.last_response is not None:
            logger.debug("intercept replay: id=%s (retried request)", session.trace.id)
            return _completion_response(session.last_response)
        if session.trace.is_completed:
            stop = session.trace.stop_condition or "completed"
            return web.json_response(
                dialect.error_body(f"rollout stopped: {stop}"), status=400
            )

        async def coalesced(inflight: "asyncio.Future[dict | None]") -> web.Response:
            # Await the first attempt instead of re-sampling. None means it produced no servable
            # response (it errored/refused), so let the SDK retry afresh.
            logger.debug(
                "intercept coalesce: id=%s (retry of in-flight turn)", session.trace.id
            )
            completion = await inflight
            if completion is None:
                return web.json_response(
                    dialect.error_body("upstream attempt failed"), status=503
                )
            return _completion_response(completion)

        if (inflight := session.inflight.get(req_hash)) is not None:
            return await coalesced(inflight)
        logger.debug(
            "intercept %s: id=%s stream=%s",
            request.path,
            session.trace.id,
            streaming,
        )
        # The proxy preserves native JSON fields except model + sampling. `prompt` is only the
        # dialect's typed view for building the trace; the renderer re-derives its own from `body`.
        # A user simulator extends both each turn (`dialect.extend` for wire, `prompt` for trace).
        prompt: Messages
        # `tools` is recorded onto the trace only when a turn commits (below / in `_stream`):
        # the request is ground truth for what the model saw, but a refused or failed request
        # was never seen at all.
        prompt, tools = dialect.parse_request(body)
        # Cache the opening so retries do not advance the simulator twice.
        if (
            session.user is not None
            and session.trace.task.data.prompt is None
            and all(m.role != "assistant" for m in prompt)
        ):
            if session.opening is None:
                session.opening = await session.user("", len(prompt))
            body = dialect.extend(body, None, session.opening)
            prompt = [*prompt, *session.opening]
            # If the simulator ended at the open (its task's `@stop` now fires), the loop's
            # `refused()` below halts the harness before any model call — no special-casing here.
        if streaming:
            # Re-check after the simulator opening await above, then claim the request without
            # another await so two identical streams cannot both reach the provider.
            replay = self.stream_replays.get(secret)
            if replay is not None and replay.request == req_hash:
                return await self._replay_stream(request, replay)
            streams = self.stream_requests.get(secret)
            if streams is None or secret not in self.requests:
                return web.json_response(
                    dialect.error_body("rollout stopped"), status=400
                )
            if (pending := streams.get(req_hash)) is not None:
                return await self._replay_stream(request, pending)
            pending = asyncio.get_running_loop().create_future()
            streams[req_hash] = pending
            try:
                return await self._stream(
                    request,
                    session,
                    dialect,
                    body,
                    prompt,
                    tools,
                    secret=secret,
                    req_hash=req_hash,
                    attempt=pending,
                )
            finally:
                if streams.get(req_hash) is pending:
                    streams.pop(req_hash, None)
                if not pending.done():
                    pending.set_result(None)
        headers = request.headers.copy()
        # Claim the in-flight slot so a retry arriving mid-flight coalesces onto it (above) rather
        # than starting a second inference. Re-check first: an identical request may have claimed
        # it while we awaited the simulator opening. The get / create / assign below run with no
        # await between them, so two concurrent identical requests can never both become owner.
        if (inflight := session.inflight.get(req_hash)) is not None:
            return await coalesced(inflight)
        fut: asyncio.Future[dict | None] = asyncio.get_running_loop().create_future()
        session.inflight[req_hash] = fut

        async def serve(response: Response) -> web.Response:
            # Record the served turn and hand it to any coalesced retry, so a retried
            # byte-identical request replays instead of re-sampling and forking the graph.
            # `Response.raw` is the full native provider object (or the renderer's synthesized
            # completion) that the server serializes back to the program.
            session.last_request = req_hash
            session.last_response = response.raw
            if not fut.done():
                fut.set_result(response.raw)
            replay = self.stream_replays.pop(secret, None)
            if replay is not None:
                self._retire_replay(secret, replay)
            return _completion_response(response.raw)

        # A user simulator turns one program request into a multi-turn exchange: after each
        # model turn the simulator's reply is injected as a user turn and the model is
        # re-prompted, so a whole game plays out here and only the final assistant message
        # returns to the (simulator-unaware) program. Without a simulator the loop runs once.
        response: Response | None = (
            None  # the latest committed turn (None until the first)
        )
        try:
            while True:
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
                    # Refuse the first model call to halt the harness; once a simulated
                    # conversation is under way, just end it and return the last turn cleanly.
                    if response is None:
                        return web.json_response(
                            dialect.error_body(f"rollout stopped: {refused}"),
                            status=400,
                        )
                    return await serve(response)
                turn = graph.prepare_turn(session.trace, prompt)
                upstream_request: dict | None = None
                call_response: Response | None = None
                node: int | None = None
                error: Exception | None = None
                started = time.time()
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
                            headers=headers,
                            session_id=session.trace.id,
                            turn=turn,
                        )
                        logger.debug(
                            "intercept turn: id=%s tools=%d",
                            session.trace.id,
                            len(call_response.message.tool_calls or []),
                        )
                        if session.trace.is_completed or secret not in self.requests:
                            stop = session.trace.stop_condition or "completed"
                            return web.json_response(
                                dialect.error_body(f"rollout stopped: {stop}"),
                                status=400,
                            )
                        # One node per new message; branches fall out of walking the
                        # graph (see Trace.branches / verifiers.v1.graph).
                        node = turn.commit(call_response, tools)
                        session.error = None
                        response = call_response
                    except OverlongPromptError as e:
                        # An overlong prompt is a budget limit, not a crash: end the rollout cleanly
                        # as a truncation — return the last turn if there is one, else refuse to halt
                        # the harness (same shape as `refused` above).
                        error = e
                        session.trace.stop("context_length")
                        logger.debug("prompt too long: id=%s", session.trace.id)
                        if response is None:
                            return web.json_response(
                                dialect.error_body("rollout stopped: context_length"),
                                status=400,
                            )
                        return await serve(response)
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
                        node=node,
                        finish_reason=call_response.finish_reason
                        if call_response
                        else None,
                        usage=call_response.usage if call_response else None,
                        error=error,
                    )
                # Hand back to the program when the model wants a tool (the program runs it) or
                # when there's no user simulator to keep the conversation going.
                if response.message.tool_calls or session.user is None:
                    return await serve(response)
                prompt = [*prompt, response.message]
                try:
                    user_messages = await session.user(
                        response.message.content or "", len(prompt)
                    )
                except RolloutError as e:
                    return self._fail(session, dialect, e)
                except Exception as e:
                    return self._fail(
                        session,
                        dialect,
                        UserError(f"user simulator failed: {type(e).__name__}: {e}"),
                    )
                # Inject the model turn + the simulator's user turn(s): into the wire request for
                # the next model call (`dialect.extend`, which keeps the model turn verbatim so
                # reasoning survives) and into the typed prompt for the trace. The simulator ends
                # the trajectory through its task's `@stop` (e.g. a `user_finished` flag it set on
                # `self.state`), caught by `refused()` at the top of the next iteration — the
                # interception server holds no opinion about the state's contents.
                body = dialect.extend(body, response.raw, user_messages)
                prompt = [*prompt, *user_messages]
                # The simulator changed the payload, so this is a new operation not a retry.
                headers.popall("idempotency-key", None)
                headers.popall("x-idempotency-key", None)
        finally:
            # Free the in-flight slot and unblock any coalesced retry; None signals "no servable
            # response" (an error/refuse return above), so the waiter surfaces a retryable error.
            # Only clear our own entry — never one a later owner may have installed.
            if session.inflight.get(req_hash) is fut:
                session.inflight.pop(req_hash, None)
            if not fut.done():
                fut.set_result(None)

    async def _stream(
        self,
        request: web.Request,
        session: RolloutSession,
        dialect: Dialect,
        body: dict,
        prompt: Messages,
        tools: list[Tool] | None = None,
        *,
        secret: str,
        req_hash: bytes,
        attempt: asyncio.Future[_StreamReplay | None],
    ) -> web.StreamResponse:
        """A streamed (SSE) model turn: buffer and validate the provider stream, commit it to
        the trace, then replay its native events to the program. Single-shot — a streamed turn
        never drives a user simulator (the only client that streams is the eval relay)."""
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
            return web.json_response(
                dialect.error_body(f"rollout stopped: {refused}"), status=400
            )
        upstream_request: dict | None = None
        reply = None
        response: Response | None = None
        node: int | None = None
        error: Exception | None = None
        turn = graph.prepare_turn(session.trace, prompt)
        started = time.time()
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
                    dialect.error_body("rollout stopped: context_length"),
                    status=400,
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
                    dialect.error_body(str(e)),
                    status=getattr(e, "status_code", 502),
                )
            except Exception as e:  # surface to the program as an API error
                error = e
                logger.warning("model call failed: id=%s %s", session.trace.id, e)
                return web.json_response(dialect.error_body(str(e)), status=502)

            parser = dialect.stream_parser()
            resp = web.StreamResponse(
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
            content_type = reply.content_type.split(";")[0].strip()
            resp.content_type = content_type

            async def write_downstream(event: bytes) -> bool:
                nonlocal error
                try:
                    await resp.write(event)
                except ConnectionError as e:
                    error = e
                    return False
                return True

            events = tempfile.SpooledTemporaryFile(max_size=_STREAM_SPOOL_MEMORY_BYTES)
            cached = False
            try:
                size = 0
                terminal: int | None = None
                chunk: asyncio.Future[bytes] | None = None
                try:
                    try:
                        await resp.prepare(request)
                    except ConnectionError as e:
                        error = e
                        return resp
                    iterator = aiter(reply.chunks)
                    chunk = asyncio.ensure_future(anext(iterator))
                    loop = asyncio.get_running_loop()
                    next_keepalive = loop.time() + _KEEPALIVE_INTERVAL_SECONDS
                    while True:
                        if loop.time() >= next_keepalive:
                            if not await write_downstream(b": keepalive\n\n"):
                                return resp
                            next_keepalive = loop.time() + _KEEPALIVE_INTERVAL_SECONDS
                            continue
                        timeout = next_keepalive - loop.time()
                        done, _ = await asyncio.wait((chunk,), timeout=timeout)
                        if not done:
                            if not await write_downstream(b": keepalive\n\n"):
                                return resp
                            next_keepalive = loop.time() + _KEEPALIVE_INTERVAL_SECONDS
                            continue
                        try:
                            event = chunk.result()
                        except StopAsyncIteration:
                            break
                        chunk = asyncio.ensure_future(anext(iterator))
                        size += len(event)
                        if size > _MAX_STREAM_RESPONSE_BYTES:
                            raise ValueError(
                                f"stream exceeded {_MAX_STREAM_RESPONSE_BYTES} bytes"
                            )
                        if terminal is None and dialect.is_terminal_event(event):
                            terminal = events.tell()
                        events.write(event)
                        if parser.on_done is not None and is_sse_done_event(event):
                            parser.on_done()
                        parser.feed(event)
                    response = parser.finish()
                    # Probe the downstream once more after buffering. A client that left
                    # during a fast provider response must not produce a committed turn.
                    if not await write_downstream(b": keepalive\n\n"):
                        return resp
                except Exception as e:
                    session.error = model_error(
                        f"malformed upstream response: {type(e).__name__}: {e}",
                        status_code=502,
                    )
                    # Already-admitted siblings must not commit and clear this failure.
                    current = asyncio.current_task()
                    for sibling in tuple(self.requests.get(secret, ())):
                        if sibling is not current:
                            sibling.cancel()
                    error = session.error
                    logger.warning(
                        "model call failed: id=%s %s: %s",
                        session.trace.id,
                        type(session.error).__name__,
                        session.error,
                    )
                    if request.transport is not None:
                        request.transport.close()
                    return resp
                finally:
                    if chunk is not None:
                        chunk.cancel()
                        await asyncio.gather(chunk, return_exceptions=True)
                    await reply.close()

                # The provider response is now fully valid. Deliver everything before its
                # terminal event, but withhold the completion boundary until the trace commit.
                boundary = size if terminal is None else terminal
                events.seek(0)
                remaining = boundary
                try:
                    while remaining:
                        event = events.read(min(remaining, 64 * 1024))
                        await resp.write(event)
                        remaining -= len(event)
                except ConnectionError as e:
                    error = e
                    return resp
                if session.trace.is_completed or secret not in self.requests:
                    if request.transport is not None:
                        request.transport.close()
                    return resp
                node = turn.commit(response, tools)
                session.error = None
                session.last_request = req_hash
                session.last_response = None
                replay = _StreamReplay(req_hash, content_type, events)
                old = self.stream_replays.get(secret)
                self.stream_replays[secret] = replay
                cached = True  # the per-session retry cache owns the spool now
                if not attempt.done():
                    attempt.set_result(replay)
                logger.debug("intercept stream turn: id=%s", session.trace.id)
                if old is not None:
                    self._retire_replay(secret, old)
                with contextlib.suppress(ConnectionError):
                    await replay.write(resp, boundary)
                    await resp.write_eof()
                return resp
            finally:
                if not cached:
                    events.close()
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
        return self.sessions.get(secret)

    async def handle_state_get(self, request: web.Request) -> web.Response:
        """Hand a rollout's tool/user server the current shared `trace.state` (it pulls before each
        `@vf.tool`/`respond` call, so it sees writes from the other servers)."""
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
        """Hand a launched tool/user server the rollout's task (class ref + JSON) so it can run
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
        session.trace.state = new_state
        return web.json_response({"ok": True})
