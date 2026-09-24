"""The interception server: harness chat-completions, caught and proxied.

Every rollout runs an harness program whose OpenAI-style calls are caught here: a small
localhost server routes each `POST /v1/chat/completions` to our `Client`, records the turn
into the trace's message graph, and returns the result in OpenAI shape. We inject
`OPENAI_BASE_URL`/`OPENAI_API_KEY` so the program's SDK talks to us. Both non-streaming and
SSE requests are supported.

One server multiplexes many rollouts: each rollout registers separate model and state
capabilities, and the server routes each to the right session. So N rollouts need one
server (and, behind a remote runtime, one tunnel) per pool member rather than one each —
see `interception.pool`. The server also owns the model clients (one per distinct endpoint
config, assigned to each session at register and closed with the server), so its rollouts
share one bounded keepalive connection pool upstream instead of churning per-rollout TCP.

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
from collections.abc import AsyncIterator, Awaitable, Collection, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal

import httpx
from aiohttp import web
from pydantic import ValidationError
from pydantic_core import PydanticSerializationError, from_json, to_json

from verifiers.v1 import graph
from verifiers.v1.clients import Client, resolve_client
from verifiers.v1.clients.base import join_url
from verifiers.v1.clients.client import RelayReply
from verifiers.v1.configs.client import (
    BaseClientConfig,
    TrainClientConfig,
    resolve_api_key,
)
from verifiers.v1.dialects import DIALECTS, Dialect
from verifiers.v1.dialects.policy import PROVIDER_CAPABILITY_POLICY_CODE, mediate
from verifiers.v1.errors import (
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
from verifiers.v1.semantic import ACPInfo, extract_acp_info
from verifiers.v1.session import IdempotentRequest, ReplayResponse, RolloutSession
from verifiers.v1.trace import Error, ModelCall, PolicyEvent, TimeSpan
from verifiers.v1.types import FinishReason, Response, Usage

logger = logging.getLogger(__name__)


# Each session proxies one rollout's own harness requests, so aiohttp's default 1 MiB body
# cap is an artificial bottleneck — a large tool result (e.g. a `cat` of a big file) trips it
# and the harness gets a 413. Allow large bodies; the upstream provider and the model's
# context window are the real limits, this is just a host-OOM backstop.
MAX_REQUEST_BODY = 1024**3  # 1 GiB (aiohttp's default is 1 MiB)
KEEPALIVE_INTERVAL_SECONDS = 3
# A streamed turn commits its SSE stream only after this long: a result within it keeps
# its HTTP status (so the harness SDK can retry 5xx/429), and a longer one is kept alive
# well inside the tunnel's response-header timeout.
KEEPALIVE_GRACE_SECONDS = 60
# blake2b saturates ~1.7 GB/s, so a body up to this size hashes inline in well under a
# millisecond; a larger one (bodies may reach `MAX_REQUEST_BODY`) is hashed off the event
# loop instead — see `_request_digest`.
HASH_INLINE_MAX = 1024**2  # 1 MiB
# Attempt counter the stainless-generated SDKs (OpenAI, Anthropic) send on every request:
# 0 on the first attempt, incremented on each retry of the same request.
RETRY_COUNT_HEADER = "x-stainless-retry-count"
IDEMPOTENCY_KEY_HEADER = "Idempotency-Key"
IDEMPOTENCY_CACHE_TTL_SECONDS = 600
IDEMPOTENCY_CACHE_MAX_COMPLETED = 64


def is_retried_request(headers: Mapping[str, str]) -> bool:
    try:
        return int(headers.get(RETRY_COUNT_HEADER, 0)) > 0
    except ValueError:
        return False


def _body_digest(raw: bytes) -> bytes:
    return hashlib.blake2b(raw, digest_size=16).digest()


async def _request_digest(raw: bytes) -> bytes:
    """Digest a request body for the retry-replay guard. Hash a small body inline; offload a
    large one to a thread so it does not stall every multiplexed rollout on the event loop
    (blake2b releases the GIL, so the thread runs the hash off the loop)."""
    if len(raw) <= HASH_INLINE_MAX:
        return _body_digest(raw)
    return await asyncio.to_thread(_body_digest, raw)


def _completion_response(completion: dict | None) -> web.Response:
    """Serialize a model's JSON-native response without an intermediate string."""
    try:
        body = to_json(completion, inf_nan_mode="constants")
    except PydanticSerializationError:
        return web.json_response(completion)
    return web.Response(body=body, content_type="application/json", charset="utf-8")


def _capture_response(response: web.Response) -> ReplayResponse:
    body = response.body
    if body is None:
        data = b""
    elif isinstance(body, bytes):
        data = body
    elif isinstance(body, bytearray):
        data = bytes(body)
    else:
        raise TypeError("coalesced interception responses must have a byte body")
    return ReplayResponse(
        status=response.status,
        body=data,
        content_type=response.headers["Content-Type"],
    )


def _replay_response(response: ReplayResponse) -> web.Response:
    return web.Response(
        body=response.body,
        status=response.status,
        headers={"Content-Type": response.content_type},
    )


@dataclass(frozen=True)
class _IdempotentAttempt:
    session: RolloutSession
    key: str
    request: IdempotentRequest
    future: asyncio.Future[ReplayResponse | None]


_IDEMPOTENT_ATTEMPT = web.RequestKey("idempotent_attempt", _IdempotentAttempt)


def _finish_idempotent_attempt(
    request: web.Request, response: ReplayResponse | None
) -> None:
    attempt = request.get(_IDEMPOTENT_ATTEMPT)
    if attempt is None:
        return
    record = attempt.request
    if record.inflight is attempt.future:
        record.inflight = None
    if (
        record.response is None
        and attempt.session.idempotent_requests.get(attempt.key) is record
    ):
        attempt.session.idempotent_requests.pop(attempt.key)
    if not attempt.future.done():
        attempt.future.set_result(response)
    if record.response is not None:
        _prune_idempotent_requests(attempt.session, time.monotonic())


def _prune_idempotent_requests(session: RolloutSession, now: float) -> None:
    """Expire and cap completed replays; active attempts are never evicted."""
    completed = [
        (key, request)
        for key, request in session.idempotent_requests.items()
        if request.response is not None and request.inflight is None
    ]
    for key, request in completed:
        completed_at = request.completed_at or 0.0
        if (
            now - completed_at >= IDEMPOTENCY_CACHE_TTL_SECONDS
            and session.idempotent_requests.get(key) is request
        ):
            session.idempotent_requests.pop(key)
    completed = [
        (key, request)
        for key, request in completed
        if session.idempotent_requests.get(key) is request
    ]
    excess = len(completed) - IDEMPOTENCY_CACHE_MAX_COMPLETED
    if excess > 0:
        for key, request in sorted(
            completed, key=lambda item: item[1].completed_at or 0.0
        )[:excess]:
            if session.idempotent_requests.get(key) is request:
                session.idempotent_requests.pop(key)


def _sse_data(raw: bytes) -> bytes | None:
    """One complete SSE event's data payload, or None for a comment-only event."""
    lines = [
        line.removeprefix(b"data:").strip()
        for line in raw.splitlines()
        if line.startswith(b"data:")
    ]
    return b"\n".join(lines) if lines else None


def _sse_json(data: bytes) -> dict:
    try:
        return from_json(data)
    except ValueError:
        logger.warning(
            "SSE JSON fast-path failed; falling back to stdlib with invalid UTF-8 replacement"
        )
        return json.loads(data.decode("utf-8", errors="replace"))


async def _collect_stream(
    dialect: Dialect, reply: RelayReply
) -> tuple[Response, bytes]:
    """Read a relayed provider stream whole: the assembled response and the events to serve
    for it. Comment-only events (the provider's own keepalives) are dropped; the served
    stream sends its own."""
    events = bytearray()
    parser = dialect.stream_parser()
    saw_terminal = False
    try:
        async for chunk in reply.chunks:
            data = _sse_data(chunk)
            if data is None:
                continue
            events += chunk
            if data == b"[DONE]":
                saw_terminal |= "[DONE]" in dialect.terminal_events
            elif data:
                event = _sse_json(data)
                # The provider SDKs raise on an event carrying `error`; so does the turn.
                if error := event.get("error"):
                    detail = (
                        error.get("message", error)
                        if isinstance(error, dict)
                        else error
                    )
                    raise ProviderError(f"upstream stream error: {detail}")
                saw_terminal |= event.get("type") in dialect.terminal_events
                parser.feed(event)
        if not saw_terminal:
            raise ProviderError("upstream stream ended before its terminal event")
        raw = parser.finish()
        response = dialect.parse_response(raw)
        response.raw = raw
        return response, bytes(events)
    except RolloutError:
        raise
    except Exception as e:  # a malformed provider stream
        raise ProviderError(str(e)) from e
    finally:
        await reply.close()


async def _buffered_stream(
    request: web.Request,
    dialect: Dialect,
    pending: Awaitable[web.Response],
    trace_id: str,
) -> web.StreamResponse:
    """Serve a turn to an SSE client once it is committed, keeping the connection alive
    while it is produced. A result within the grace period is served as is; after it the
    stream is committed and sent the dialect's keepalives, so neither a proxy's timeout
    nor a client's own idle timeout cuts a long turn. Once committed, a failure is framed
    as the dialect's SSE error. A reader that goes away leaves the turn running for its
    retries to coalesce onto."""
    task = asyncio.ensure_future(pending)
    started = time.monotonic()
    try:
        done, _ = await asyncio.wait({task}, timeout=KEEPALIVE_GRACE_SECONDS)
        if done:
            return task.result()
        stream = web.StreamResponse(
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
        connected = True
        try:
            await stream.prepare(request)
            first = True
            while not task.done():
                await stream.write(dialect.stream_keepalive(first))
                first = False
                await asyncio.wait({task}, timeout=KEEPALIVE_INTERVAL_SECONDS)
        except ConnectionResetError:
            # A reader that goes away mid-turn is the failure a tunnel or proxy drop looks
            # like from here; its retry (if any) coalesces onto this turn.
            connected = False
            logger.warning(
                "intercept stream: reader disconnected: id=%s after=%.1fs",
                trace_id,
                time.monotonic() - started,
            )
        response = await task
        replay = _capture_response(response)
        # Release coalesced retries now rather than after the write to this reader.
        _finish_idempotent_attempt(request, replay)
        if connected:
            try:
                await stream.write(
                    replay.body
                    if response.status < 400
                    else dialect.stream_error(from_json(replay.body))
                )
                await stream.write_eof()
            except ConnectionResetError:
                logger.warning(
                    "intercept stream: reader disconnected before the turn was served: "
                    "id=%s after=%.1fs",
                    trace_id,
                    time.monotonic() - started,
                )
        return stream
    finally:
        if not task.done():
            logger.info(
                "intercept stream: turn cancelled: id=%s after=%.1fs",
                trace_id,
                time.monotonic() - started,
            )
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


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
        state_service_secrets: Collection[str] = (),
    ) -> None:
        super().__init__()
        self.sessions: dict[str, RolloutSession] = {}
        self.clients: dict[str, Client] = {}
        self.state_sessions: dict[str, RolloutSession] = {}
        self.state_routes: dict[str, RolloutSession] = {}
        self.state_service_secrets = frozenset(state_service_secrets)
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

    def _client(self, config: BaseClientConfig) -> Client:
        """The server-owned client for `config` — one per distinct endpoint config, shared
        by every session registered under it, so the rollouts this server multiplexes reuse
        one bounded keepalive pool instead of each opening (and tearing down) their own
        connections. Closed with the server."""
        key = config.model_dump_json()
        client = self.clients.get(key)
        if client is None:
            client = self.clients[key] = resolve_client(config)
            self.stack.push_async_callback(client.close)
        return client

    def register(self, session: RolloutSession) -> tuple[str, str]:
        """Register separate capabilities for model inference and private task state, and
        assign the session its server-owned model client."""
        session.client = self._client(session.ctx.client)
        model_secret = secrets.token_urlsafe(16)
        state_secret = secrets.token_urlsafe(16)
        self.sessions[model_secret] = session
        self.state_sessions[state_secret] = session
        self.state_routes[session.trace.id] = session
        return model_secret, state_secret

    def unregister(self, model_secret: str, state_secret: str) -> None:
        session = self.sessions.pop(model_secret, None)
        self.state_sessions.pop(state_secret, None)
        if session is not None:
            self.state_routes.pop(session.trace.id, None)
            # The rollout concluded; its trace is sealed. Cancel straggler handlers
            # (aiohttp keeps them alive past client death) so a slow upstream call
            # can't commit a late turn onto the concluded trace.
            session.release()

    @asynccontextmanager
    async def acquire(self, session: RolloutSession) -> AsyncIterator[Slot]:
        model_secret, state_secret = self.register(session)
        try:
            yield self.base_url, model_secret, state_secret
        finally:
            self.unregister(model_secret, state_secret)

    def _handler_for(self, dialect: Dialect):
        """Bind a route's dialect to the request handler — the route the SDK posts to is what
        selects the wire format (see `dialects.DIALECTS`)."""

        async def handler(request: web.Request) -> web.StreamResponse:
            try:
                response = await self.handle_request(request, dialect)
            except BaseException:
                _finish_idempotent_attempt(request, None)
                raise
            replay = (
                _capture_response(response)
                if isinstance(response, web.Response)
                else None
            )
            _finish_idempotent_attempt(request, replay)
            return response

        return handler

    def _aux_handler_for(self, dialect: Dialect, route: str):
        async def handler(request: web.Request) -> web.Response:
            return await self.handle_aux(request, dialect, route)

        return handler

    async def start(self) -> None:
        app = web.Application(client_max_size=MAX_REQUEST_BODY)
        for dialect in DIALECTS:
            for route in dialect.routes:
                app.router.add_post(route, self._handler_for(dialect))
            for aux in dialect.aux_routes:
                app.router.add_post(aux, self._aux_handler_for(dialect, aux))
        app.router.add_get("/v1/models", self.handle_models)
        # Tool servers use a state-only capability; the model bearer cannot reach these.
        app.router.add_get("/state", self.handle_state_get)
        app.router.add_put("/state", self.handle_state_put)
        app.router.add_post("/tool", self.handle_tool)
        # A launched tool server fetches its rollout's task here to run `setup_task` — the task
        # is never passed via env, only over this channel, keyed by the state bearer.
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
        """Stash a model-turn-adjacent failure (such as a hook raising) so the rollout
        re-raises it as the real cause, and report it to the harness as an HTTP error."""
        session.error = error
        logger.warning(
            "rollout %s failed: %s: %s", session.trace.id, type(error).__name__, error
        )
        return web.json_response(
            dialect.error_body(str(error)),
            status=getattr(error, "status_code", 502),
        )

    def mediate_capabilities(
        self, session: RolloutSession, dialect: Dialect, body: dict
    ) -> tuple[dict, list[str]]:
        if not session.network_policy.network_restricted:
            return body, []
        mediated, capabilities = mediate(dialect, body, session.network_policy)
        if capabilities:
            logger.warning(
                "interception removed provider content/capabilities blocked by the network "
                "policy or unable to enforce it: id=%s paths=%s",
                session.trace.id,
                ",".join(capabilities),
            )
        return mediated, capabilities

    async def handle_tool(self, request: web.Request) -> web.Response:
        """`POST /tool`: the harness's gate asks whether to run a tool call — its
        `{tool_call_id, name, arguments}` in, the verdict of `RolloutSession.decide_tool`
        out. The model bearer keys it: it never alters what the model already received."""
        session = self.sessions.get(
            request.headers.get("Authorization", "").removeprefix("Bearer ")
        )
        if session is None:
            return web.json_response({"error": "unauthorized"}, status=401)
        session.adopt(asyncio.current_task())
        if session.released:
            return web.json_response({"error": "rollout concluded"}, status=409)
        body = from_json(await request.read())
        try:
            return web.json_response(
                await session.decide_tool(
                    str(body.get("tool_call_id", "")),
                    body.get("name"),
                    body.get("arguments"),
                )
            )
        except RolloutError as error:
            session.error = error
            return web.json_response({"error": str(error)}, status=400)

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
        policy_paths: list[str] | None = None,
        acp: ACPInfo | None = None,
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
                policy=PolicyEvent(
                    code=PROVIDER_CAPABILITY_POLICY_CODE,
                    paths=policy_paths,
                )
                if policy_paths
                else None,
                acp=acp,
            )
        )
        session.trace.notify()

    async def handle_request(
        self, request: web.Request, dialect: Dialect
    ) -> web.StreamResponse:
        session = self.sessions.get(dialect.secret(request.headers))
        if session is None:
            logger.warning("interception: unauthorized request")
            return web.json_response(dialect.error_body("unauthorized"), status=401)
        session.adopt(asyncio.current_task())
        raw = await request.read()
        try:
            body = from_json(raw)
        except ValueError:
            body = json.loads(raw)
        body = dialect.apply_overrides(body, session.ctx.model, session.ctx.sampling)
        streaming = bool(body.get("stream"))
        # A streamed request is served whole once its turn commits: the eval client's
        # provider stream is read to the end, and the train client generates the response.
        relay = streaming and not isinstance(session.ctx.client, TrainClientConfig)
        req_hash = await _request_digest(raw)
        # Keep `read()` for aiohttp's size guard, then release its cache and our local
        # alias after parsing so the wire body does not survive model inference.
        request._read_bytes = None
        del raw
        try:
            acp, upstream_headers = extract_acp_info(request.headers)
        except ValueError as error:
            return web.json_response(dialect.error_body(str(error)), status=400)
        logger.debug(
            "intercept %s: id=%s stream=%s retry=%s",
            request.path,
            session.trace.id,
            streaming,
            request.headers.get(RETRY_COUNT_HEADER, "0"),
        )
        # Graph atomicity under retries: one logical buffered call must commit at most
        # one turn. An explicit key identifies that call directly; otherwise only the SDK's
        # retry marker activates body-digest replay, since an unmarked repeated body can be a
        # legitimate later turn.
        retried = is_retried_request(request.headers)
        idempotent: IdempotentRequest | None = None
        idempotency_key = request.headers.get(IDEMPOTENCY_KEY_HEADER)
        binding = (request.path, req_hash)
        if idempotency_key:
            replay_key = f"explicit:{idempotency_key}"
            # This key identifies the harness-to-interception hop. The server owns its
            # replay semantics, and the body has since been rewritten with rollout model
            # and sampling overrides, so never expose the local key to the provider.
            upstream_headers = {
                name: value
                for name, value in upstream_headers.items()
                if name.lower() != IDEMPOTENCY_KEY_HEADER.lower()
            }
        else:
            replay_key = f"retry:{request.path}:{req_hash.hex()}"

        now = time.monotonic()
        _prune_idempotent_requests(session, now)
        if idempotency_key or retried:
            idempotent = session.idempotent_requests.get(replay_key)
        if idempotent is not None and idempotent.binding != binding:
            return web.json_response(
                dialect.error_body(
                    "Idempotency-Key was reused with a different request"
                ),
                status=400,
            )
        if idempotent is not None and idempotent.response is not None:
            logger.debug(
                "intercept replay: id=%s (idempotent request)", session.trace.id
            )
            idempotent.completed_at = now
            return _replay_response(idempotent.response)

        try:
            model_request, setters = dialect.parse_request(body)
        except ValueError as error:
            return web.json_response(dialect.error_body(str(error)), status=400)
        if session.released:
            return web.json_response(
                dialect.error_body("rollout concluded"), status=409
            )
        if session.stopped:
            return web.json_response(
                dialect.error_body(f"rollout stopped: {session.trace.stop_condition}"),
                status=400,
            )
        if idempotent is None:
            idempotent = IdempotentRequest(binding=binding)
            session.idempotent_requests[replay_key] = idempotent

        async def coalesced(
            inflight: "asyncio.Future[ReplayResponse | None]",
        ) -> web.Response:
            logger.debug(
                "intercept coalesce: id=%s (retry of in-flight turn)", session.trace.id
            )
            response = await asyncio.shield(inflight)
            if response is None:
                return web.json_response(
                    dialect.error_body("upstream attempt failed"), status=503
                )
            return _replay_response(response)

        if idempotent.inflight is not None:
            if streaming:
                return await _buffered_stream(
                    request, dialect, coalesced(idempotent.inflight), session.trace.id
                )
            return await coalesced(idempotent.inflight)
        future: asyncio.Future[ReplayResponse | None] = (
            asyncio.get_running_loop().create_future()
        )
        idempotent.inflight = future
        request[_IDEMPOTENT_ATTEMPT] = _IdempotentAttempt(
            session=session,
            key=replay_key,
            request=idempotent,
            future=future,
        )

        try:
            refused = await session.refused()
            if refused is not None:
                return web.json_response(
                    dialect.error_body(f"rollout stopped: {refused}"), status=400
                )
            original_request = model_request
            model_request, request_rewrites, stopped = await session.rewrite_request(
                model_request
            )
            session.trace.request_rewrites.extend(request_rewrites)
            # A pinned tool result changes the request without a fresh record.
            if stopped is None and model_request != original_request:
                for setter, before, after in zip(
                    setters,
                    original_request.messages,
                    model_request.messages,
                    strict=True,
                ):
                    if after != before:
                        setter(after)
        except RolloutError as error:
            return self._fail(session, dialect, error)
        except Exception as error:  # noqa: BLE001 - surface task hook failures
            return self._fail(
                session,
                dialect,
                TaskError(
                    f"model boundary hook failed: {type(error).__name__}: {error}"
                ),
            )
        except BaseException:
            raise
        if stopped is not None:
            turn = graph.prepare_turn(
                session.trace, model_request.messages, model_request.tools
            )
            turn.commit_prompt()
            session.trace.stop(stopped)
            return web.json_response(
                dialect.error_body(f"rollout stopped: {stopped}"),
                status=400,
            )

        try:
            body, policy_paths = self.mediate_capabilities(session, dialect, body)
            # Restricted mediation can mutate the body without reporting policy paths.
            if request_rewrites or session.network_policy.network_restricted:
                model_request = dialect.parse_request(body)[0]
            turn = graph.prepare_turn(
                session.trace, model_request.messages, model_request.tools
            )
        except ValueError as error:
            return web.json_response(dialect.error_body(str(error)), status=400)
        except RolloutError as error:
            return self._fail(session, dialect, error)
        # The tail is what the harness added since the last turn (tool results, user
        # turns): live watchers see it now rather than with the model's reply.
        session.trace.preview(turn, turn.tail)

        def serve(response: Response, events: bytes | None) -> web.Response:
            if streaming:
                # The committed turn as SSE: the provider's own events when relayed
                # unchanged, else framed from the response (training generates it whole,
                # and a hook may have rewritten a relayed one).
                served = web.Response(
                    body=events
                    if events is not None
                    else b"".join(dialect.stream_events(response.raw or {})),
                    content_type="text/event-stream",
                )
            else:
                served = _completion_response(response.raw)
            idempotent.response = _capture_response(served)
            idempotent.completed_at = time.monotonic()
            return served

        async def sample() -> web.Response:
            session.error = None
            call_response: Response | None = None
            events: bytes | None = None
            node: int | None = None
            error: Exception | None = None
            started = time.time()
            try:
                try:
                    # What actually goes upstream: the native body with the rollout's model +
                    # sampling imposed — recorded raw on the trace, per call.
                    if relay:
                        reply = await session.client.relay(
                            dialect,
                            body,
                            headers=upstream_headers,
                            session_id=session.trace.id,
                        )
                        call_response, events = await _collect_stream(dialect, reply)
                    else:
                        call_response = await session.client.get_response(
                            dialect,
                            body,
                            session.ctx.sampling,
                            headers=upstream_headers,
                            session_id=session.trace.id,
                            turn=turn,
                        )
                    logger.debug(
                        "intercept turn: id=%s tools=%d",
                        session.trace.id,
                        len(call_response.message.tool_calls or []),
                    )
                    if session.released:  # concluded while sampling — seal holds
                        return web.json_response(
                            dialect.error_body("rollout concluded"), status=409
                        )
                    response_rewrites = []
                    stopped = None
                    if session.response_interceptors or session.response_stops:
                        (
                            call_response,
                            response_rewrites,
                            stopped,
                        ) = await session.rewrite_response(call_response)
                        if response_rewrites:
                            events = None
                            assert call_response.raw is not None
                            dialect.rewrite_response(
                                call_response.raw, call_response.message.content or ""
                            )
                            raw_response = call_response.raw
                            call_response = dialect.parse_response(raw_response)
                            call_response.raw = raw_response
                    if session.stopped:
                        return web.json_response(
                            dialect.error_body(
                                f"rollout stopped: {session.trace.stop_condition}"
                            ),
                            status=400,
                        )
                    node = turn.commit(call_response)
                    session.consume_prepared(turn.tail)
                    session.trace.response_rewrites.extend(response_rewrites)
                    if stopped is None:
                        stopped = await session.gate_tool_calls(node)
                    if stopped is not None:
                        session.trace.stop(stopped)
                        return web.json_response(
                            dialect.error_body(f"rollout stopped: {stopped}"),
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
                except Exception as e:  # noqa: BLE001 - surface as an API error
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
                if node is None:
                    turn.abandon()
                # The turn's one per-exchange record: settings, timing, outcome, and
                # the error that ended it (if any).
                self.record_call(
                    session,
                    dialect,
                    body,
                    started,
                    node=node,
                    finish_reason=call_response.finish_reason
                    if call_response
                    else None,
                    usage=call_response.usage if call_response else None,
                    error=error,
                    policy_paths=policy_paths,
                    acp=acp,
                )
            return serve(call_response, events)

        if streaming:
            return await _buffered_stream(request, dialect, sample(), session.trace.id)
        return await sample()

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
            body = await request.json()
            body["model"] = session.ctx.model
            body = self.mediate_capabilities(session, dialect, body)[0]
            result = await session.client.relay_aux(
                dialect, route, body, headers=request.headers
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
        except Exception as e:  # noqa: BLE001 - surface auxiliary relay failures
            logger.warning("aux call failed: id=%s %s", session.trace.id, e)
            return web.json_response(dialect.error_body(str(e)), status=502)
        return web.json_response(result)

    async def handle_models(self, request: web.Request) -> web.Response:
        """`GET /v1/models`: relay the upstream model listing so agent loops can read a
        provider context-window extension (e.g. vLLM's `max_model_len`). The path is shared
        by every dialect; only the auth carrier differs, so the bearer is tried per dialect.
        A pure relay from the session's endpoint config — never recorded on the trace, and a
        failure never fails the rollout."""
        for dialect in DIALECTS:
            session = self.sessions.get(dialect.secret(request.headers))
            if session is not None:
                break
        else:
            return web.json_response({"error": "unauthorized"}, status=401)
        session.adopt(asyncio.current_task())
        logger.debug("intercept models: id=%s", session.trace.id)
        config = session.ctx.client
        headers = dict(config.headers or {})
        headers.update(dialect.auth_headers(resolve_api_key(config)))
        try:
            # Finite read timeout: a hung provider must not stall threshold discovery
            # for the rollout's whole outer timeout - the loop falls back to no compaction.
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=5.0)
            ) as client:
                upstream = await client.get(
                    join_url(config.base_url, "/v1/models"), headers=headers
                )
        except httpx.HTTPError as e:
            logger.warning("models call failed: id=%s %s", session.trace.id, e)
            return web.json_response(dialect.error_body(str(e)), status=502)
        return web.Response(
            body=upstream.content,
            status=upstream.status_code,
            content_type="application/json",
        )

    def _session_for(
        self, request: web.Request, *, allow_service: bool = False
    ) -> RolloutSession | None:
        """Resolve a private state bearer, or a trusted shared server plus route id."""
        auth = request.headers.get("Authorization", "")
        secret = auth[len("Bearer ") :] if auth.startswith("Bearer ") else ""
        session = self.state_sessions.get(secret)
        if session is None and allow_service and secret in self.state_service_secrets:
            session = self.state_routes.get(
                request.headers.get("X-Verifiers-State-Route", "")
            )
        if session is not None:  # state writes must not land on a sealed trace either
            session.adopt(asyncio.current_task())
        return session

    async def handle_state_get(self, request: web.Request) -> web.Response:
        """Hand a rollout's tool server the current shared `trace.state` (it pulls before each
        `@vf.tool` call, so it sees writes from the other servers)."""
        session = self._session_for(request, allow_service=True)
        if session is None:
            return web.json_response({"error": "unauthorized"}, status=401)
        logger.debug("intercept GET /state: id=%s", session.trace.id)
        state = session.trace.state
        return web.Response(
            # TypeAdapter emits UTF-8 bytes directly, avoiding a JSON str copy in aiohttp.
            body=session.state_adapter.dump_json(state),
            content_type="application/json",
            charset="utf-8",
        )

    async def handle_task_get(self, request: web.Request) -> web.Response:
        """Hand a launched tool server the rollout's task (class ref + JSON) so it can run
        `setup_task` for this rollout — keyed by its private state bearer."""
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
        session = self._session_for(request, allow_service=True)
        if session is None:
            return web.json_response({"error": "unauthorized"}, status=401)
        logger.debug("intercept PUT /state: id=%s", session.trace.id)
        state_cls = type(session.trace.state)
        raw = await request.read()
        try:
            new_state = session.state_adapter.validate_json(raw)
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
