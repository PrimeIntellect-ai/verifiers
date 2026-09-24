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
from collections.abc import AsyncIterator, Collection, Mapping
from contextlib import asynccontextmanager
from functools import partial
from typing import Literal

import httpx
from aiohttp import web
from pydantic import ValidationError
from pydantic_core import PydanticSerializationError, from_json, to_json

from verifiers.v1 import graph
from verifiers.v1.clients import Client, resolve_client
from verifiers.v1.clients.base import join_url
from verifiers.v1.configs.client import (
    BaseClientConfig,
    resolve_api_key,
)
from verifiers.v1.dialects import DIALECTS, Dialect
from verifiers.v1.dialects.policy import PROVIDER_CAPABILITY_POLICY_CODE, mediate
from verifiers.v1.errors import (
    InterceptionError,
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
from verifiers.v1.types import FinishReason, Request, Response, Usage

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


def _json_reply(
    completion: dict | None, *, status: int = 200, retryable: bool = False
) -> ReplayResponse:
    """Encode a buffered JSON result once, before HTTP delivery or replay."""
    try:
        body = to_json(completion, inf_nan_mode="constants")
    except PydanticSerializationError:
        body = json.dumps(completion).encode("utf-8")
    return ReplayResponse(status, body, "application/json; charset=utf-8", retryable)


def _replay_response(response: ReplayResponse) -> web.Response:
    return web.Response(
        body=response.body,
        status=response.status,
        headers={"Content-Type": response.content_type},
    )


def _prune_idempotent_requests(session: RolloutSession, now: float) -> None:
    """Expire and cap completed replays; active attempts are never evicted."""
    completed = sorted(
        [
            (request.completed_at, key)
            for key, request in session.idempotent_requests.items()
            if request.completed_at is not None
        ],
        key=lambda item: item[0],
    )
    excess = len(completed) - IDEMPOTENCY_CACHE_MAX_COMPLETED
    for position, (completed_at, key) in enumerate(completed):
        if position < excess or now - completed_at >= IDEMPOTENCY_CACHE_TTL_SECONDS:
            session.idempotent_requests.pop(key)


async def _buffered_stream(
    request: web.Request,
    dialect: Dialect,
    pending: asyncio.Future[ReplayResponse],
    trace_id: str,
) -> web.StreamResponse:
    """Serve a turn to an SSE client once it is committed, keeping the connection alive
    while it is produced. A result within the grace period is served as is; after it the
    stream is committed and sent the dialect's keepalives, so neither a proxy's timeout
    nor a client's own idle timeout cuts a long turn. Once committed, a failure is framed
    as the dialect's SSE error. A reader that goes away leaves the turn running for its
    retries to coalesce onto."""
    task = asyncio.shield(pending)
    started = time.monotonic()
    try:
        done, _ = await asyncio.wait({task}, timeout=KEEPALIVE_GRACE_SECONDS)
        if done:
            return _replay_response(task.result())
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
        replay = await task
        if connected:
            try:
                body = replay.body
                if replay.status >= 400:
                    error = from_json(body)
                    error["error"].update(
                        status_code=replay.status, retryable=replay.retryable
                    )
                    body = dialect.stream_error(error)
                await stream.write(body)
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
                "intercept stream: reader cancelled: id=%s after=%.1fs",
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

    async def start(self) -> None:
        app = web.Application(client_max_size=MAX_REQUEST_BODY)
        for dialect in DIALECTS:
            for route in dialect.routes:
                app.router.add_post(
                    route, partial(self.handle_request, dialect=dialect)
                )
            for aux in dialect.aux_routes:
                app.router.add_post(
                    aux, partial(self.handle_aux, dialect=dialect, route=aux)
                )
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
        self.port = self.runner.addresses[0][1]
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
        self,
        session: RolloutSession,
        dialect: Dialect,
        error: Exception,
        *,
        committed: bool = False,
    ) -> ReplayResponse:
        """Stash a model or hook failure so the rollout re-raises the real cause,
        and report it to the harness as an HTTP error."""
        session.error = (
            error if isinstance(error, RolloutError) else InterceptionError(str(error))
        )
        logger.warning(
            "rollout %s failed: %s: %s", session.trace.id, type(error).__name__, error
        )
        status = error.status_code if isinstance(error, ProviderError) else 400
        return _json_reply(
            dialect.error_body(str(error)),
            status=status,
            retryable=not committed and (status in (408, 409, 429) or status >= 500),
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
        if idempotent is not None and idempotent.task.done():
            logger.debug(
                "intercept replay: id=%s (idempotent request)", session.trace.id
            )
            idempotent.completed_at = now
            return _replay_response(idempotent.task.result())

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

        async def sample(body: dict, model_request: Request) -> ReplayResponse:
            try:
                refused = await session.refused()
                if refused is not None:
                    return _json_reply(
                        dialect.error_body(f"rollout stopped: {refused}"), status=400
                    )
                original_request = model_request
                (
                    model_request,
                    request_rewrites,
                    stopped,
                ) = await session.rewrite_request(model_request)
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
                            assert setter is not None
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
            if stopped is not None:
                turn = graph.prepare_turn(
                    session.trace, model_request.messages, model_request.tools
                )
                turn.commit_prompt()
                session.trace.stop(stopped)
                return _json_reply(
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
                return _json_reply(dialect.error_body(str(error)), status=400)
            except RolloutError as error:
                return self._fail(session, dialect, error)
            # The tail is what the harness added since the last turn (tool results, user
            # turns): live watchers see it now rather than with the model's reply.
            session.trace.preview(turn, turn.tail)

            session.error = None
            call_response: Response | None = None
            events: bytes | None = None
            node: int | None = None
            error: BaseException | None = None
            started = time.time()
            try:
                # What actually goes upstream: the native body with the rollout's model +
                # sampling imposed — recorded raw on the trace, per call.
                call_response, events = await session.client._complete(
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
                    return _json_reply(
                        dialect.error_body("rollout concluded"),
                        status=409,
                        retryable=True,
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
                    return _json_reply(
                        dialect.error_body(
                            f"rollout stopped: {session.trace.stop_condition}"
                        ),
                        status=400,
                    )
                # Encode before committing; an unusable reply must not publish a turn.
                served = (
                    ReplayResponse(
                        status=200,
                        body=events
                        if events is not None
                        else b"".join(dialect.stream_events(call_response.raw or {})),
                        content_type="text/event-stream",
                    )
                    if streaming
                    else _json_reply(call_response.raw)
                )
                node = turn.commit(call_response)
                session.consume_prepared(turn.tail)
                session.trace.response_rewrites.extend(response_rewrites)
                if stopped is None:
                    stopped = await session.gate_tool_calls(node)
                if stopped is not None:
                    session.trace.stop(stopped)
                    return _json_reply(
                        dialect.error_body(f"rollout stopped: {stopped}"),
                        status=400,
                    )
                return served
            except Exception as e:  # noqa: BLE001 - surface as an API error
                error = e
                return self._fail(session, dialect, e, committed=node is not None)
            except BaseException as e:
                # A cancelled exchange (session teardown, shutdown) is still
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

        async def serve() -> ReplayResponse:
            try:
                reply = await sample(body, model_request)
                if not reply.retryable:
                    idempotent.completed_at = time.monotonic()
                return reply
            finally:
                if idempotent.completed_at is not None:
                    _prune_idempotent_requests(session, time.monotonic())
                elif session.idempotent_requests.get(replay_key) is idempotent:
                    session.idempotent_requests.pop(replay_key)

        if idempotent is None:
            idempotent = IdempotentRequest(binding=binding)
            session.idempotent_requests[replay_key] = idempotent
            # Keep the task for both in-flight coalescing and completed-result replay.
            idempotent.task = asyncio.create_task(serve())
            session.adopt(idempotent.task)
        else:
            logger.debug(
                "intercept coalesce: id=%s (retry of in-flight turn)", session.trace.id
            )
        if streaming:
            return await _buffered_stream(
                request, dialect, idempotent.task, session.trace.id
            )
        return _replay_response(await asyncio.shield(idempotent.task))

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
