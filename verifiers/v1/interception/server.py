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
from tempfile import SpooledTemporaryFile
from typing import Literal

from aiohttp import web
from pydantic import BaseModel, ValidationError
from pydantic_core import PydanticSerializationError, from_json, to_json

from verifiers.v1 import graph
from verifiers.v1.clients import Client, resolve_client
from verifiers.v1.configs.client import BaseClientConfig
from verifiers.v1.dialects import DIALECTS, Dialect
from verifiers.v1.dialects.base import (
    PROVIDER_CAPABILITY_POLICY_CODE,
    is_sse_done_event,
)
from verifiers.v1.errors import (
    HarnessError,
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
from verifiers.v1.session import RolloutSession
from verifiers.v1.trace import Error, ModelCall, PolicyEvent, TimeSpan
from verifiers.v1.types import FinishReason, Request, Response, ToolMessage, Usage

logger = logging.getLogger(__name__)


# Each session proxies one rollout's own harness requests, so aiohttp's default 1 MiB body
# cap is an artificial bottleneck — a large tool result (e.g. a `cat` of a big file) trips it
# and the harness gets a 413. Allow large bodies; the upstream provider and the model's
# context window are the real limits, this is just a host-OOM backstop.
MAX_REQUEST_BODY = 1024**3  # 1 GiB (aiohttp's default is 1 MiB)
KEEPALIVE_INTERVAL_SECONDS = 3
STREAM_QUEUE_MAXSIZE = 16
STREAM_MEMORY_BUFFER = 4 * 1024**2
# blake2b saturates ~1.7 GB/s, so a body up to this size hashes inline in well under a
# millisecond; a larger one (bodies may reach `MAX_REQUEST_BODY`) is hashed off the event
# loop instead — see `_request_digest`.
HASH_INLINE_MAX = 1024**2  # 1 MiB
# Attempt counter the stainless-generated SDKs (OpenAI, Anthropic) send on every request:
# 0 on the first attempt, incremented on each retry of the same request.
RETRY_COUNT_HEADER = "x-stainless-retry-count"


class ToolHookRequest(BaseModel):
    phase: Literal["before", "after"]
    message: ToolMessage
    content: Literal["any", "none", "nonempty_text"] = "any"
    resultPrefix: str = ""
    resultSuffix: str = ""


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
        state_service_secrets: Collection[str] = (),
    ) -> None:
        super().__init__()
        self.sessions: dict[str, RolloutSession] = {}
        self.tool_sessions: dict[str, RolloutSession] = {}
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

    def register(self, session: RolloutSession) -> tuple[str, str, str]:
        """Register separate model, task-state, and tool-interception capabilities."""
        session.client = self._client(session.ctx.client)
        model_secret = secrets.token_urlsafe(16)
        state_secret = secrets.token_urlsafe(16)
        tool_secret = secrets.token_urlsafe(16)
        self.sessions[model_secret] = session
        self.state_sessions[state_secret] = session
        self.tool_sessions[tool_secret] = session
        self.state_routes[session.trace.id] = session
        return model_secret, state_secret, tool_secret

    def unregister(
        self, model_secret: str, state_secret: str, tool_secret: str
    ) -> None:
        session = self.sessions.pop(model_secret, None)
        self.state_sessions.pop(state_secret, None)
        self.tool_sessions.pop(tool_secret, None)
        if session is not None:
            self.state_routes.pop(session.trace.id, None)
            # The rollout concluded; its trace is sealed. Cancel straggler handlers
            # (aiohttp keeps them alive past client death) so a slow upstream call
            # can't commit a late turn onto the concluded trace.
            session.release()

    @asynccontextmanager
    async def acquire(self, session: RolloutSession) -> AsyncIterator[Slot]:
        model_secret, state_secret, tool_secret = self.register(session)
        try:
            yield self.base_url, model_secret, state_secret, tool_secret
        finally:
            self.unregister(model_secret, state_secret, tool_secret)

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
        app = web.Application(client_max_size=MAX_REQUEST_BODY)
        for dialect in DIALECTS:
            for route in dialect.routes:
                app.router.add_post(route, self._handler_for(dialect))
            for aux in dialect.aux_routes:
                app.router.add_post(aux, self._aux_handler_for(dialect, aux))
        # Auxiliary services use capabilities separate from the model bearer.
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
        mediated, capabilities = dialect.mediate_external_capabilities(
            body, session.network_policy
        )
        capabilities = list(dict.fromkeys(capabilities))
        if capabilities:
            logger.warning(
                "interception removed provider capabilities: id=%s paths=%s",
                session.trace.id,
                ",".join(capabilities),
            )
        return mediated, capabilities

    async def handle_tool(self, request: web.Request) -> web.Response:
        session = self.tool_sessions.get(
            request.headers.get("Authorization", "").removeprefix("Bearer ")
        )
        if session is None:
            return web.json_response({"error": "unauthorized"}, status=401)
        session.adopt(asyncio.current_task())
        try:
            async with session.tool_interception_lock:
                if session.released:
                    return web.json_response({"error": "rollout concluded"}, status=409)
                if session.stopped:
                    return web.json_response(
                        {"action": "stop", "reason": session.trace.stop_condition}
                    )
                if session.fatal_error is not None:
                    return web.json_response(
                        {"error": str(session.fatal_error)}, status=409
                    )
                hook = ToolHookRequest.model_validate_json(await request.read())
                decision = await session.handle_tool(
                    hook.phase,
                    hook.message,
                    hook.content,
                    hook.resultPrefix,
                    hook.resultSuffix,
                )
            return web.json_response(decision)
        except Exception as error:  # noqa: BLE001 - malformed native-hook traffic is fatal
            failure = (
                error
                if isinstance(error, RolloutError)
                else HarnessError(
                    f"tool interception failed: {type(error).__name__}: {error}"
                )
            )
            session.fatal_error = failure
            return web.json_response({"error": str(failure)}, status=400)

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
        body = dialect.apply_overrides(body, session.ctx.model, session.ctx.sampling)
        streaming = dialect.streaming(body)
        logger.debug(
            "intercept %s: id=%s stream=%s",
            request.path,
            session.trace.id,
            streaming,
        )
        if session.released:
            return web.json_response(
                dialect.error_body("rollout concluded"), status=409
            )
        if session.stopped:
            return web.json_response(
                dialect.error_body(f"rollout stopped: {session.trace.stop_condition}"),
                status=400,
            )
        if session.fatal_error is not None:
            return web.json_response(
                dialect.error_body(str(session.fatal_error)), status=400
            )
        # Graph atomicity under retries: sampling a marked SDK retry again would commit a
        # second turn and fork the graph, so serve it the recorded response (or the
        # still-computing attempt's result). Only marked retries match — a repeated body
        # alone is no proof of a retry (compaction can legitimately regenerate an identical
        # request), and a stale replay would loop the rollout.
        retried = is_retried_request(request.headers)
        if (
            retried
            and session.last_request == req_hash
            and session.last_response is not None
        ):
            logger.debug("intercept replay: id=%s (retried request)", session.trace.id)
            return _completion_response(session.last_response)
        if session.last_request == req_hash:
            # A fresh attempt supersedes the recorded response for the same body: drop it
            # so this attempt's own retries coalesce or re-run instead of replaying the
            # previous turn.
            session.last_request = None
            session.last_response = None

        try:
            model_request = dialect.parse_request(body)
        except ValueError as error:
            return web.json_response(dialect.error_body(str(error)), status=400)

        async def coalesced(inflight: "asyncio.Future[dict | None]") -> web.Response:
            logger.debug(
                "intercept coalesce: id=%s (retry of in-flight turn)", session.trace.id
            )
            completion = await inflight
            if completion is None:
                return web.json_response(
                    dialect.error_body("upstream attempt failed"), status=503
                )
            return _completion_response(completion)

        fut: asyncio.Future[dict | None] | None = None
        if not streaming:
            if retried and (inflight := session.inflight.get(req_hash)) is not None:
                return await coalesced(inflight)
            fut = asyncio.get_running_loop().create_future()
            session.inflight[req_hash] = fut

        def finish_inflight(completion: dict | None = None) -> None:
            if fut is None:
                return
            if session.inflight.get(req_hash) is fut:
                session.inflight.pop(req_hash, None)
            if not fut.done():
                fut.set_result(completion)

        try:
            refused = await session.refused()
            if refused is not None:
                finish_inflight()
                return web.json_response(
                    dialect.error_body(f"rollout stopped: {refused}"), status=400
                )
            original_request = model_request
            model_request, request_rewrites, stopped = await session.rewrite_request(
                model_request
            )
            if request_rewrites:
                session.trace.request_rewrites.extend(request_rewrites)
            if stopped is None and model_request != original_request:
                dialect.rewrite_request(body, original_request, model_request)
        except RolloutError as error:
            finish_inflight()
            return self._fail(session, dialect, error)
        except Exception as error:  # noqa: BLE001 - surface task hook failures
            finish_inflight()
            return self._fail(
                session,
                dialect,
                TaskError(
                    f"model boundary hook failed: {type(error).__name__}: {error}"
                ),
            )
        except BaseException:
            finish_inflight()
            raise
        if stopped is not None:
            turn = graph.prepare_turn(session.trace, model_request.messages)
            turn.commit_prompt(model_request.tools)
            session.trace.stop(stopped)
            finish_inflight()
            return web.json_response(
                dialect.error_body(f"rollout stopped: {stopped}"),
                status=400,
            )

        try:
            body, policy_paths = self.mediate_capabilities(session, dialect, body)
            model_request = dialect.parse_request(body)
            turn = graph.prepare_turn(session.trace, model_request.messages)
        except ValueError as error:
            finish_inflight()
            return web.json_response(dialect.error_body(str(error)), status=400)
        except RolloutError as error:
            finish_inflight()
            return self._fail(session, dialect, error)

        # A native hook may run as soon as its SDK sees a complete tool-use block,
        # before the terminal stream event. Buffering keeps the hook behind commit().
        inspect_response = bool(
            session.response_interceptors
            or session.response_stops
            or session.native_tool_interception
        )
        if streaming:
            return await self._stream(
                request,
                session,
                dialect,
                body,
                model_request,
                turn=turn,
                inspect_response=inspect_response,
                policy_paths=policy_paths,
            )

        def serve(response: Response) -> web.Response:
            # Record the served turn and hand it to any coalesced retry, so a retried
            # byte-identical request replays instead of re-sampling and forking the graph.
            # `Response.raw` is the full native provider object (or the renderer's synthesized
            # completion) that the server serializes back to the program.
            session.last_request = req_hash
            session.last_response = response.raw
            finish_inflight(response.raw)
            return _completion_response(response.raw)

        try:
            session.error = None
            call_response: Response | None = None
            node: int | None = None
            error: Exception | None = None
            started = time.time()
            try:
                try:
                    # What actually goes upstream: the native body with the rollout's model +
                    # sampling imposed — recorded raw on the trace, per call.
                    call_response = await session.client.get_response(
                        dialect,
                        body,
                        session.ctx.sampling,
                        headers=request.headers,
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
                            assert call_response.raw is not None
                            dialect.rewrite_response(
                                call_response.raw, call_response.message.content or ""
                            )
                            raw_response = call_response.raw
                            call_response = dialect.parse_response(
                                dialect.validate_response(raw_response)
                            )
                            call_response.raw = raw_response
                    if session.stopped:
                        return web.json_response(
                            dialect.error_body(
                                f"rollout stopped: {session.trace.stop_condition}"
                            ),
                            status=400,
                        )
                    node = turn.commit(call_response, model_request.tools)
                    session.consume_prepared(turn)
                    session.trace.response_rewrites.extend(response_rewrites)
                    if stopped is not None:
                        session.trace.stop(stopped)
                        return web.json_response(
                            dialect.error_body(f"rollout stopped: {stopped}"),
                            status=400,
                        )
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
                )
            return serve(call_response)
        finally:
            # Free the in-flight slot and unblock any coalesced retry; None signals "no servable
            # response" (an error/refuse return above), so the waiter surfaces a retryable error.
            finish_inflight()

    async def _stream(
        self,
        request: web.Request,
        session: RolloutSession,
        dialect: Dialect,
        body: dict,
        model_request: Request,
        *,
        turn: graph.PendingTurn,
        inspect_response: bool,
        policy_paths: list[str] | None = None,
    ) -> web.StreamResponse:
        """A streamed (SSE) model turn: relay the provider's stream through to the program,
        incrementally assembling the response to record on the trace (the only client that
        streams is the eval relay)."""
        session.error = None
        reply = None
        response: Response | None = None
        node: int | None = None
        error: Exception | None = None
        started = time.time()
        try:
            try:
                reply = await session.client.relay(
                    dialect,
                    body,
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
            except Exception as e:  # noqa: BLE001 - surface as an API error
                error = e
                logger.warning("model call failed: id=%s %s", session.trace.id, e)
                return web.json_response(dialect.error_body(str(e)), status=502)

            if inspect_response:
                buffered = SpooledTemporaryFile(  # noqa: SIM115 - closed before every exit
                    max_size=STREAM_MEMORY_BUFFER
                )
                parser = dialect.stream_parser()
                saw_terminal = False
                try:
                    async for chunk in reply.chunks:
                        buffered.write(chunk)
                        saw_terminal |= dialect.is_terminal_event(chunk)
                        if parser.on_done is not None and is_sse_done_event(chunk):
                            parser.on_done()
                        parser.feed(chunk)
                    if not saw_terminal:
                        raise ProviderError(
                            "upstream stream ended before its terminal event"
                        )
                    response = parser.finish()
                    response_rewrites = []
                    stopped = None
                    if session.response_interceptors or session.response_stops:
                        (
                            response,
                            response_rewrites,
                            stopped,
                        ) = await session.rewrite_response(response)
                        if response_rewrites:
                            assert response.raw is not None
                            dialect.rewrite_response(
                                response.raw, response.message.content or ""
                            )
                            raw_response = response.raw
                            response = dialect.parse_response(
                                dialect.validate_response(raw_response)
                            )
                            response.raw = raw_response
                except RolloutError as e:
                    buffered.close()
                    error = e
                    session.error = e
                    return self._fail(session, dialect, e)
                except Exception as e:  # noqa: BLE001 - malformed provider stream
                    buffered.close()
                    error = ProviderError(str(e))
                    session.error = error
                    return self._fail(session, dialect, error)
                finally:
                    await reply.close()

                if session.released or session.stopped:
                    buffered.close()
                    return web.json_response(
                        dialect.error_body(
                            "rollout concluded"
                            if session.released
                            else f"rollout stopped: {session.trace.stop_condition}"
                        ),
                        status=409 if session.released else 400,
                    )
                node = turn.commit(response, model_request.tools)
                session.consume_prepared(turn)
                session.trace.response_rewrites.extend(response_rewrites)
                if stopped is not None:
                    buffered.close()
                    session.trace.stop(stopped)
                    return web.json_response(
                        dialect.error_body(f"rollout stopped: {stopped}"),
                        status=400,
                    )

                resp = web.StreamResponse(
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
                )
                resp.content_type = reply.content_type.split(";")[0].strip()
                try:
                    await resp.prepare(request)
                    if response_rewrites:
                        for event in dialect.stream_events(response.raw or {}):
                            await resp.write(event)
                    else:
                        buffered.seek(0)
                        while chunk := buffered.read(64 * 1024):
                            await resp.write(chunk)
                    await resp.write_eof()
                except ConnectionResetError:
                    pass
                finally:
                    buffered.close()
                return resp

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
                maxsize=STREAM_QUEUE_MAXSIZE
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
                        async with asyncio.timeout(KEEPALIVE_INTERVAL_SECONDS):
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
                            except Exception as e:  # noqa: BLE001 - defer parser failure
                                parser_error = e
                        # forwarded after the turn is committed, below
                        deferred.append(chunk)
                        continue
                    await resp.write(chunk)
                    if parser_error is None:
                        try:
                            feed_event(chunk)
                        except Exception as e:  # noqa: BLE001 - defer parser failure
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
                await reply.close()

            try:
                if parser_error is not None:
                    raise parser_error
                response = parser.finish()
                if not session.released and not session.stopped:
                    node = turn.commit(response, model_request.tools)
                    session.consume_prepared(turn)
                    logger.debug("intercept stream turn: id=%s", session.trace.id)
                elif session.stopped:
                    with contextlib.suppress(ConnectionResetError):
                        await resp.write_eof()
                    return resp
            finally:
                # Release the withheld events only now — after the commit — then close.
                with contextlib.suppress(ConnectionResetError):
                    for event in deferred:
                        await resp.write(event)
                    await resp.write_eof()
            return resp
        except OverlongPromptError as e:
            # A streamed terminal provider failure is discovered only after its response body
            # was relayed. Context exhaustion remains a clean truncation like earlier failures.
            error = e
            session.trace.stop("context_length")
            logger.debug("prompt too long: id=%s", session.trace.id)
            return resp
        except RolloutError as e:
            # A streamed terminal provider failure is discovered only after the
            # response body has been relayed. Keep it off the graph and preserve
            # the typed cause for the rollout if the native SDK does not retry it.
            if node is None:
                error = e
                session.error = e
            raise
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
                body,
                started,
                node=node,
                finish_reason=response.finish_reason if response is not None else None,
                usage=response.usage if response is not None else None,
                error=error,
                policy_paths=policy_paths,
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
