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
import math
import os
import re
import secrets
import time
import traceback
from collections.abc import AsyncIterator, Awaitable, Collection, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
from aiohttp import web
from pydantic import ValidationError
from pydantic_core import PydanticSerializationError, from_json, to_json

from verifiers.v1 import graph, steering
from verifiers.v1.clients import Client, resolve_client
from verifiers.v1.clients.base import join_url
from verifiers.v1.clients.client import EncodedBody, RelayReply
from verifiers.v1.clients.gate import REJECTION_STATUSES
from verifiers.v1.configs.client import (
    BaseClientConfig,
    ProviderRetryConfig,
    TrainClientConfig,
    resolve_api_key,
    resolve_headers,
)
from verifiers.v1.configs.interception import InterceptionServerConfig
from verifiers.v1.dialects import DIALECTS, Dialect
from verifiers.v1.dialects.base import (
    PROVIDER_CAPABILITY_POLICY_CODE,
    is_sse_done_event,
)
from verifiers.v1.errors import (
    ProviderError,
    RolloutError,
    TaskError,
    model_error,
)
from verifiers.v1.interception.base import Interception, Slot
from verifiers.v1.interception.tunnel import Tunnel, make_tunnel
from verifiers.v1.semantic import ACPInfo, extract_acp_info
from verifiers.v1.session import (
    IdempotentRequest,
    ReplayResponse,
    RolloutSession,
)
from verifiers.v1.trace import Error, ModelCall, PolicyEvent, TimeSpan
from verifiers.v1.types import FinishReason, Response, Usage
from verifiers.v1.utils.retries import provider_retry_wait

logger = logging.getLogger(__name__)


# Each session proxies one rollout's own harness requests, so aiohttp's default 1 MiB body
# cap is an artificial bottleneck — a large tool result (e.g. a `cat` of a big file) trips it
# and the harness gets a 413. Allow large bodies; the upstream provider and the model's
# context window are the real limits, this is just a host-OOM backstop.
MAX_REQUEST_BODY = 1024**3  # 1 GiB (aiohttp's default is 1 MiB)
KEEPALIVE_INTERVAL_SECONDS = 3
# A streamed turn commits its SSE stream only after this long: a result within it keeps
# its HTTP status, and a longer one is kept alive well inside the tunnel's response-header
# timeout.
KEEPALIVE_GRACE_SECONDS = 60
# A model call the provider failed transiently is sent again in place (`sample`), first
# after about this long, doubling per failure up to `ProviderRetryConfig.max_backoff`,
# until `ProviderRetryConfig.budget` runs out.
RETRY_BACKOFF_SECONDS = 2.0
# blake2b saturates ~1.7 GB/s, so a body up to this size hashes inline in well under a
# millisecond; a larger one (bodies may reach `MAX_REQUEST_BODY`) is hashed off the event
# loop instead — see `_request_digest`.
HASH_INLINE_MAX = 1024**2  # 1 MiB
# Attempt counter the stainless-generated SDKs (OpenAI, Anthropic) send on every request:
# 0 on the first attempt, incremented on each retry of the same request.
RETRY_COUNT_HEADER = "x-stainless-retry-count"
# Response header the stainless-generated SDKs obey over their own status rules.
SHOULD_RETRY_HEADER = "x-should-retry"


MAX_STREAM_CUT_OFFS = 5
"""Streams a provider cuts mid-turn that each restart the call's retry budget; past it, the
budget runs on as for any failure (a provider that cuts every stream still gives up)."""
REROUTE_AFTER_SECONDS = 120.0
REROUTE_STALLS = 2
"""A provider that routes by session (`SESSION_ID_HEADER`) keeps sending a rollout to the
replica that served it; when that replica wedges, every exchange of the rollout waits out
the provider's first-token timeout with nothing streamed, for hours, while fresh sessions
are served in seconds (GLM 5.3, 2026-10-29). After `REROUTE_STALLS` such exchanges in a row,
each at least `REROUTE_AFTER_SECONDS` long, the rollout continues under a new routing id
(`_reroute`): its prefix cache is lost, the rollout is not. A stream the provider cuts after
at least `REROUTE_AFTER_SECONDS` reroutes at once: a slow replica streams for an hour or more
before the cut, and the regeneration would go back to it."""


SLOW_STREAM_AFTER_SECONDS = 600.0
SLOW_STREAM_EVENTS_PER_SECOND = 6.0
"""A stream that has run `SLOW_STREAM_AFTER_SECONDS` at under `SLOW_STREAM_EVENTS_PER_SECOND`
(about a token each) is cut here as the provider would cut it: a slow GLM 5.3 replica streams
~4 tok/s (the others ~19) for up to 100 minutes before the provider cuts it, and its rollout
waits all that time (2026-10-30). The cut is a long one, so the rollout reroutes at once
(`REROUTE_AFTER_SECONDS`) and the turn is sampled again on another replica."""


def _retry_wait(
    error: ProviderError,
    failures: int,
    failing_for: float,
    retry: ProviderRetryConfig,
    gated: bool,
) -> float | None:
    """How long to wait before sending a failed model call again, or None when the
    failure is final (`provider_retry_wait`), or a rejection (429/529) the request gate
    already retried (`RequestGateConfig.max_rejections`)."""
    if gated and error.status_code in REJECTION_STATUSES:
        return None
    return provider_retry_wait(
        error, failures, failing_for, retry, base=RETRY_BACKOFF_SECONDS
    )


_WINDOW_OVERFLOWS = (
    # Prime inference: "This model configuration accepts at most 1048576 combined input and
    # output tokens. However, your request has 919998 input tokens and asks for 131072 ..."
    re.compile(
        r"at most (?P<window>\d+) combined input and output tokens.*?has (?P<prompt>\d+) input tokens",
        re.DOTALL,
    ),
    # OpenRouter: "This endpoint's maximum context length is N tokens. However, you
    # requested about M tokens (X of text input, Y in the output)."
    re.compile(
        r"maximum context length is (?P<window>\d+) tokens.*?\((?P<prompt>\d+) of text input",
        re.DOTALL,
    ),
    # Providers behind OpenRouter, quoted in its `metadata.raw` (2026-10-26, ProgramBench):
    # "input (919960 tokens) plus requested output (131072 tokens) exceeds the safe limit of
    # 1043334 tokens for model context window of 1048576 tokens"
    re.compile(
        r"input \((?P<prompt>\d+) tokens\) plus requested output \(\d+ tokens\) exceeds the safe limit of (?P<window>\d+) tokens",
        re.DOTALL,
    ),
    # vLLM: "maximum context length of 1048576 tokens. You requested a total of 1070000
    # tokens: 938928 tokens from the input messages and 131072 tokens for the completion"
    re.compile(
        r"maximum context length of (?P<window>\d+) tokens\. You requested a total of \d+ tokens: (?P<prompt>\d+) tokens from the input",
        re.DOTALL,
    ),
)
MIN_FITTED_OUTPUT = 4096
"""Least output worth resending for; below it the prompt itself is the overflow."""


def _fit_output_to_window(
    error: ProviderError, held: dict | EncodedBody
) -> dict | None:
    """The request body with its output cap lowered to what the provider says fits, when
    it refused a prompt that fits its window only because of the output asked for (2026-10-26:
    OpenCode compacts after a turn ends above its threshold, so one large tool result can
    carry the next prompt to 920K, and 920K + 131,072 output tokens tops a 1M window)."""
    if error.status_code != 400:
        return None
    for pattern in _WINDOW_OVERFLOWS:
        if match := pattern.search(str(error)):
            window, prompt = int(match.group("window")), int(match.group("prompt"))
            break
    else:
        return None
    body = from_json(held.data) if isinstance(held, EncodedBody) else dict(held)
    key = next(
        (k for k in ("max_completion_tokens", "max_tokens") if body.get(k)), None
    )
    fits = window - prompt - 256
    if key is None or fits < MIN_FITTED_OUTPUT or fits >= body[key]:
        return None
    return {**body, key: fits}


IDEMPOTENCY_KEY_HEADER = "Idempotency-Key"
REJECTED_REQUEST_DIR_ENV = "VF_REJECTED_REQUEST_DIR"
"""A directory to write each request body a provider rejected with a non-retryable 4xx
(an opaque `invalid_request` names no cause; the body is the only evidence). Unset: off."""
REJECTED_REQUEST_DUMPS = 200
"""At most this many dumps per process, so a provider rejecting everything cannot fill the disk."""
_rejected_dumps = 0


def _dump_rejected(
    body: dict | EncodedBody | None, error: BaseException | None, trace_id: str
) -> None:
    """Write the full body sent upstream (as the client holds it) when the provider
    rejected it with a non-retryable 4xx and `VF_REJECTED_REQUEST_DIR` is set."""
    global _rejected_dumps
    directory = os.environ.get(REJECTED_REQUEST_DIR_ENV)
    status = getattr(error, "status_code", None)
    if not directory or body is None or not isinstance(error, ProviderError):
        return
    if status is None or not 400 <= status < 500 or status in (408, 429):
        return
    if _rejected_dumps >= REJECTED_REQUEST_DUMPS:
        return
    _rejected_dumps += 1
    try:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(
            directory, f"{int(time.time())}-{trace_id}-{secrets.token_hex(3)}.json"
        )
        request = from_json(body.data) if isinstance(body, EncodedBody) else body
        with open(path, "w") as out:
            json.dump(
                {
                    "trace": trace_id,
                    "status": status,
                    "error": str(error),
                    "request": request,
                },
                out,
                default=str,
            )
    except OSError:
        logger.warning("could not dump a rejected request", exc_info=True)


IDEMPOTENCY_CACHE_TTL_SECONDS = 600
IDEMPOTENCY_CACHE_MAX_COMPLETED = 4
"""Completed replays a session keeps. A harness retries one of its latest requests, and
each replay holds the whole served response (a relayed stream's events, about 250
bytes per generated token), so a long-running seat must not keep dozens of them."""


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


LEAKED_TOOL_CALL = ("<tool_call>", "</tool_call>", "<arg_key>", "<arg_value>")
"""Raw tool-call markup a provider's parser left in a reply's reasoning: GLM's tool call
whole, or only its arguments and closing tag when the opening one was consumed (GLM 5.3
on Prime, 2026-10-25: a PostTrainBench turn whose reasoning ends in
`</arg_value></tool_call>`, the agent writing out Qwen3's own `<tool_call>` template)."""


DROPPED_OUTPUT_TOKENS = 3
"""Completion tokens past the reasoning that an empty reply may account for (end-of-turn
markers); more means the provider generated output it did not send. A model that ends its
run with an empty reply counts at most 3 (Nemotron 3.5: 143 such endings, none past 3)."""


UNSENT_TAIL_TOKENS = 16
"""Completion tokens a reply with text and no tool call may leave unaccounted for: more
means the provider generated a tool call after the text and sent none of it. The text is
counted at 3.5 characters a token when short (a sentence announcing the next step, the
shape the dropped calls take) and at 2 when long (a final summary's markdown and code
pack more characters into a token), and never at fewer tokens than it has words and
punctuation marks: a short JSON answer packs fewer characters into a token than prose
(a record tagger's all-empty verdict on 11 records, 264 characters, is ~100 tokens on
GLM 5.3 and DeepSeek v4.1 flash, which 3.5 characters a token counts as 76, so the same
answer was refused on every retry, 2026-10-28)."""

SHORT_TEXT_CHARS = 300

TEXT_PIECE = re.compile(r"\w+|[^\w\s]")


def _check_unsent_tail(response: Response, content: str) -> None:
    """Raise a transient `ProviderError` for a reply with text, no tool calls and usage
    counting clearly more output than its reasoning and text (GLM 5.3 on Prime,
    2026-10-25: "I'll start by exploring the environment ..." of ~20 tokens in a
    62-token completion, finish "stop", the tool call the text announced dropped; 1 in
    6 samples of a first turn). The harness would take the turn as the end of the run."""
    usage = response.usage
    if usage is None or usage.reasoning_tokens is None:
        return
    per_token = 3.5 if len(content) <= SHORT_TEXT_CHARS else 2
    text_tokens = max(
        math.ceil(len(content) / per_token), len(TEXT_PIECE.findall(content))
    )
    unsent = usage.completion_tokens - usage.reasoning_tokens - text_tokens
    if unsent > UNSENT_TAIL_TOKENS:
        raise model_error(
            "malformed upstream response: a turn with text and no tool call whose usage "
            f"counts ~{unsent} completion tokens past its reasoning and text that never "
            "arrived",
            status_code=502,
        )


def _reroute(session, ctx, reason: str):
    """The rollout's next exchanges under a new routing id (`REROUTE_STALLS`): the
    rollout's trace id with `~r<n>`, kept on the trace (its model binding and agent
    config) so later turns and a resume stay off the wedged replica too."""
    from dataclasses import replace

    from verifiers.v1.model_routing import MODEL_BINDING

    n = int(session.trace.info.get("reroutes") or 0) + 1
    base = (ctx.session_id or session.trace.id).split("~r")[0]
    routing = f"{base}~r{n}"
    ctx = replace(ctx, session_id=routing)
    session.ctx = ctx
    session.trace.info["reroutes"] = n
    binding = session.trace.info.get(MODEL_BINDING)
    if isinstance(binding, dict):
        binding["session_id"] = routing
    config = session.trace.agent.config
    session.trace.agent.config = config.model_copy(update={"session_id": routing})
    session.trace.notify()
    logger.warning(
        "%s; rerouting: id=%s session=%s",
        reason,
        session.trace.id,
        routing,
    )
    return ctx


def _check_tool_call_leak(response: Response) -> None:
    """Raise a transient `ProviderError` for a reply with no content and no tool calls
    whose tool calls the provider failed to parse: raw tool-call markup left in its
    reasoning, or output its usage counts past the reasoning that never arrived (GLM 5.3
    on Prime, 2026-10-24: a stream ending `stop` with no tool call while its usage counts
    ~80 tokens past the reasoning, the call the reasoning announced). The empty turn would
    otherwise end the harness's run."""
    message = response.message
    if message.tool_calls:
        return
    content = (message.content or "").strip()
    if content:
        _check_unsent_tail(response, content)
        return
    if any(markup in (message.reasoning_content or "") for markup in LEAKED_TOOL_CALL):
        raise model_error(
            "malformed upstream response: an empty turn whose reasoning holds "
            "unparsed tool-call markup",
            status_code=502,
        )
    usage = response.usage
    if usage is None:
        return
    if usage.completion_tokens <= 1:
        # one end-of-turn token and nothing else, not even reasoning: GLM 5.3 on Prime
        # mid-run (2026-10-25, 10 of ProgramBench's early endings); no model ends a run
        # it reasoned through this way
        raise model_error(
            "malformed upstream response: an empty turn of "
            f"{usage.completion_tokens} completion token(s), no reasoning",
            status_code=502,
        )
    if usage.reasoning_tokens is not None:
        dropped = usage.completion_tokens - usage.reasoning_tokens
        if dropped > DROPPED_OUTPUT_TOKENS:
            raise model_error(
                f"malformed upstream response: an empty turn whose usage counts "
                f"{dropped} completion tokens past its reasoning that never arrived",
                status_code=502,
            )


def _error_response(
    dialect: Dialect,
    message: str,
    *,
    status: int,
    headers: dict[str, str] | None = None,
) -> web.Response:
    """Keep the native error classification consistent with its HTTP status."""
    return web.json_response(
        dialect.error_body(message, status_code=status), status=status, headers=headers
    )


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


def _stream_error(chunk: bytes) -> str | None:
    """The message of an SSE error event in `chunk` (a `data:` payload whose JSON object has
    an `error`), or None."""
    for line in chunk.splitlines():
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if not payload.startswith(b"{") or b'"error"' not in payload:
            continue
        try:
            data = json.loads(payload)
        except ValueError:
            continue
        if isinstance(data, dict) and data.get("error"):
            error = data["error"]
            return (
                str(error.get("message") or error)
                if isinstance(error, dict)
                else str(error)
            )
    return None


def _streams_upstream(dialect: Dialect, client_config: object) -> bool:
    """Whether a non-streaming request is sent upstream as a stream (`_as_streaming`):
    a chat completion through an eval client; a train client generates the response."""
    return dialect.name == "chat_completions" and not isinstance(
        client_config, TrainClientConfig
    )


def _as_streaming(upstream: dict | EncodedBody) -> dict:
    """The request body asking the provider to stream, with usage in the last event."""
    body = dict(
        from_json(upstream.data) if isinstance(upstream, EncodedBody) else upstream
    )
    body["stream"] = True
    body["stream_options"] = {
        **(body.get("stream_options") or {}),
        "include_usage": True,
    }
    return body


async def _collect_completion(dialect: Dialect, reply: RelayReply) -> Response:
    """A provider that answered a streaming request with a plain JSON completion: parsed as
    the eval client parses one, a malformed body a transient 502."""
    try:
        data = b"".join([chunk async for chunk in reply.chunks])
    finally:
        await reply.close()
    try:
        raw = from_json(data)
        response = dialect.parse_response(dialect.validate_response(raw))
    except (ValueError, ValidationError) as e:
        error = ProviderError(
            f"malformed upstream response: {type(e).__name__}: {e}", status_code=502
        )
        raise error from e
    response.raw = raw
    return response


async def _collect_stream(
    dialect: Dialect, reply: RelayReply, *, watch_slow: bool = True
) -> tuple[Response, bytes]:
    """Read a relayed provider stream whole: the assembled response and the events to serve
    for it. Comment-only events (the provider's own keepalives) are dropped; the served
    stream sends its own."""
    events = bytearray()
    parser = dialect.stream_parser()
    saw_terminal = False
    started = time.monotonic()
    count = 0
    last = b""
    try:
        async for chunk in reply.chunks:
            elapsed = time.monotonic() - started
            if (
                watch_slow
                and elapsed >= SLOW_STREAM_AFTER_SECONDS
                and count < SLOW_STREAM_EVENTS_PER_SECOND * elapsed
            ):
                error = ProviderError(
                    f"upstream stream too slow: {count / elapsed:.1f} events/s "
                    f"(after {elapsed:.0f}s, {count} events, {len(events)} bytes)"
                )
                error.streamed = count > 0
                error.events = count
                raise error
            if not any(line.startswith(b"data:") for line in chunk.splitlines()):
                continue
            if message := _stream_error(chunk):
                # The provider failed the turn mid-stream with an error event (Nemotron's
                # gateway: "Service error. Please retry shortly.", then [DONE]): a cut, not
                # an answer to forward, so the turn is sampled again in place. An error event
                # before any content (an overloaded gateway's 503 as its stream's first event)
                # is a refusal, not a cut: it does not restart the call's retry budget.
                error = ProviderError(
                    f"upstream stream sent an error event: {message} "
                    f"(after {time.monotonic() - started:.0f}s, {count} events, {len(events)} bytes)"
                )
                error.streamed = count > 0
                raise error
            events += chunk
            count += 1
            last = chunk
            saw_terminal |= dialect.is_terminal_event(chunk)
            if parser.on_done is not None and is_sse_done_event(chunk):
                parser.on_done()
            parser.feed(chunk)
        if not saw_terminal:
            # Where the provider cut it: how long it streamed, how much arrived, and its
            # last event, so a fixed cut-off (a proxy's stream limit) shows as one.
            tail = last.decode(errors="replace").strip()[-200:]
            error = ProviderError(
                "upstream stream ended before its terminal event "
                f"(after {time.monotonic() - started:.0f}s, {count} events, "
                f"{len(events)} bytes; last event: {tail!r})"
            )
            error.streamed = count > 0
            raise error
        return parser.finish(), bytes(events)
    except RolloutError:
        raise
    except Exception as e:  # a malformed provider stream, or its connection dropped
        error = ProviderError(
            f"{e} (after {time.monotonic() - started:.0f}s, {count} events, {len(events)} bytes)"
            if count
            else str(e)
        )
        error.streamed = count > 0
        raise error from e
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
    nor a client's own idle timeout cuts a long turn (provider retries included), and a
    failure is framed as the dialect's SSE error. A reader that goes away leaves the turn
    running, retries and all, for the harness's retry to coalesce onto."""
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
                    else dialect.stream_error(
                        from_json(replay.body), status_code=response.status
                    )
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
                from verifiers.v1.model_routing import request_binding

                session = self.sessions.get(dialect.secret(request.headers))
                if session is None:
                    response = await self.handle_request(request, dialect)
                else:
                    session.adopt(asyncio.current_task())
                    async with request_binding(session, dialect, self._client):
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
        app.router.add_get("/v1/inbox", self.handle_inbox)
        # Tool servers use a state-only capability; the model bearer cannot reach these.
        app.router.add_get("/state", self.handle_state_get)
        app.router.add_put("/state", self.handle_state_put)
        app.router.add_post("/tool", self.handle_tool)
        # A launched tool server fetches its rollout's task here to run `setup_task` — the task
        # is never passed via env, only over this channel, keyed by the state bearer.
        app.router.add_get("/task", self.handle_task_get)
        # A rollout's `message_agent` tool steers a sibling agent of its episode.
        app.router.add_post("/steer", self.handle_steer)
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
        return _error_response(
            dialect, str(error), status=getattr(error, "status_code", 502)
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
                "interception removed provider content/capabilities blocked by the network "
                "policy or unable to enforce it: id=%s paths=%s",
                session.trace.id,
                ",".join(capabilities),
            )
        return mediated, capabilities

    async def handle_inbox(self, request: web.Request) -> web.Response:
        """`GET /v1/inbox`: the user messages steered into this rollout since its harness
        last asked, oldest first (`RolloutSession.take_inbox`). A harness with an inbox
        (`Harness.SUPPORTS_INBOX`) asks once before each model call and adds them to its
        own conversation; asking acknowledges them. The model bearer keys it."""
        session = self.sessions.get(
            request.headers.get("Authorization", "").removeprefix("Bearer ")
        )
        if session is None:
            return web.json_response({"error": "unauthorized"}, status=401)
        session.adopt(asyncio.current_task())
        if session.released:
            return web.json_response({"error": "rollout concluded"}, status=409)
        return web.json_response(
            {
                "messages": [
                    {"role": "user", "content": record.text}
                    for record in session.take_inbox()
                ]
            }
        )

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
            return _error_response(dialect, "unauthorized", status=401)
        session.adopt(asyncio.current_task())
        ctx, client = session.ctx, session.client
        started = time.monotonic()
        try:
            raw = await request.read()
        except ConnectionError:
            # Client dropped mid-upload.
            logger.warning(
                "intercept: request body cut off: id=%s received=%d/%s bytes after=%.1fs",
                session.trace.id,
                request.content.total_bytes,
                request.content_length or "?",
                time.monotonic() - started,
            )
            raise
        try:
            body = from_json(raw)
        except ValueError:
            body = json.loads(raw)
        missing = object()
        original_settings = {}

        def apply_overrides(body: dict) -> dict:
            nonlocal original_settings
            # Retain only fields changed by the previous policy, not another copy
            # of the conversation. A new model starts from the harness's settings,
            # including settings the old model added or removed automatically.
            restored = dict(body)
            for key, value in original_settings.items():
                if value is missing:
                    restored.pop(key, None)
                else:
                    restored[key] = value
            overridden = dialect.apply_overrides(restored, ctx.model, ctx.sampling)
            original_settings = {
                key: restored.get(key, missing)
                for key in restored.keys() | overridden.keys()
                if restored.get(key, missing) != overridden.get(key, missing)
            }
            return overridden

        body = apply_overrides(body)
        streaming = dialect.streaming(body)
        retry = ctx.client.retry
        # The gate is the one retry layer for 429/529 only if it absorbs some.
        gate = ctx.client.request_gate
        gated = gate is not None and gate.max_rejections > 0
        # A streamed request is served whole once its turn commits: the eval client's
        # provider stream is read to the end, and the train client generates the response.
        relay = streaming and not isinstance(ctx.client, TrainClientConfig)
        # The slow-stream watchdog cuts a turn once: the cut reroutes it, and when the next
        # replica is as slow the model is (DeepSeek QC streams ~5 events/s on every replica,
        # cut and regenerated each 600 s six times over, 2026-10-30), so it runs to the end.
        watch_slow = True
        req_hash = await _request_digest(raw)
        # Keep `read()` for aiohttp's size guard, then release its cache and our local
        # alias after parsing so the wire body does not survive model inference.
        request._read_bytes = None
        del raw
        try:
            acp, upstream_headers = extract_acp_info(request.headers)
        except ValueError as error:
            return _error_response(dialect, str(error), status=400)
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
            return _error_response(
                dialect,
                "Idempotency-Key was reused with a different request",
                status=400,
            )
        if idempotent is not None and idempotent.response is not None:
            logger.debug(
                "intercept replay: id=%s (idempotent request)", session.trace.id
            )
            idempotent.completed_at = now
            return _replay_response(idempotent.response)

        try:
            model_request = dialect.parse_request(body)
        except ValueError as error:
            return _error_response(dialect, str(error), status=400)
        if session.released:
            return _error_response(dialect, "rollout concluded", status=409)
        if session.stopped:
            return _error_response(
                dialect, f"rollout stopped: {session.trace.stop_condition}", status=400
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
                return _error_response(dialect, "upstream attempt failed", status=503)
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
                return _error_response(
                    dialect, f"rollout stopped: {refused}", status=400
                )
            original_request = model_request
            model_request, request_rewrites, stopped = await session.rewrite_request(
                model_request, dialect=dialect.name
            )
            session.trace.request_rewrites.extend(request_rewrites)
            # A pinned tool result changes the request without a fresh record.
            if stopped is None and model_request != original_request:
                dialect.rewrite_request(body, original_request, model_request)
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
                session.trace, model_request.messages, model_request.tools, dialect.name
            )
            turn.commit_prompt()
            session.trace.stop(stopped)
            return _error_response(dialect, f"rollout stopped: {stopped}", status=400)

        try:
            body, policy_paths = self.mediate_capabilities(session, dialect, body)
            # Restricted mediation can mutate the body without reporting policy paths.
            if request_rewrites or session.network_policy.network_restricted:
                model_request = dialect.parse_request(body)
            turn = graph.prepare_turn(
                session.trace, model_request.messages, model_request.tools, dialect.name
            )
        except ValueError as error:
            return _error_response(dialect, str(error), status=400)
        except RolloutError as error:
            return self._fail(session, dialect, error)
        # What the exchanges need of the request from here on: the body as the client
        # sends (and resends) it, and its settings for the call records. The parsed body
        # and request go now: held through inference they were the conversation twice
        # over, and the turn's prompt shares the trace's own messages.
        upstream = client.hold(body)
        settings = {
            key: value
            for key, value in body.items()
            if key == "model" or key in dialect.sampling_fields
        }
        del body, model_request, original_request
        # The tail is what the harness added since the last turn (tool results, user
        # turns): live watchers see it now rather than with the model's reply.
        session.trace.preview(turn, turn.tail)
        if session.on_turn_input is not None:
            on_input, session.on_turn_input = session.on_turn_input, None
            on_input()

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

        async def send() -> tuple[Response, bytes | None]:
            """One provider exchange. What actually goes upstream: the native body with
            the rollout's model + sampling imposed — recorded raw on the trace, per call."""
            if relay:
                reply = await client.relay(
                    dialect,
                    upstream,
                    headers=upstream_headers,
                    session_id=ctx.session_id or session.trace.id,
                )
                response, events = await _collect_stream(
                    dialect, reply, watch_slow=watch_slow
                )
            elif _streams_upstream(dialect, ctx.client):
                # A non-streaming request still streams from the provider: bytes keep
                # arriving through a long generation, which a client read timeout would
                # otherwise abandon (the provider keeps generating it, holding one of the
                # key's concurrent slots). The client gets the assembled completion.
                reply = await client.relay(
                    dialect,
                    client.hold(_as_streaming(upstream)),
                    headers=upstream_headers,
                    session_id=ctx.session_id or session.trace.id,
                )
                if "text/event-stream" in (reply.content_type or ""):
                    response, _ = await _collect_stream(
                        dialect, reply, watch_slow=watch_slow
                    )
                else:
                    response = await _collect_completion(dialect, reply)
                events = None
            else:
                response = await client.get_response(
                    dialect,
                    upstream,
                    ctx.sampling,
                    headers=upstream_headers,
                    session_id=ctx.session_id or session.trace.id,
                    turn=turn,
                )
                events = None
            _check_tool_call_leak(response)
            return response, events

        async def exchange(*, retrying: bool) -> tuple[Response, bytes | None]:
            nonlocal ctx, client, upstream, settings, retry, gated, relay, policy_paths
            if not retrying or (session.model_change is None and session.ctx is ctx):
                return await send()
            # A rejected exchange has finished: a queued policy may govern its
            # next send without interrupting an in-flight call or rebuilding the
            # graph turn. Use the same binding/admission path as a new request.
            from verifiers.v1.model_routing import request_binding

            async with request_binding(session, dialect, self._client):
                ctx, client = session.ctx, session.client
                body = (
                    from_json(upstream.data)
                    if isinstance(upstream, EncodedBody)
                    else upstream
                )
                body = apply_overrides(body)
                body, paths = self.mediate_capabilities(session, dialect, body)
                policy_paths = list(dict.fromkeys([*policy_paths, *paths]))
                upstream = client.hold(body)
                settings = {
                    key: value
                    for key, value in body.items()
                    if key == "model" or key in dialect.sampling_fields
                }
                del body
                retry = ctx.client.retry
                gate = ctx.client.request_gate
                gated = gate is not None and gate.max_rejections > 0
                relay = streaming and not isinstance(ctx.client, TrainClientConfig)
                return await send()

        async def sample() -> web.Response:
            nonlocal upstream, ctx, watch_slow
            session.error = None
            call_response: Response | None = None
            events: bytes | None = None
            node: int | None = None
            error: Exception | None = None
            started = time.time()
            try:
                try:
                    # A transient provider failure is retried here, in place: nothing of
                    # the turn is committed until an exchange succeeds, and the harness's
                    # own request (or its retry, coalesced onto this one) waits throughout.
                    failures = 0
                    first_failure = 0.0
                    cut_offs = 0
                    refitted = False
                    stalled_starts = 0
                    while True:
                        try:
                            call_response, events = await exchange(
                                retrying=bool(failures)
                            )
                            break
                        except ProviderError as e:
                            fitted = (
                                None if refitted else _fit_output_to_window(e, upstream)
                            )
                            if fitted is not None:
                                # The prompt fits the window but not with the output the
                                # harness asked for: send it once more asking only for what
                                # fits (the harness compacts on its next turn).
                                refitted = True
                                self.record_call(
                                    session,
                                    dialect,
                                    settings,
                                    started,
                                    error=e,
                                    policy_paths=policy_paths,
                                    acp=acp,
                                )
                                upstream = client.hold(fitted)
                                logger.warning(
                                    "model call over the context window, resending with "
                                    "the output that fits: id=%s %s",
                                    session.trace.id,
                                    e,
                                )
                                started = time.time()
                                continue
                            now = time.monotonic()
                            if not failures:
                                first_failure = now
                            elif (
                                getattr(e, "streamed", False)
                                and cut_offs < MAX_STREAM_CUT_OFFS
                            ):
                                # The provider was serving the turn when it cut the stream:
                                # its time is no failing time, so the budget starts again
                                # (a long turn's regeneration would otherwise spend it all).
                                first_failure = now
                            if getattr(e, "streamed", False):
                                cut_offs += 1
                            if str(e).startswith(
                                "upstream stream too slow"
                            ) and getattr(e, "events", 0):
                                # A replica that streams slowly would be cut again for
                                # nothing; one that sent no event at all (an outage,
                                # 2026-10-30) is cut every time.
                                watch_slow = False
                            long_exchange = (
                                time.time() - started >= REROUTE_AFTER_SECONDS
                            )
                            if getattr(e, "streamed", False):
                                stalled_starts = 0
                                if long_exchange:
                                    # A long stream the provider cut: its replica served
                                    # slowly (4 tok/s against ~19 elsewhere, cut after up
                                    # to 100 minutes, GLM 5.3 2026-10-30) and would serve
                                    # the regeneration the same way.
                                    ctx = _reroute(
                                        session,
                                        ctx,
                                        "a stream ran %.0fs before the provider cut it"
                                        % (time.time() - started),
                                    )
                            elif long_exchange:
                                stalled_starts += 1
                            else:
                                stalled_starts = 0
                            if stalled_starts >= REROUTE_STALLS:
                                stalled_starts = 0
                                ctx = _reroute(
                                    session,
                                    ctx,
                                    f"model call never started {REROUTE_STALLS} times in a row",
                                )
                            wait = (
                                None
                                if session.released or session.stopped
                                else _retry_wait(
                                    e, failures, now - first_failure, retry, gated
                                )
                            )
                            if wait is None:
                                raise
                            # The failed exchange keeps its own record.
                            self.record_call(
                                session,
                                dialect,
                                settings,
                                started,
                                error=e,
                                policy_paths=policy_paths,
                                acp=acp,
                            )
                            failures += 1
                            logger.warning(
                                "model call failed, retrying in %.1fs: id=%s "
                                "failures=%d failing_for=%.0fs %s: %s",
                                wait,
                                session.trace.id,
                                failures,
                                now - first_failure,
                                type(e).__name__,
                                e,
                            )
                            await asyncio.sleep(wait)
                            started = time.time()
                    logger.debug(
                        "intercept turn: id=%s tools=%d",
                        session.trace.id,
                        len(call_response.message.tool_calls or []),
                    )
                    if session.released:  # concluded while sampling — seal holds
                        return _error_response(dialect, "rollout concluded", status=409)
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
                            call_response = dialect.parse_response(
                                dialect.validate_response(raw_response)
                            )
                            call_response.raw = raw_response
                    if session.stopped:
                        return _error_response(
                            dialect,
                            f"rollout stopped: {session.trace.stop_condition}",
                            status=400,
                        )
                    node = turn.commit(call_response)
                    session.consume_prepared(turn.tail)
                    if (committed := session.on_commit) is not None:
                        session.on_commit = None
                        committed()
                    session.trace.response_rewrites.extend(response_rewrites)
                    if stopped is None:
                        stopped = await session.gate_tool_calls(node)
                    if stopped is not None:
                        session.trace.stop(stopped)
                        return _error_response(
                            dialect, f"rollout stopped: {stopped}", status=400
                        )
                except RolloutError as e:
                    # Stash the real cause; the rollout re-raises it after the harness returns.
                    # The failure is final: a transient one was retried above, so the
                    # harness SDK is told not to retry it on top.
                    error = e
                    session.error = e
                    logger.warning(
                        "model call failed: id=%s %s: %s",
                        session.trace.id,
                        type(e).__name__,
                        e,
                    )
                    return _error_response(
                        dialect,
                        str(e),
                        status=getattr(e, "status_code", 502),
                        headers={SHOULD_RETRY_HEADER: "false"},
                    )
                except Exception as e:  # noqa: BLE001 - surface as an API error
                    error = e
                    logger.warning(
                        "model call failed: id=%s %s: %s",
                        session.trace.id,
                        type(e).__name__,
                        e,
                    )
                    return _error_response(dialect, str(e), status=502)
                except BaseException as e:
                    # A cancelled exchange (harness disconnect, shutdown) is still
                    # recorded, coupled to its cancellation.
                    error = e
                    raise
            finally:
                if node is None:
                    turn.abandon()
                _dump_rejected(upstream, error, session.trace.id)
                # The turn's one per-exchange record: settings, timing, outcome, and
                # the error that ended it (if any).
                self.record_call(
                    session,
                    dialect,
                    settings,
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
                if node is not None and session.fetched and not session.released:
                    session.land_steers(
                        node, len(session.trace.calls) - 1, len(turn.tail)
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
            return _error_response(dialect, "unauthorized", status=401)
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
            return _error_response(
                dialect, str(e), status=getattr(e, "status_code", 502)
            )
        except Exception as e:  # noqa: BLE001 - surface auxiliary relay failures
            logger.warning("aux call failed: id=%s %s", session.trace.id, e)
            return _error_response(dialect, str(e), status=502)
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
        headers = httpx.Headers(resolve_headers(config))
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
            return _error_response(dialect, str(e), status=502)
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

    async def handle_steer(self, request: web.Request) -> web.Response:
        """`POST /steer`: a rollout's `message_agent` tool, keyed by its private state
        bearer, steers a sibling seat of its episode (`vf.steer` on the seat's live
        trace) — `{seat, text}` in, a `{queued, reason}` receipt out."""
        session = self._session_for(request)
        if session is None:
            return web.json_response({"error": "unauthorized"}, status=401)
        if session.released:
            return web.json_response({"error": "rollout concluded"}, status=409)
        try:
            body = from_json(await request.read())
            seat, text = body["seat"], body["text"]
            if not isinstance(seat, str) or not isinstance(text, str):
                raise TypeError("seat and text must be strings")
        except (ValueError, KeyError, TypeError) as error:
            return web.json_response({"error": f"invalid steer: {error}"}, status=400)
        if session.peers is None:
            reason = "messaging is not enabled for this agent"
        elif (trace_id := session.peers.get(seat)) is None:
            reason = f"no agent {seat!r} in this episode"
        elif trace_id == session.trace.id:
            reason = "that seat is this agent"
        else:
            try:
                receipt = await steering.steer(
                    text, trace_id=trace_id, sender=session.trace.agent.name
                )
            except ValueError as error:
                receipt = steering.SteerReceipt(queued=False, reason=str(error))
            return web.json_response(
                receipt.model_dump(mode="json", include={"queued", "reason"})
            )
        return web.json_response({"queued": False, "reason": reason})

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
