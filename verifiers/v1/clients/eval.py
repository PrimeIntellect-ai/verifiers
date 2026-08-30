"""The eval client: proxies harness-native request to the provider."""

import asyncio
import re
from collections.abc import Mapping

import httpx
from pydantic import ValidationError
from pydantic_core import from_json, to_json

from verifiers.v1.clients.base import DEFAULT_LIMITS, DEFAULT_TIMEOUT, join_url
from verifiers.v1.clients.client import SESSION_ID_HEADER, Client, RelayReply
from verifiers.v1.configs.client import BaseClientConfig, resolve_api_key
from verifiers.v1.dialects import Dialect
from verifiers.v1.errors import model_error
from verifiers.v1.graph import PendingTurn
from verifiers.v1.types import Response, SamplingConfig

# These fields describe the localhost request, its original bytes, or its connection. HTTPX
# rebuilds the provider request from JSON; endpoint configuration and provider auth apply last.
_BLOCKED_REQUEST_HEADERS = frozenset(
    {
        # The harness uses this rollout secret to authenticate with the localhost server.
        # The dialect adds the actual provider authorization after filtering.
        "authorization",
        # HTTPX recalculates these for the provider URL, JSON bytes, and supported decoders.
        "accept-encoding",
        "content-encoding",
        "content-length",
        "content-type",
        "host",
        "transfer-encoding",
        # These control only the localhost HTTP exchange.
        "expect",
        "keep-alive",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "upgrade",
        # Provider affinity must not compete with the rollout-wide session header below.
        "session_id",
        # The eval owns the model and sampling settings, so it changes those JSON fields before
        # sending upstream. Hashes and signatures calculated from the intercepted body are stale.
        "content-digest",
        "content-md5",
        "digest",
        "repr-digest",
        "signature",
        "signature-input",
    }
)


# Atomic so one CRLF cannot backtrack into two line endings and split an event mid-field.
_SSE_EVENT_END = re.compile(rb"(?>\r\n|\r|\n){2}")

_UPSTREAM_ATTEMPTS = 7
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504, 520, 522, 529})
_BACKOFF_CAP_SECONDS = 60.0


class EvalClient(Client):
    """Relay native JSON to the provider and parse a copy for the trace."""

    def __init__(self, config: BaseClientConfig) -> None:
        self.base_url = config.base_url
        self.api_key = resolve_api_key(config)
        # Keep endpoint headers separate so they can override intercepted request headers before
        # the dialect's provider authentication is applied.
        self.headers = dict(config.headers or {})
        self.client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, limits=DEFAULT_LIMITS)

    async def get_response(
        self,
        dialect: Dialect,
        body: dict,
        sampling: SamplingConfig,
        session_id: str | None = None,
        turn: PendingTurn | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        resp = await self._request(
            join_url(self.base_url, dialect.upstream_path),
            body,
            self._headers(dialect, headers, session_id),
        )
        # A corrupted response (e.g. an HTML error page or a truncated body on a
        # flaky tunnel) surfaces as a JSON parse failure or a schema validation
        # failure — map these to a retryable 502 so the harness SDK retries the
        # call instead of crashing the whole rollout on one bad response.
        try:
            raw = from_json(resp.content)
            response = dialect.parse_response(dialect.validate_response(raw))
        except (ValueError, ValidationError) as e:
            raise model_error(
                f"malformed upstream response: {type(e).__name__}: {e}",
                status_code=502,
            ) from e
        # The interception server returns this full native provider object to the program.
        response.raw = raw
        return response

    def _headers(
        self,
        dialect: Dialect,
        incoming: Mapping[str, str] | None,
        session_id: str | None,
    ) -> httpx.Headers:
        """Provider headers from the intercepted request: keep feature headers
        (`openai-beta`, `anthropic-beta`), discard localhost auth and framing, then apply
        endpoint headers, session routing, and real provider auth."""
        headers = httpx.Headers(incoming)
        connection = headers.pop("connection", "")
        for name in _BLOCKED_REQUEST_HEADERS | set(
            map(str.strip, connection.lower().split(","))
        ):
            headers.pop(name, None)
        headers.update(self.headers)
        if session_id:
            headers[SESSION_ID_HEADER] = session_id
        headers.update(dialect.auth_headers(self.api_key))
        return headers

    async def _request(
        self,
        url: str,
        body: dict,
        headers: httpx.Headers,
        *,
        stream: bool = False,
    ) -> httpx.Response:
        headers.setdefault("content-type", "application/json")
        # Load-shed and transport faults retry HERE, next to the provider, before the
        # failure crosses the tunnel: an upstream 5xx/429 or a dropped connection is
        # infrastructure answering for the model, and surfacing it un-retried kills an
        # otherwise healthy session (measured live: "upstream 504: upstream request
        # timeout" storms under fleet concurrency). Bounded and short — the harness
        # keeps its own outer ladder.
        last_error: Exception | None = None
        for attempt in range(1, _UPSTREAM_ATTEMPTS + 1):
            request = self.client.build_request(
                "POST",
                url,
                content=to_json(body, inf_nan_mode="null"),
                headers=headers,
            )
            try:
                response = await self.client.send(request, stream=stream)
            except httpx.TimeoutException as e:
                last_error = model_error(str(e), status_code=504)
                last_error.__cause__ = e
            except httpx.HTTPError as e:
                last_error = model_error(str(e), status_code=503)
                last_error.__cause__ = e
            except ConnectionResetError as e:
                last_error = model_error(str(e), status_code=503)
                last_error.__cause__ = e
            else:
                if stream:
                    break
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as e:
                    last_error = model_error(
                        f"upstream {e.response.status_code}: {e.response.text}",
                        status_code=e.response.status_code,
                    )
                    last_error.__cause__ = e
                    if e.response.status_code not in _RETRY_STATUSES:
                        raise last_error
                else:
                    return response
            if attempt < _UPSTREAM_ATTEMPTS:
                # Exponential to a minute: measured 504 storms run multi-minute, and a
                # turn stalled in retry beats a dead session under wall-clock stops.
                await asyncio.sleep(min(_BACKOFF_CAP_SECONDS, 2.0 * (2 ** (attempt - 1))))
        if not stream:
            raise last_error
        if response.status_code < 400:
            return response
        try:
            text = (await response.aread()).decode("utf-8", errors="replace")
        finally:
            await response.aclose()
        raise model_error(
            f"upstream {response.status_code}: {text}", status_code=response.status_code
        )

    async def relay(
        self,
        dialect: Dialect,
        body: dict,
        session_id: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> RelayReply:
        # Relay complete SSE events so the interception server can safely insert keepalives
        # between them. Error responses are mapped before any event is handed back.
        resp = await self._request(
            join_url(self.base_url, dialect.upstream_path),
            body,
            self._headers(dialect, headers, session_id),
            stream=True,
        )

        async def chunks():
            buffer = bytearray()
            search_from = 0
            async for chunk in resp.aiter_bytes():
                buffer += chunk
                while match := _SSE_EVENT_END.search(buffer, search_from):
                    yield bytes(buffer[: match.end()])
                    del buffer[: match.end()]
                    search_from = 0
                # A delimiter is at most four bytes and can straddle chunks.
                search_from = max(0, len(buffer) - 3)
            if buffer:
                yield bytes(buffer)

        return RelayReply(
            content_type=resp.headers.get("content-type", "text/event-stream"),
            chunks=chunks(),
            close=resp.aclose,
        )

    async def relay_aux(
        self,
        dialect: Dialect,
        route: str,
        body: dict,
        headers: Mapping[str, str] | None = None,
    ) -> dict:
        # A side request (e.g. count_tokens): relay its native JSON and return the provider JSON.
        resp = await self._request(
            join_url(self.base_url, route),
            body,
            self._headers(dialect, headers, None),
        )
        return from_json(resp.content)

    async def close(self) -> None:
        await self.client.aclose()
