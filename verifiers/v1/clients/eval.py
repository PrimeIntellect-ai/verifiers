"""The eval client: relay the program's native request to the provider.

`EvalClient` (the default) is a thin `httpx` forwarder: it sends the program's request body
without a typed round-trip, mutating only what the eval owns (model + sampling, via the dialect's
`apply_overrides`). Eligible end-to-end request headers are forwarded too; rollout auth, body
framing, and connection headers are replaced. The provider response is parsed into a vf
`Response` for the trace, while its full JSON object stays on `Response.raw` for the interception
server to return.

The transport is provider-agnostic: the dialect supplies the canonical route + auth headers, so
a new wire format (incl. non-OpenAI providers like Anthropic) is just a new `Dialect` — no client
change. Endpoint config (base url, api key, billing headers) comes from the client config.
"""

from collections.abc import Mapping
import re

import httpx
from pydantic import ValidationError
from pydantic_core import from_json, to_json

from verifiers.v1.clients.client import SESSION_ID_HEADER, Client, RelayReply
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


class EvalClient(Client):
    """Relay native JSON to the provider and parse a copy for the trace."""

    def __init__(
        self, base_url: str, api_key: str, headers: dict[str, str] | None = None
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        # Keep endpoint headers separate so they can override intercepted request headers before
        # the dialect's provider authentication is applied.
        self.headers = dict(headers or {})
        # No timeout: agentic completions are slow and the rollout timeout is the real backstop.
        # base_url includes the provider's /v1 prefix. Build full URLs ourselves because httpx
        # base-url joining drops the base path for a leading-slash request path.
        # Match V1's default concurrency while retaining HTTPX's 20-idle keepalive bound.
        self.http = httpx.AsyncClient(
            timeout=None,
            limits=httpx.Limits(max_connections=128, max_keepalive_connections=20),
        )

    async def get_response(
        self,
        dialect: Dialect,
        body: dict,
        model: str,
        sampling_args: SamplingConfig,
        session_id: str | None = None,
        turn: PendingTurn | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        resp = await self._request(
            self.base_url + dialect.route.removeprefix("/v1"),
            dialect.apply_overrides(body, model, sampling_args),
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
        """Build provider headers from the intercepted request.

        Preserve provider feature headers such as `openai-beta` / `anthropic-beta`,
        discard localhost auth and transport framing, then apply endpoint-configured headers,
        session routing, and real provider auth.
        """
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
        request = self.http.build_request(
            "POST",
            url,
            content=to_json(body, inf_nan_mode="null"),
            headers=headers,
        )
        try:
            response = await self.http.send(request, stream=stream)
        except httpx.TimeoutException as e:
            raise model_error(str(e), status_code=504) from e
        except httpx.HTTPError as e:
            raise model_error(str(e), status_code=503) from e
        except ConnectionResetError as e:
            raise model_error(str(e), status_code=503) from e
        if not stream:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                # relay the provider's status (and body) so the harness SDK retries 5xx/429 and not
                # 4xx; an empty/HTML body (e.g. a 404 from a base_url missing `/v1`) would otherwise
                # make an information-free ProviderError
                raise model_error(
                    f"upstream {e.response.status_code}: {e.response.text}",
                    status_code=e.response.status_code,
                ) from e
            return response
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
        model: str,
        sampling_args: SamplingConfig,
        session_id: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> RelayReply:
        # Relay complete SSE events so the interception server can safely insert keepalives
        # between them. Error responses are mapped before any event is handed back.
        resp = await self._request(
            self.base_url + dialect.route.removeprefix("/v1"),
            dialect.apply_overrides(body, model, sampling_args),
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
            self.base_url + route.removeprefix("/v1"),
            body,
            self._headers(dialect, headers, None),
        )
        return from_json(resp.content)

    async def close(self) -> None:
        await self.http.aclose()
