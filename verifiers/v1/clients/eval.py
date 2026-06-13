"""The eval client: relay the program's native request to the provider.

`EvalClient` (the default) is a thin `httpx` forwarder: it sends the program's request body
without a typed round-trip, mutating only what the eval owns (model + sampling, via the dialect's
`apply_overrides`). Safe end-to-end request headers are forwarded too; transport headers and the
rollout secret are replaced. The provider response is parsed into a vf `Response` for the trace,
while its full JSON object stays on `Response.raw` for the interception server to return.

The transport is provider-agnostic: the dialect supplies the upstream path + auth headers, so a
new wire format (incl. non-OpenAI providers like Anthropic) is just a new `Dialect` — no client
change. Endpoint config (base url, api key, billing headers) comes from the client config.
"""

from collections.abc import Mapping
from contextlib import aclosing

import httpx

from verifiers.v1.clients.client import Client, RelayReply
from verifiers.v1.dialects import Dialect
from verifiers.v1.errors import model_error
from verifiers.v1.types import Response, SamplingConfig

# These describe the localhost connection or its request framing. HTTPX must rebuild them for
# the provider connection; forwarding them can send the wrong host or body length upstream.
_PROXY_MANAGED_HEADERS = frozenset(
    {
        "connection",
        "content-length",
        "host",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
# The incoming bearer token authenticates the harness to the interception server. Never leak that
# rollout secret upstream; the dialect adds the real provider credentials after filtering.
_INTERCEPTION_AUTH_HEADERS = frozenset({"authorization"})


class EvalClient(Client):
    """Relay native JSON to the provider and parse a copy for the trace."""

    def __init__(
        self, base_url: str, api_key: str, headers: dict[str, str] | None = None
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        # Keep endpoint headers separate so they can override intercepted request headers before
        # the dialect's provider authentication is applied.
        self.headers = headers or {}
        # No timeout: agentic completions are slow and the rollout timeout is the real backstop.
        # Build full URLs ourselves (base_url + dialect.upstream_path) rather than relying on
        # httpx base-url joining, which drops the base path for a leading-slash request path.
        self.http = httpx.AsyncClient(timeout=None)

    async def get_response(
        self,
        dialect: Dialect,
        body: dict,
        model: str,
        sampling_args: SamplingConfig,
        request_headers: Mapping[str, str] | None = None,
    ) -> Response:
        url, headers, upstream = self._upstream(
            dialect, body, model, sampling_args, request_headers
        )
        resp = await self._request(url, upstream, headers)
        raw = resp.json()
        response = dialect.parse_response(dialect.response_type.model_validate(raw))
        response.raw = raw
        return response

    def _headers(
        self,
        dialect: Dialect,
        request_headers: Mapping[str, str] | None,
    ) -> dict[str, str]:
        """Build provider headers from the intercepted request.

        Preserve provider feature headers such as `openai-beta`, discard localhost auth and
        transport framing, then apply endpoint-configured headers and real provider auth.
        """
        incoming = httpx.Headers(request_headers)
        blocked = (
            _PROXY_MANAGED_HEADERS
            | _INTERCEPTION_AUTH_HEADERS
            # RFC 9110 allows `Connection` to name additional hop-by-hop fields.
            | {
                name.strip().lower()
                for name in incoming.get("connection", "").split(",")
                if name.strip()
            }
        )
        forwarded = {
            name: value
            for name, value in incoming.items()
            if name.lower() not in blocked
        }
        headers = httpx.Headers(forwarded)
        headers.update(self.headers)
        headers.update(dialect.auth_headers(self.api_key))
        return dict(headers)

    def _upstream(
        self,
        dialect: Dialect,
        body: dict,
        model: str,
        sampling_args: SamplingConfig,
        request_headers: Mapping[str, str] | None,
    ) -> tuple[str, dict, dict]:
        """The (url, headers, steered body) for a forwarded request — shared by the non-stream
        and streaming paths."""
        return (
            self.base_url + dialect.upstream_path,
            self._headers(dialect, request_headers),
            dialect.apply_overrides(body, model, sampling_args),
        )

    async def _request(
        self,
        url: str,
        body: dict,
        headers: dict[str, str],
        *,
        stream: bool = False,
    ) -> httpx.Response:
        request = self.http.build_request("POST", url, json=body, headers=headers)
        try:
            response = await self.http.send(request, stream=stream)
        except httpx.HTTPError as e:
            raise model_error(str(e)) from e
        if response.status_code < 400:
            return response
        text = (await response.aread()).decode("utf-8", errors="replace")
        await response.aclose()
        raise model_error(f"upstream {response.status_code}: {text}")

    async def relay(
        self,
        dialect: Dialect,
        body: dict,
        model: str,
        sampling_args: SamplingConfig,
        request_headers: Mapping[str, str] | None = None,
    ) -> RelayReply:
        # Stream the provider's response bytes through (SSE for a streaming request). An error
        # status is read fully and mapped before any byte is handed back, so the retry +
        # truncation machinery treat a relayed call exactly like a non-streamed one.
        url, headers, upstream = self._upstream(
            dialect, body, model, sampling_args, request_headers
        )
        resp = await self._request(url, upstream, headers, stream=True)

        async def chunks():
            async with aclosing(resp):
                async for chunk in resp.aiter_bytes():
                    yield chunk

        return RelayReply(
            content_type=resp.headers.get("content-type", "text/event-stream"),
            chunks=chunks(),
        )

    async def relay_aux(self, dialect: Dialect, route: str, body: dict) -> dict:
        # A side request (e.g. count_tokens): relay its native JSON and return the provider JSON.
        resp = await self._request(
            self.base_url + route,
            body,
            self._headers(dialect, None),
        )
        return resp.json()

    async def close(self) -> None:
        await self.http.aclose()
