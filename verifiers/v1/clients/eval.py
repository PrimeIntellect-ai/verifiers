"""The eval client: relay the program's native request to the provider.

`EvalClient` (the default) is a thin `httpx` forwarder: it sends the program's request body
without a typed round-trip, mutating only what the eval owns (model + sampling, via the dialect's
`apply_overrides`). Eligible end-to-end request headers are forwarded too; rollout auth, body
framing, and connection headers are replaced. The provider response is parsed into a vf
`Response` for the trace, while its full JSON object stays on `Response.raw` for the interception
server to return.

The transport is provider-agnostic: the dialect supplies the upstream path + auth headers, so a
new wire format (incl. non-OpenAI providers like Anthropic) is just a new `Dialect` — no client
change. Endpoint config (base url, api key, billing headers) comes from the client config.
"""

from collections.abc import Mapping
from contextlib import aclosing

import httpx

from verifiers.v1.clients.client import (
    BLOCKED_REQUEST_HEADERS,
    SESSION_ID_HEADER,
    Client,
    RelayReply,
)
from verifiers.v1.dialects import ChatDialect, Dialect
from verifiers.v1.errors import model_error
from verifiers.v1.types import Response, SamplingConfig


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
        # Build full URLs ourselves (base_url + dialect.upstream_path) rather than relying on
        # httpx base-url joining, which drops the base path for a leading-slash request path.
        self.http = httpx.AsyncClient(timeout=None)

    async def get_response(
        self,
        dialect: Dialect,
        body: dict,
        model: str,
        sampling_args: SamplingConfig,
        session_id: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        resp = await self._request(
            self.base_url + dialect.upstream_path,
            dialect.apply_overrides(body, model, sampling_args),
            self._headers(dialect, headers, session_id),
        )
        raw = resp.json()
        response = dialect.parse_response(dialect.response_type.model_validate(raw))
        response.raw = raw
        return response

    def _headers(
        self,
        dialect: Dialect,
        incoming: Mapping[str, str] | None,
        session_id: str | None,
    ) -> httpx.Headers:
        """Build provider headers from the intercepted request.

        Preserve provider feature headers such as `openai-beta`, discard localhost auth and
        transport framing, then apply endpoint-configured headers, session routing, and real
        provider auth.
        """
        headers = httpx.Headers(incoming if isinstance(dialect, ChatDialect) else None)
        connection = headers.pop("connection", "")
        for name in BLOCKED_REQUEST_HEADERS | set(
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
        request = self.http.build_request("POST", url, json=body, headers=headers)
        try:
            response = await self.http.send(request, stream=stream)
        except httpx.HTTPError as e:
            raise model_error(str(e)) from e
        if not stream:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise model_error(e.response.text) from e
            return response
        if response.status_code < 400:
            return response
        try:
            text = (await response.aread()).decode("utf-8", errors="replace")
        finally:
            await response.aclose()
        raise model_error(f"upstream {response.status_code}: {text}")

    async def relay(
        self,
        dialect: Dialect,
        body: dict,
        model: str,
        sampling_args: SamplingConfig,
        session_id: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> RelayReply:
        # Stream the provider's response bytes through (SSE for a streaming request). An error
        # status is read fully and mapped before any byte is handed back, so the retry +
        # truncation machinery treat a relayed call exactly like a non-streamed one.
        resp = await self._request(
            self.base_url + dialect.upstream_path,
            dialect.apply_overrides(body, model, sampling_args),
            self._headers(dialect, headers, session_id),
            stream=True,
        )

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
            self._headers(dialect, None, None),
        )
        return resp.json()

    async def close(self) -> None:
        await self.http.aclose()
