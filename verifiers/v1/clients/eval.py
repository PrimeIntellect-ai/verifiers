"""The proxy client: forward the program's request to the provider 1:1.

`EvalClient` (the default) is a thin `httpx` forwarder: it sends the program's request body
verbatim to the provider — mutating only what the eval owns (model + sampling, via the dialect's
`apply_overrides`) — and parses the provider's response into a vf `Response` (via the request's
dialect) for the trace, carrying the raw response on `Response.raw` so the interception server
hands it back to the program untouched, no field lost to a typed round-trip.

The transport is provider-agnostic: the dialect supplies the upstream path + auth headers, so a
new wire format (incl. non-OpenAI providers like Anthropic) is just a new `Dialect` — no client
change. Endpoint config (base url, api key, billing headers) comes from the client config.
"""

import httpx

from verifiers.v1.clients.client import Client, RelayReply
from verifiers.v1.dialects import Dialect
from verifiers.v1.errors import model_error
from verifiers.v1.types import Response, SamplingConfig

# Generous transport defaults (mirroring the v0 client): a long read timeout for slow agentic
# completions, and a high connection ceiling so one process can fan out many concurrent rollouts.
_TIMEOUT = httpx.Timeout(3600.0, connect=10.0)
_LIMITS = httpx.Limits(max_connections=28000, max_keepalive_connections=28000)


class EvalClient(Client):
    """The default client: forward the program's request 1:1 to the provider, parse the response
    into a vf `Response` (via the request's dialect) for the trace, and carry the raw response on
    `Response.raw` so it reaches the program untouched."""

    def __init__(
        self, base_url: str, api_key: str, headers: dict[str, str] | None = None
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        # Build full URLs ourselves (base_url + dialect.upstream_path) rather than relying on
        # httpx base-url joining, which drops the base path for a leading-slash request path.
        self.http = httpx.AsyncClient(
            timeout=_TIMEOUT, limits=_LIMITS, headers=headers or {}
        )

    async def get_response(
        self,
        dialect: Dialect,
        body: dict,
        model: str,
        sampling_args: SamplingConfig,
    ) -> Response:
        # Byte-exact forward, save for the eval's model + sampling (imposed by the dialect).
        upstream = dialect.apply_overrides(body, model, sampling_args)
        try:
            resp = await self.http.post(
                self.base_url + dialect.upstream_path,
                json=upstream,
                headers=dialect.auth_headers(self.api_key),
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            # The provider's error body carries the message (e.g. a context-length 400); the
            # status-error str alone doesn't, so map from the body to catch overlong prompts.
            raise model_error(e.response.text) from e
        except httpx.HTTPError as e:  # connect / read timeout / transport failure
            raise model_error(str(e)) from e
        raw = resp.json()
        response = dialect.parse_response(dialect.response_type.model_validate(raw))
        response.raw = raw  # the program gets the provider's bytes back 1:1
        return response

    def _upstream(
        self, dialect: Dialect, body: dict, model: str, sampling_args: SamplingConfig
    ) -> tuple[str, dict, dict]:
        """The (url, headers, steered body) for a forwarded request — shared by the non-stream
        and streaming paths."""
        return (
            self.base_url + dialect.upstream_path,
            dialect.auth_headers(self.api_key),
            dialect.apply_overrides(body, model, sampling_args),
        )

    async def relay(
        self,
        dialect: Dialect,
        body: dict,
        model: str,
        sampling_args: SamplingConfig,
    ) -> RelayReply:
        # Stream the provider's response bytes through (SSE for a streaming request). An error
        # status is read fully and mapped before any byte is handed back, so the retry +
        # truncation machinery treat a relayed call exactly like a non-streamed one.
        url, headers, upstream = self._upstream(dialect, body, model, sampling_args)
        request = self.http.build_request("POST", url, json=upstream, headers=headers)
        try:
            resp = await self.http.send(request, stream=True)
        except httpx.HTTPError as e:
            raise model_error(str(e)) from e
        if resp.status_code >= 400:
            text = (await resp.aread()).decode("utf-8", errors="replace")
            await resp.aclose()
            raise model_error(f"upstream {resp.status_code}: {text}")

        async def chunks():
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
            finally:
                await resp.aclose()

        return RelayReply(
            content_type=resp.headers.get("content-type", "text/event-stream"),
            chunks=chunks(),
        )

    async def relay_aux(self, dialect: Dialect, route: str, body: dict) -> dict:
        # A side request (e.g. count_tokens): forward verbatim to the provider, return its JSON.
        try:
            resp = await self.http.post(
                self.base_url + route,
                json=body,
                headers=dialect.auth_headers(self.api_key),
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise model_error(e.response.text) from e
        except httpx.HTTPError as e:
            raise model_error(str(e)) from e
        return resp.json()

    async def close(self) -> None:
        await self.http.aclose()
