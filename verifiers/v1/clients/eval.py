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

import re

import httpx

from verifiers.v1.clients.client import SESSION_ID_HEADER, Client, RelayReply
from verifiers.v1.dialects import Dialect
from verifiers.v1.errors import model_error
from verifiers.v1.types import Response, SamplingConfig

_SSE_EVENT_END = re.compile(rb"(?:\r\n|\r|\n){2}")


class EvalClient(Client):
    """The default client: forward the program's request 1:1 to the provider, parse the response
    into a vf `Response` (via the request's dialect) for the trace, and carry the raw response on
    `Response.raw` so it reaches the program untouched."""

    def __init__(
        self, base_url: str, api_key: str, headers: dict[str, str] | None = None
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        # No timeout: agentic completions are slow and the rollout timeout is the real backstop.
        # Build full URLs ourselves (base_url + dialect.upstream_path) rather than relying on
        # httpx base-url joining, which drops the base path for a leading-slash request path.
        self.http = httpx.AsyncClient(timeout=None, headers=headers or {})

    async def get_response(
        self,
        dialect: Dialect,
        body: dict,
        model: str,
        sampling_args: SamplingConfig,
        session_id: str | None = None,
    ) -> Response:
        # Byte-exact forward, save for the eval's model + sampling (imposed by the dialect).
        url, headers, upstream = self._upstream(
            dialect, body, model, sampling_args, session_id
        )
        try:
            resp = await self.http.post(url, json=upstream, headers=headers)
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
        self,
        dialect: Dialect,
        body: dict,
        model: str,
        sampling_args: SamplingConfig,
        session_id: str | None = None,
    ) -> tuple[str, dict, dict]:
        """The (url, headers, steered body) for a forwarded request — shared by the non-stream
        and streaming paths. A `session_id` is forwarded as the `SESSION_ID_HEADER` so the
        provider's router can pin the rollout's turns to one engine."""
        headers = dialect.auth_headers(self.api_key)
        if session_id:
            headers = {**headers, SESSION_ID_HEADER: session_id}
        return (
            self.base_url + dialect.upstream_path,
            headers,
            dialect.apply_overrides(body, model, sampling_args),
        )

    async def relay(
        self,
        dialect: Dialect,
        body: dict,
        model: str,
        sampling_args: SamplingConfig,
        session_id: str | None = None,
    ) -> RelayReply:
        # Relay complete SSE events so the interception server can safely insert keepalives
        # between them. Error responses are mapped before any event is handed back.
        url, headers, upstream = self._upstream(
            dialect, body, model, sampling_args, session_id
        )
        request = self.http.build_request("POST", url, json=upstream, headers=headers)
        try:
            resp = await self.http.send(request, stream=True)
        except httpx.HTTPError as e:
            raise model_error(str(e)) from e
        if resp.status_code >= 400:
            try:
                text = (await resp.aread()).decode("utf-8", errors="replace")
            finally:
                await resp.aclose()
            raise model_error(f"upstream {resp.status_code}: {text}")

        async def chunks():
            buffer = bytearray()
            async for chunk in resp.aiter_bytes():
                buffer += chunk
                while match := _SSE_EVENT_END.search(buffer):
                    yield bytes(buffer[: match.end()])
                    del buffer[: match.end()]
            if buffer:
                yield bytes(buffer)

        return RelayReply(
            content_type=resp.headers.get("content-type", "text/event-stream"),
            chunks=chunks(),
            close=resp.aclose,
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
