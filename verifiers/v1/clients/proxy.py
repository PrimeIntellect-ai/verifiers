"""The proxy client: forward the program's request to the provider 1:1.

`ProxyClient` (the default) forwards the program's request body verbatim to an OpenAI-
compatible endpoint — mutating only what the eval owns (model + sampling, via the dialect's
`apply_overrides`) — and parses the provider's response into a vf `Response` (via the request's
dialect) for the trace, carrying the raw response on `Response.raw` so the interception server
hands it back to the program untouched, no field lost to a typed round-trip.

The transport is `AsyncOpenAI` used as a raw poster (`post(..., cast_to=httpx.Response)`): it
handles the OpenAI-compatible endpoint family (base url, Bearer auth, billing headers) that all
OpenAI-SDK dialects share. Everything protocol-specific lives in the `Dialect`.
"""

import httpx
from openai import AsyncOpenAI, OpenAIError

from verifiers.v1.clients.client import Client
from verifiers.v1.dialects import Dialect
from verifiers.v1.errors import model_error
from verifiers.v1.types import Response, SamplingConfig


class ProxyClient(Client):
    """The default client: forward the program's request 1:1 to an OpenAI-compatible endpoint,
    parse the provider's response into a vf `Response` (via the request's dialect) for the trace,
    and carry the raw response on `Response.raw` so it reaches the program untouched."""

    def __init__(self, openai: AsyncOpenAI) -> None:
        self.openai = openai

    async def get_response(
        self,
        body: dict,
        dialect: Dialect,
        model: str,
        sampling_args: SamplingConfig,
    ) -> Response:
        # Byte-exact forward, save for the eval's model + sampling (imposed by the dialect).
        upstream = dialect.apply_overrides(body, model, sampling_args)
        try:
            resp = await self.openai.post(
                dialect.upstream_path, cast_to=httpx.Response, body=upstream
            )
        except OpenAIError as e:
            raise model_error(e) from e
        raw = resp.json()
        response = dialect.parse_response(dialect.response_type.model_validate(raw))
        response.raw = raw  # the program gets the provider's bytes back 1:1
        return response

    async def close(self) -> None:
        await self.openai.close()
