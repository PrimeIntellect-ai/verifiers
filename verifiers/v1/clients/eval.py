"""Forward native evaluation requests through the provider SDKs."""

import asyncio
from collections.abc import Mapping

import anthropic
import httpx2 as httpx
import openai

from verifiers.v1.clients.base import (
    DEFAULT_LIMITS,
    DEFAULT_TIMEOUT,
    MAX_RETRIES,
    SESSION_ID_HEADER,
    join_url,
)
from verifiers.v1.clients.client import Client
from verifiers.v1.configs.client import BaseClientConfig, resolve_api_key
from verifiers.v1.dialects import AnthropicDialect, ChatDialect, Dialect
from verifiers.v1.errors import RolloutError, model_error
from verifiers.v1.graph import PendingTurn
from verifiers.v1.semantic import ACP_EXTENSION_HEADERS
from verifiers.v1.types import Response, SamplingConfig

# Local authentication, framing, and signatures must not reach the provider.
_BLOCKED_REQUEST_HEADERS = (
    frozenset(
        {
            # The harness uses this rollout secret to authenticate with the localhost server.
            # The SDK supplies provider authorization after filtering.
            "authorization",
            "x-api-key",
            "x-stainless-raw-response",
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
    | ACP_EXTENSION_HEADERS
)


class EvalClient(Client):
    """Preserve native payloads; project SDK responses into the trace."""

    def __init__(self, config: BaseClientConfig) -> None:
        self.base_url = config.base_url
        self.headers = dict(config.headers or {})
        options = {
            "base_url": config.base_url,
            "api_key": resolve_api_key(config),
            "timeout": DEFAULT_TIMEOUT,
            "max_retries": MAX_RETRIES,
            "http_client": httpx.AsyncClient(
                timeout=DEFAULT_TIMEOUT, limits=DEFAULT_LIMITS
            ),
        }
        self.openai = openai.AsyncOpenAI(**options)
        self.anthropic = anthropic.AsyncAnthropic(**options)

    async def _complete(
        self,
        dialect: Dialect,
        body: dict,
        sampling: SamplingConfig,
        session_id: str | None = None,
        turn: PendingTurn | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> tuple[Response, bytes | None]:
        client, reply = await self._request(
            dialect, dialect.upstream_path, body, headers, session_id
        )
        try:
            events = None
            if body.get("stream"):
                # Parse during receipt, retaining decoded bytes for atomic delivery/replay.
                # Bounded chunks keep cached or coalesced responses from starving other turns.
                recorded = bytearray()
                chunks = reply.iter_bytes(chunk_size=16_384)

                async def capture():
                    async for chunk in chunks:
                        recorded.extend(chunk)
                        yield chunk
                        await asyncio.sleep(0)

                stream_type = (
                    anthropic.AsyncStream
                    if isinstance(client, anthropic.AsyncAnthropic)
                    else openai.AsyncStream
                )
                async with stream_type(
                    cast_to=dialect.event_type,
                    response=httpx.Response(
                        reply.status_code,
                        request=reply.http_response.request,
                        content=capture(),
                    ),
                    client=client,
                ) as stream:
                    raw = await dialect.read_stream(stream)
                # Chat's SDK stops at [DONE]; retain any trailing provider bytes too.
                async for chunk in chunks:
                    recorded.extend(chunk)
                    await asyncio.sleep(0)
                events = bytes(recorded)
                if isinstance(dialect, ChatDialect) and not await asyncio.to_thread(
                    any,
                    (
                        event.data == "[DONE]"
                        for event in anthropic.Stream.raw_events(
                            httpx.Response(200, content=events)
                        )
                    ),
                ):
                    raise ValueError("Chat stream ended without [DONE]")
            else:
                raw = await reply.json()
            response = dialect.parse_response(raw)
        except (
            openai.OpenAIError,
            anthropic.AnthropicError,
            httpx.HTTPError,
            ConnectionResetError,
        ) as error:
            raise model_error(error) from error
        except RolloutError:
            raise
        except Exception as error:
            # SDK accumulation can reject malformed event sequences.
            raise model_error(
                f"malformed upstream response: {error}", status_code=502
            ) from error
        finally:
            await reply.close()
        response.raw = raw
        return response, events

    def _headers(
        self,
        incoming: Mapping[str, str] | None,
        session_id: str | None,
    ) -> dict[str, str]:
        """Retain feature headers; let the SDK supply provider auth and HTTP framing."""
        headers = httpx.Headers(incoming)
        connection = headers.pop("connection", "")
        for name in _BLOCKED_REQUEST_HEADERS | set(
            map(str.strip, connection.lower().split(","))
        ):
            headers.pop(name, None)
        headers.update(self.headers)
        # Configured headers cannot replace the provider key with a local rollout secret.
        headers.pop("authorization", None)
        headers.pop("x-api-key", None)
        if session_id:
            headers[SESSION_ID_HEADER] = session_id
        return dict(headers)

    async def _request(
        self,
        dialect: Dialect,
        route: str,
        body: dict,
        headers: Mapping[str, str] | None,
        session_id: str | None = None,
    ):
        client, response_type = (
            (self.anthropic, anthropic.AsyncAPIResponse)
            if isinstance(dialect, AnthropicDialect)
            else (self.openai, openai.AsyncAPIResponse)
        )
        request_headers = self._headers(headers, session_id)
        # SDK defaults are plain dicts: match their casing so forwarded headers replace
        # them instead of producing duplicate User-Agent or provider feature headers.
        default_names = {name.lower(): name for name in client.default_headers}
        request_headers = {
            default_names.get(name.lower(), name): value
            for name, value in request_headers.items()
        }
        if body.get("stream"):
            # Ask the SDK for its unread response wrapper instead of a typed SSE iterator.
            request_headers["X-Stainless-Raw-Response"] = "stream"
        try:
            # Low-level SDK requests preserve extension fields without signature filtering.
            # The raw wrapper retains bytes and supports the SDK's permissive typed parse.
            reply = await client.post(
                join_url(self.base_url, route),
                body=body,
                cast_to=response_type[dict],
                options={"headers": request_headers},
                stream=bool(body.get("stream")),
            )
        except (openai.OpenAIError, anthropic.AnthropicError) as error:
            raise model_error(error) from error
        return client, reply

    async def relay_aux(
        self,
        dialect: Dialect,
        route: str,
        body: dict,
        headers: Mapping[str, str] | None = None,
    ) -> dict:
        _, reply = await self._request(dialect, route, body, headers)
        try:
            return await reply.json()
        finally:
            await reply.close()

    async def close(self) -> None:
        await self.openai.close()
        await self.anthropic.close()
