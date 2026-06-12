"""The client abstraction: turn a prompt into a `Response`.

Collapsed from v1's 4-typevar generic ABC with five conversion hooks to a single
abstract method. Each concrete client owns its own wire translation internally.

A client whose endpoint natively speaks a registered dialect also declares it
(`dialect`) and supports `relay`: forwarding a program's request *bytes* verbatim to
its endpoint, the interception server's no-translation fast path. Clients without a
matching ingress dialect (renderer, google) leave `dialect = None` and are only ever
reached through the typed `get_response`.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import ClassVar

import httpx
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
)

from verifiers.v1.errors import ModelError, OverlongPromptError, classify_model_error
from verifiers.v1.types import Messages, Response, SamplingConfig, Tool

logger = logging.getLogger(__name__)


@dataclass
class RelayReply:
    """A relayed upstream response: its content type and body chunks (one chunk for a
    JSON body, many for SSE). The connection closes when `chunks` is exhausted."""

    content_type: str
    chunks: AsyncIterator[bytes]


async def relay_post(
    http: httpx.AsyncClient, url: str, headers: Mapping[str, str], body: bytes
) -> RelayReply:
    """POST `body` verbatim and stream the response back. An error status is read fully
    and raised as a `ModelError` (an overlong prompt as `OverlongPromptError`), so the
    retry and truncation machinery treat relayed calls exactly like typed ones."""
    # Lowercase keys so e.g. an SDK "Content-Type" and ours can't both go on the wire
    # (duplicate content-type headers make providers reject the body).
    normalized = {k.lower(): v for k, v in headers.items()}
    normalized.setdefault("content-type", "application/json")
    request = http.build_request("POST", url, content=body, headers=normalized)
    response = await http.send(request, stream=True)
    if response.status_code >= 400:
        text = (await response.aread()).decode("utf-8", errors="replace")
        await response.aclose()
        raise classify_model_error(f"upstream {response.status_code}: {text}")

    async def chunks() -> AsyncIterator[bytes]:
        try:
            async for chunk in response.aiter_bytes():
                yield chunk
        finally:
            await response.aclose()

    return RelayReply(
        content_type=response.headers.get("content-type", "application/json"),
        chunks=chunks(),
    )


def relay_headers(sdk) -> dict[str, str]:
    """An SDK client's headers (defaults + auth, minus its Omit sentinels) — what a
    byte relay sends upstream. `auth_headers` is merged explicitly because newer SDK
    versions no longer fold auth into `default_headers`."""
    merged: Mapping[str, object] = {**sdk.default_headers, **sdk.auth_headers}
    return {k: v for k, v in merged.items() if isinstance(v, str)}


class Client(ABC):
    dialect: ClassVar[str | None] = None
    """The registered ingress dialect this client's endpoint natively speaks (see
    `verifiers.v1.dialects`). When a program's request arrives in this dialect, the
    interception server relays its bytes via `relay` instead of translating."""

    @abstractmethod
    async def get_response(
        self,
        prompt: Messages,
        model: str,
        sampling_args: SamplingConfig,
        tools: list[Tool] | None = None,
    ) -> Response:
        """Run one completion, translating to/from this client's wire format."""

    async def relay(self, body: bytes, route: str) -> RelayReply:
        """Forward a program's request bytes verbatim to this client's endpoint.
        `route` is the ingress route the request arrived on (e.g. `/v1/messages`);
        the client maps it onto its own base URL."""
        raise NotImplementedError(f"{type(self).__name__} does not relay")

    async def close(self) -> None:
        """Release any underlying resources. Default no-op."""


class RetryingClient(Client):
    """Wraps a client to retry each completion on a transient `ModelError` (tenacity, up to
    `max_retries` retries). An `OverlongPromptError` is never retried — it's a budget limit
    the interception server turns into a clean truncation, not a transient fault."""

    def __init__(self, inner: Client, max_retries: int) -> None:
        self.inner = inner
        self.max_retries = max_retries
        # One Retrying, reused across (and concurrent within) calls: the control flow runs
        # off a per-call RetryCallState, so only its bookkeeping `.statistics` is shared.
        self._retrying = AsyncRetrying(
            stop=stop_after_attempt(max_retries + 1),
            retry=retry_if_exception_type(ModelError)
            & retry_if_not_exception_type(OverlongPromptError),
            before_sleep=self._log_retry,
            reraise=True,
        )

    @property
    def dialect(self) -> str | None:  # type: ignore[override]
        return self.inner.dialect

    def _log_retry(self, state: RetryCallState) -> None:
        logger.warning(
            "retrying model call (attempt %d/%d) after error: %s",
            state.attempt_number,
            self.max_retries + 1,
            state.outcome.exception(),
        )

    async def get_response(
        self,
        prompt: Messages,
        model: str,
        sampling_args: SamplingConfig,
        tools: list[Tool] | None = None,
    ) -> Response:
        return await self._retrying(
            self.inner.get_response, prompt, model, sampling_args, tools
        )

    async def relay(self, body: bytes, route: str) -> RelayReply:
        # Safe to retry: a relay raises (and is retried) before any response byte has
        # been handed back; once a `RelayReply` is returned, streaming is underway.
        return await self._retrying(self.inner.relay, body, route)

    async def close(self) -> None:
        await self.inner.close()


@dataclass(frozen=True)
class RolloutContext:
    """The collaborators a single rollout needs (client + model + sampling), bundled
    so harnesses hold no rollout state. Built by the Environment."""

    client: Client
    model: str
    sampling: SamplingConfig
