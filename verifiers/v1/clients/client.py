"""Client interfaces for model inference and relay."""

import logging
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field

import httpx
from openai import AsyncOpenAI

from verifiers.v1.configs.client import (
    BaseClientConfig,
    ClientConfig,
    TrainClientConfig,
    resolve_api_key,
)
from verifiers.v1.dialects import Dialect
from verifiers.v1.graph import PendingTurn
from verifiers.v1.types import Response, Sampling, SamplingConfig

logger = logging.getLogger(__name__)

# Transport settings shared by every client, mirroring the OpenAI SDK's own defaults so a
# rollout behaves the same whether its turns are relayed (eval) or rendered (train) — and
# the same as the SDK the harness itself is using on the other side of the interception.
DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=600.0, write=600.0, pool=600.0)
DEFAULT_LIMITS = httpx.Limits(max_connections=1000, max_keepalive_connections=100)
MAX_RETRIES = 0
"""No client-side retries: a failed call surfaces to the harness SDK and the trace instead of
being silently reattempted, so the framework's retry surfaces stay the only ones."""


def build_async_openai(config: BaseClientConfig) -> AsyncOpenAI:
    """An `AsyncOpenAI` for `config` (resolved key + extra headers) — for in-env model calls
    (e.g. a judge) and the training client's engine connection."""
    return AsyncOpenAI(
        base_url=config.base_url,
        api_key=resolve_api_key(config),
        default_headers=config.headers or None,
        timeout=DEFAULT_TIMEOUT,
        max_retries=MAX_RETRIES,
        http_client=httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, limits=DEFAULT_LIMITS),
    )


# An API version path segment (`v1`, `v2`, ...) — the only kind `join_url` dedups.
VERSION_SEGMENT = re.compile(r"v\d+")


def join_url(base_url: str, path: str) -> str:
    """Join `base_url` with a dialect path without repeating the API version segment:
    `.../api/v1` + `/v1/messages` -> `.../api/v1/messages`. Only version-shaped segments
    dedup, so a base ending in `/chat` doesn't swallow `/chat/completions`."""
    head = path.split("/")[1] if path.startswith("/") else ""
    base = base_url.rstrip("/")
    if VERSION_SEGMENT.fullmatch(head) and base.endswith(f"/{head}"):
        base = base[: -len(head) - 1]
    return base + path


SESSION_ID_HEADER = "X-Session-ID"
"""Per-rollout routing header. Every turn of one rollout sends the same value (the trace id),
so a session-affinity router (e.g. vLLM's ``consistent_hash`` policy keyed on its
``request_id_headers``) pins all of a rollout's turns to the same engine — keeping the
growing cross-turn prefix warm in that engine's KV cache instead of re-prefilling it
cold on a random shard each turn."""


@dataclass
class RelayReply:
    """A relayed upstream response: content type, complete SSE events, and connection cleanup."""

    content_type: str
    chunks: AsyncIterator[bytes]
    close: Callable[[], Awaitable[None]]


class Client(ABC):
    @abstractmethod
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
        """Run one completion -> a vf `Response`. The eval client forwards the native JSON and
        eligible end-to-end headers, then parses a copy via `dialect`; the train client derives
        the typed prompt from `body` and tokenizes it.

        `session_id` is the rollout's stable id (the trace id); when set, the client sends it
        as the `SESSION_ID_HEADER` so a session-affinity router keeps the rollout's turns on
        one engine for cross-turn prefix-cache reuse. `turn` is the graph-resolved prompt
        prefix; train clients may use it for renderer bridging, while relay clients ignore it."""

    async def relay(
        self,
        dialect: Dialect,
        body: dict,
        model: str,
        sampling_args: SamplingConfig,
        session_id: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> RelayReply:
        """Stream a (possibly SSE) response back, relaying the provider's bytes — the proxy's
        path for a streaming request. Only the relay (eval) client supports it; the renderer
        generates and cannot stream."""
        raise NotImplementedError(f"{type(self).__name__} does not support streaming")

    async def relay_aux(
        self,
        dialect: Dialect,
        route: str,
        body: dict,
        headers: Mapping[str, str] | None = None,
    ) -> dict:
        """Relay a non-model-turn side request (an `aux_route`, e.g. Anthropic's `count_tokens`)
        as native JSON and return the provider JSON. Only the relay (eval) client supports it."""
        raise NotImplementedError(f"{type(self).__name__} does not relay aux routes")

    async def close(self) -> None:
        pass


def resolve_client(config: BaseClientConfig) -> Client:
    if isinstance(config, TrainClientConfig):
        from verifiers.v1.clients.train import TrainClient

        return TrainClient(config)
    from verifiers.v1.clients.eval import EvalClient

    return EvalClient(config)


@dataclass(frozen=True)
class ModelContext:
    model: str
    client: ClientConfig
    sampling: Sampling = field(default_factory=Sampling)
