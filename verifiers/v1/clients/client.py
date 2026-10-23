"""Client interfaces for model inference and relay."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field

from verifiers.v1.configs.client import (
    BaseClientConfig,
    ClientConfig,
    TrainClientConfig,
)
from verifiers.v1.dialects import Dialect
from verifiers.v1.graph import PendingTurn
from verifiers.v1.types import Response, Sampling, SamplingConfig


class Client(ABC):
    @abstractmethod
    async def _complete(
        self,
        dialect: Dialect,
        body: dict,
        sampling: SamplingConfig,
        session_id: str | None = None,
        turn: PendingTurn | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> tuple[Response, bytes | None]:
        """Complete an effective native request, retaining provider SSE for atomic replay.

        Eval forwards the request; training renders its graph-resolved prompt to tokens.
        The response always carries native JSON; generated responses have no provider SSE.
        """

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
    """Model, endpoint config, and sampling for one rollout."""

    model: str
    client: ClientConfig
    sampling: Sampling = field(default_factory=Sampling)
