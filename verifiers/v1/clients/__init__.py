"""The client abstraction and its OpenAI-compatible + renderer implementations."""

from verifiers.v1.clients.client import Client, RetryingClient, RolloutContext
from verifiers.v1.clients.config import (
    BaseClientConfig,
    ClientConfig,
    EvalClientConfig,
    TrainClientConfig,
    resolve_client,
)
from verifiers.v1.clients.eval import EvalClient


def __getattr__(name: str):
    if name == "TrainClient":
        from verifiers.v1.clients.train import TrainClient

        return TrainClient
    raise AttributeError(name)


__all__ = [
    "Client",
    "RetryingClient",
    "RolloutContext",
    "BaseClientConfig",
    "ClientConfig",
    "EvalClientConfig",
    "TrainClientConfig",
    "resolve_client",
    "EvalClient",
    "TrainClient",
]
