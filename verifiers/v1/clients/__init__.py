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
    if name != "TrainClient":
        raise AttributeError(name)
    try:
        from verifiers.v1.clients.train import TrainClient
    except ModuleNotFoundError as e:
        raise ImportError(
            "TrainClient requires the renderers extra; install `verifiers[renderers]`."
        ) from e
    return TrainClient


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
