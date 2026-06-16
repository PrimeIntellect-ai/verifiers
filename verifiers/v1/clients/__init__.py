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
        raise AttributeError(f"module 'verifiers.v1.clients' has no attribute '{name}'")
    try:
        from verifiers.v1.clients.train import TrainClient
    except ModuleNotFoundError as exc:
        if exc.name != "renderers" and not (exc.name or "").startswith("renderers."):
            raise
        raise ImportError(
            "TrainClient requires the renderers extra; install `verifiers[renderers]`."
        ) from exc
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
