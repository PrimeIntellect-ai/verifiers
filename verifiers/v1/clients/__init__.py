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

try:
    from verifiers.v1.clients.train import TrainClient
except ImportError:

    class TrainClient:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            raise ImportError(
                "install renderers or run inside prime-rl to use TrainClient"
            )


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
