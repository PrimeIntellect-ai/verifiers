"""The client abstraction and its OpenAI-compatible + renderer implementations."""

from verifiers.v1.clients.client import Client, RolloutContext
from verifiers.v1.clients.config import (
    BaseClientConfig,
    ClientConfig,
    EvalClientConfig,
    ModelEndpointConfig,
    TrainClientConfig,
    resolve_client,
)
from verifiers.v1.clients.eval import EvalClient
from verifiers.v1.clients.train import TrainClient

__all__ = [
    "Client",
    "RolloutContext",
    "BaseClientConfig",
    "ClientConfig",
    "EvalClientConfig",
    "ModelEndpointConfig",
    "TrainClientConfig",
    "resolve_client",
    "EvalClient",
    "TrainClient",
]
