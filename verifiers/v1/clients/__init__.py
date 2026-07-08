"""The client abstraction and its OpenAI-compatible + renderer implementations."""

from verifiers.v1.clients.client import Client, ModelContext
from verifiers.v1.clients.config import (
    BaseClientConfig,
    ClientConfig,
    EvalClientConfig,
    TrainClientConfig,
    resolve_client,
)
from verifiers.v1.clients.eval import EvalClient
from verifiers.v1.clients.train import TrainClient

__all__ = [
    "Client",
    "ModelContext",
    "BaseClientConfig",
    "ClientConfig",
    "EvalClientConfig",
    "TrainClientConfig",
    "resolve_client",
    "EvalClient",
    "TrainClient",
]
