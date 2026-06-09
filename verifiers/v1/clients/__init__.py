"""The client abstraction and its OpenAI-compatible + renderer implementations."""

from verifiers.v1.clients.client import Client, RolloutContext
from verifiers.v1.clients.config import (
    BaseClientConfig,
    ClientConfig,
    OpenAIClientConfig,
    RendererClientConfig,
    resolve_client,
)
from verifiers.v1.clients.openai import OpenAIChatCompletionsClient
from verifiers.v1.clients.renderer import RendererClient

__all__ = [
    "Client",
    "RolloutContext",
    "BaseClientConfig",
    "ClientConfig",
    "OpenAIClientConfig",
    "RendererClientConfig",
    "resolve_client",
    "OpenAIChatCompletionsClient",
    "RendererClient",
]
