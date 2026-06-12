"""The client abstraction and built-in provider implementations."""

from verifiers.v1.clients.anthropic import AnthropicMessagesClient
from verifiers.v1.clients.client import Client, RetryingClient, RolloutContext
from verifiers.v1.clients.config import (
    AnthropicMessagesClientConfig,
    BaseClientConfig,
    ClientConfig,
    GoogleResponsesClientConfig,
    OpenAIClientConfig,
    OpenAIResponsesClientConfig,
    RendererClientConfig,
    resolve_client,
)
from verifiers.v1.clients.google import GoogleResponsesClient
from verifiers.v1.clients.openai import OpenAIChatCompletionsClient
from verifiers.v1.clients.openai_responses import OpenAIResponsesClient
from verifiers.v1.clients.renderer import RendererClient

__all__ = [
    "Client",
    "RetryingClient",
    "RolloutContext",
    "BaseClientConfig",
    "ClientConfig",
    "OpenAIClientConfig",
    "OpenAIResponsesClientConfig",
    "AnthropicMessagesClientConfig",
    "GoogleResponsesClientConfig",
    "RendererClientConfig",
    "resolve_client",
    "OpenAIChatCompletionsClient",
    "OpenAIResponsesClient",
    "AnthropicMessagesClient",
    "GoogleResponsesClient",
    "RendererClient",
]
