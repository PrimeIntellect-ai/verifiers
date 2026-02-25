from __future__ import annotations

from verifiers.clients.anthropic_messages_client import AnthropicMessagesClient
from verifiers.clients.client import Client
from verifiers.clients.human_cli_client import HumanCLIClient
from verifiers.clients.openai_chat_completions_client import OpenAIChatCompletionsClient
from verifiers.clients.openai_chat_completions_token_client import (
    OpenAIChatCompletionsTokenClient,
)
from verifiers.clients.openai_completions_client import OpenAICompletionsClient
from verifiers.types import ClientConfig


def resolve_client(client_or_config: Client | ClientConfig) -> Client:
    """Resolves a client or client config to a client."""
    if isinstance(client_or_config, Client):
        client = client_or_config
        return client
    elif isinstance(client_or_config, ClientConfig):
        client_type = client_or_config.client_type
        match client_type:
            case "openai_completions":
                return OpenAICompletionsClient(client_or_config)
            case "openai_chat_completions":
                return OpenAIChatCompletionsClient(client_or_config)
            case "openai_chat_completions_token":
                return OpenAIChatCompletionsTokenClient(client_or_config)
            case "anthropic_messages":
                return AnthropicMessagesClient(client_or_config)
    else:
        raise ValueError(f"Unsupported client type: {type(client_or_config)}")


__all__ = [
    "AnthropicMessagesClient",
    "OpenAICompletionsClient",
    "OpenAIChatCompletionsClient",
    "OpenAIChatCompletionsTokenClient",
    "HumanCLIClient",
    "Client",
]
