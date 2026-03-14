from __future__ import annotations

import importlib
from verifiers.clients.client import Client
from typing import TYPE_CHECKING

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
                from verifiers.clients.openai_completions_client import (
                    OpenAICompletionsClient,
                )

                return OpenAICompletionsClient(client_or_config)
            case "openai_chat_completions":
                from verifiers.clients.openai_chat_completions_client import (
                    OpenAIChatCompletionsClient,
                )

                return OpenAIChatCompletionsClient(client_or_config)
            case "openai_chat_completions_token":
                from verifiers.clients.openai_chat_completions_token_client import (
                    OpenAIChatCompletionsTokenClient,
                )

                return OpenAIChatCompletionsTokenClient(client_or_config)
            case "anthropic_messages":
                from verifiers.clients.anthropic_messages_client import (
                    AnthropicMessagesClient,
                )

                return AnthropicMessagesClient(client_or_config)
    else:
        raise ValueError(f"Unsupported client type: {type(client_or_config)}")


__all__ = [
    "AnthropicMessagesClient",
    "OpenAICompletionsClient",
    "OpenAIChatCompletionsClient",
    "OpenAIChatCompletionsTokenClient",
    "Client",
]

_LAZY_IMPORTS = {
    "AnthropicMessagesClient": (
        "verifiers.clients.anthropic_messages_client:AnthropicMessagesClient"
    ),
    "OpenAICompletionsClient": (
        "verifiers.clients.openai_completions_client:OpenAICompletionsClient"
    ),
    "OpenAIChatCompletionsClient": (
        "verifiers.clients.openai_chat_completions_client:OpenAIChatCompletionsClient"
    ),
    "OpenAIChatCompletionsTokenClient": (
        "verifiers.clients.openai_chat_completions_token_client:OpenAIChatCompletionsTokenClient"
    ),
}


def __getattr__(name: str):
    try:
        module_path, attr = _LAZY_IMPORTS[name].split(":")
    except KeyError as exc:
        raise AttributeError(
            f"module 'verifiers.clients' has no attribute '{name}'"
        ) from exc

    module = importlib.import_module(module_path)
    return getattr(module, attr)


if TYPE_CHECKING:
    from verifiers.clients.anthropic_messages_client import AnthropicMessagesClient
    from verifiers.clients.openai_chat_completions_client import (
        OpenAIChatCompletionsClient,
    )
    from verifiers.clients.openai_chat_completions_token_client import (
        OpenAIChatCompletionsTokenClient,
    )
    from verifiers.clients.openai_completions_client import OpenAICompletionsClient
