# ruff: noqa

from verifiers.legacy.clients.anthropic_messages_client import AnthropicMessagesClient
from verifiers.legacy.clients.client import Client
from verifiers.legacy.clients.nemorl_chat_completions_client import (
    NeMoRLChatCompletionsClient,
)
from verifiers.legacy.clients.openai_chat_completions_client import (
    OpenAIChatCompletionsClient,
)
from verifiers.legacy.clients.openai_chat_completions_token_client import (
    OpenAIChatCompletionsTokenClient,
)
from verifiers.legacy.clients.openai_completions_client import OpenAICompletionsClient
from verifiers.legacy.clients.openai_responses_client import OpenAIResponsesClient
from verifiers.legacy.types import ClientConfig


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
            case "openai_responses":
                return OpenAIResponsesClient(client_or_config)
            case "renderer":
                from verifiers.legacy.clients.renderer_client import RendererClient

                return RendererClient(client_or_config)
            case "anthropic_messages":
                return AnthropicMessagesClient(client_or_config)
            case "nemorl_chat_completions":
                return NeMoRLChatCompletionsClient(client_or_config)
    else:
        raise ValueError(f"Unsupported client type: {type(client_or_config)}")


def __getattr__(name: str):
    if name == "RendererClient":
        from verifiers.legacy.clients.renderer_client import RendererClient

        return RendererClient
    raise AttributeError(f"module 'verifiers.legacy.clients' has no attribute '{name}'")


__all__ = [
    "AnthropicMessagesClient",
    "NeMoRLChatCompletionsClient",
    "OpenAICompletionsClient",
    "OpenAIChatCompletionsClient",
    "OpenAIChatCompletionsTokenClient",
    "OpenAIResponsesClient",
    "RendererClient",
    "Client",
]
