from verifiers.clients.anthropic_messages_client import AnthropicMessagesClient
from verifiers.clients.client import Client
from verifiers.clients.nemorl_chat_completions_client import (
    NeMoRLChatCompletionsClient,
)
from verifiers.clients.openai_chat_completions_client import OpenAIChatCompletionsClient
from verifiers.clients.openai_chat_completions_token_client import (
    OpenAIChatCompletionsTokenClient,
)
from verifiers.clients.openai_completions_client import OpenAICompletionsClient
from verifiers.clients.openai_responses_client import OpenAIResponsesClient
from verifiers.types import ClientConfig


def _load_renderer_client():
    try:
        from verifiers.clients.renderer_client import RendererClient
    except ModuleNotFoundError as e:
        missing = e.name or ""
        if missing == "renderers" or missing.startswith("renderers."):
            raise ImportError(
                "RendererClient requires the renderers extra; install "
                "`verifiers[renderers]`."
            ) from e
        raise

    return RendererClient


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
                RendererClient = _load_renderer_client()
                return RendererClient(client_or_config)
            case "anthropic_messages":
                return AnthropicMessagesClient(client_or_config)
            case "nemorl_chat_completions":
                return NeMoRLChatCompletionsClient(client_or_config)
    else:
        raise ValueError(f"Unsupported client type: {type(client_or_config)}")


def __getattr__(name: str):
    if name == "RendererClient":
        return _load_renderer_client()
    raise AttributeError(f"module 'verifiers.clients' has no attribute '{name}'")


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
