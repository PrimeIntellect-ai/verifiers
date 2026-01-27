from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from verifiers.clients.anthropic import (
    AnthropicCompletionsClient,
    AnthropicMessagesClient,
)
from verifiers.clients.client import Client
from verifiers.clients.openai import (
    OAIChatCompletionsClient,
    OAIChatCompletionsTokenClient,
    OAICompletionsClient,
)
from verifiers.types import MessageType

CLIENT_REGISTRY = {
    "openai": {
        "completions": OAICompletionsClient,
        "chat_completions": OAIChatCompletionsClient,
        "chat_completions_tokens": OAIChatCompletionsTokenClient,
    },
    "anthropic": {
        "messages": AnthropicMessagesClient,
        "completions": AnthropicCompletionsClient,
    },
}


def resolve_client(
    client: Client | AsyncOpenAI | AsyncAnthropic,
    message_type: MessageType,
    interleaved_rollouts: bool,
) -> Client:
    if isinstance(client, Client):
        return client
    if interleaved_rollouts:
        assert isinstance(client, AsyncOpenAI), (
            "Interleaved rollouts are only supported for OpenAI clients"
        )
        return OAIChatCompletionsTokenClient(client)
    if isinstance(client, AsyncOpenAI):
        if message_type == "chat":
            return OAIChatCompletionsClient(client)
        elif message_type == "completion":
            return OAICompletionsClient(client)
        else:
            raise ValueError(
                f"Unsupported message type: {message_type} for OpenAI client"
            )
    elif isinstance(client, AsyncAnthropic):
        if message_type == "chat":
            return AnthropicMessagesClient(client)
        elif message_type == "completion":
            return AnthropicCompletionsClient(client)
        else:
            raise ValueError(
                f"Unsupported message type: {message_type} for Anthropic client"
            )
    else:
        raise ValueError(f"Unsupported client type: {type(client)}")


__all__ = [
    "AnthropicMessagesClient",
    "AnthropicCompletionsClient",
    "OAICompletionsClient",
    "OAIChatCompletionsClient",
    "OAIChatCompletionsTokenClient",
    "Client",
]
