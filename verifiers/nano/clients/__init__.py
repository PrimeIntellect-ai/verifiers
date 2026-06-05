"""The client abstraction and its OpenAI-compatible implementation."""

from verifiers.nano.clients.client import Client
from verifiers.nano.clients.openai import OpenAIChatCompletionsClient

__all__ = ["Client", "OpenAIChatCompletionsClient"]
