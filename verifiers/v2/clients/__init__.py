"""The client abstraction and its OpenAI-compatible implementation."""

from verifiers.v2.clients.client import Client
from verifiers.v2.clients.openai import OpenAIChatCompletionsClient

__all__ = ["Client", "OpenAIChatCompletionsClient"]
