"""Client configs: describe a model endpoint and resolve it to a Client.

A `BaseClientConfig` is an endpoint (base_url + API-key env var + extra headers)
that `resolve_client` turns into a `Client`. Prime team-billing is baked in via a
validator, so it's handled in one place. Both the eval entrypoint (its model client)
and in-env LLM calls (e.g. a judge reward) build clients from these.
"""

import os
from typing import Annotated, Literal

from anthropic import AsyncAnthropic
from google import genai
from google.genai import types as google_types
from openai import AsyncOpenAI
from pydantic import Field, model_validator
from pydantic_config import BaseConfig
from renderers import RendererConfig

from verifiers.v1.clients.anthropic import AnthropicMessagesClient
from verifiers.v1.clients.client import Client
from verifiers.v1.clients.google import GoogleResponsesClient
from verifiers.v1.clients.openai import OpenAIChatCompletionsClient
from verifiers.v1.clients.openai_responses import OpenAIResponsesClient
from verifiers.v1.clients.renderer import RendererClient

PRIME_INFERENCE_HOST = "pinference.ai"
PRIME_TEAM_ID_HEADER = "X-Prime-Team-ID"


class BaseClientConfig(BaseConfig):
    """A model endpoint. The API key is read from an env var."""

    base_url: str = "https://api.pinference.ai/api/v1"
    api_key_var: str = "PRIME_API_KEY"
    headers: dict[str, str] = Field(default_factory=dict)
    """Extra HTTP headers sent on every request."""

    @model_validator(mode="after")
    def add_prime_team_id(self) -> "BaseClientConfig":
        # Prime inference bills the personal balance unless a team is named; on
        # that endpoint, route billing to PRIME_TEAM_ID when set (explicit wins).
        team_id = os.environ.get("PRIME_TEAM_ID")
        if PRIME_INFERENCE_HOST in self.base_url and team_id:
            self.headers.setdefault(PRIME_TEAM_ID_HEADER, team_id)
        return self


class OpenAIClientConfig(BaseClientConfig):
    """The default: an OpenAI-compatible chat-completions endpoint (text in/out)."""

    type: Literal["openai"] = "openai"


class OpenAIResponsesClientConfig(BaseClientConfig):
    """An OpenAI-compatible Responses API endpoint."""

    type: Literal["openai_responses"] = "openai_responses"


class AnthropicMessagesClientConfig(BaseClientConfig):
    """The Anthropic Messages API."""

    type: Literal["anthropic_messages"] = "anthropic_messages"
    base_url: str = "https://api.anthropic.com"
    api_key_var: str = "ANTHROPIC_API_KEY"


class GoogleResponsesClientConfig(BaseClientConfig):
    """The Google Gemini generateContent API."""

    type: Literal["google_responses"] = "google_responses"
    base_url: str = "https://generativelanguage.googleapis.com/"
    api_key_var: str = "GEMINI_API_KEY"


class RendererClientConfig(BaseClientConfig):
    """A vLLM `/inference/v1/generate` endpoint with client-side tokenization, so
    responses carry token ids + logprobs. Needs a running vLLM engine."""

    type: Literal["renderers"] = "renderers"
    renderer: RendererConfig | None = None
    """The `renderers.RendererConfig` to use (the same shared type prime-rl configures).
    `None` auto-resolves from the model — which falls back to the default renderer (no
    tool support) for models not in the renderer map, so set it explicitly for
    fine-tunes / tool-using envs."""
    pool_size: int = 1
    """Renderer slots shared across concurrent rollouts (client-side tokenization)."""
    renderer_model_name: str | None = None
    """Model the tokenizer/renderer pool is built for. Pin to the base model so a LoRA
    adapter name (served only for sampling) never drives tokenizer loading. Falls back to
    the per-request model when None."""


# Discriminated union for a CLI-selectable client (`--client.type renderers`).
ClientConfig = Annotated[
    OpenAIClientConfig
    | OpenAIResponsesClientConfig
    | AnthropicMessagesClientConfig
    | GoogleResponsesClientConfig
    | RendererClientConfig,
    Field(discriminator="type"),
]


def resolve_client(config: BaseClientConfig) -> Client:
    def make_openai_client(config: BaseClientConfig) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=config.base_url,
            api_key=os.environ.get(config.api_key_var, "EMPTY"),
            default_headers=config.headers or None,
        )

    if isinstance(config, RendererClientConfig):
        return RendererClient(
            make_openai_client(config),
            pool_size=config.pool_size,
            config=config.renderer,
            renderer_model_name=config.renderer_model_name,
        )
    if isinstance(config, OpenAIResponsesClientConfig):
        return OpenAIResponsesClient(make_openai_client(config))
    if isinstance(config, AnthropicMessagesClientConfig):
        return AnthropicMessagesClient(
            AsyncAnthropic(
                base_url=config.base_url,
                api_key=os.environ.get(config.api_key_var, "EMPTY"),
                default_headers=config.headers or None,
            )
        )
    if isinstance(config, GoogleResponsesClientConfig):
        return GoogleResponsesClient(
            genai.Client(
                api_key=os.environ.get(config.api_key_var, "EMPTY"),
                http_options=google_types.HttpOptions(
                    base_url=config.base_url,
                    api_version="v1beta",
                    headers=config.headers or None,
                ),
            )
        )
    return OpenAIChatCompletionsClient(make_openai_client(config))
