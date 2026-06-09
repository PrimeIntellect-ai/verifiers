"""Client configs: describe an OpenAI-compatible endpoint and resolve it to a Client.

A `BaseClientConfig` is an OpenAI-compatible endpoint (base_url + API-key env var
+ extra headers) that `resolve_client` turns into a `Client`. Prime team-billing
is baked in via a validator, so it's handled in one place. Both the eval entrypoint
(its model client) and in-env LLM calls (e.g. a judge reward) build clients from
these — inherit `BaseClientConfig` to get the endpoint/header handling for free.
`ClientConfig` is the CLI-selectable discriminated union (openai | renderer).
"""

import os
from typing import Annotated, Literal

from openai import AsyncOpenAI
from pydantic import Field, model_validator
from pydantic_config import BaseConfig
from renderers import RendererConfig

from verifiers.v1.clients.client import Client
from verifiers.v1.clients.openai import OpenAIChatCompletionsClient
from verifiers.v1.clients.renderer import RendererClient

PRIME_INFERENCE_HOST = "pinference.ai"
PRIME_TEAM_ID_HEADER = "X-Prime-Team-ID"


class BaseClientConfig(BaseConfig):
    """An OpenAI-compatible endpoint. The API key is read from an env var."""

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


# Discriminated union for a CLI-selectable client (`--client.type renderers`).
ClientConfig = Annotated[
    OpenAIClientConfig | RendererClientConfig, Field(discriminator="type")
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
        )
    return OpenAIChatCompletionsClient(make_openai_client(config))
