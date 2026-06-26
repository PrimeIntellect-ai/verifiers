"""Client configs: describe an OpenAI-compatible endpoint and resolve it to a Client.

A `BaseClientConfig` is an OpenAI-compatible endpoint (base_url + API-key env var
+ extra headers) that `resolve_client` turns into a `Client`. Prime's wrapper
materializes its selected account as environment variables; direct Verifiers users
supply those variables explicitly. Both the eval entrypoint (its model client)
and in-env LLM calls (e.g. a judge reward) build clients from these.
`ClientConfig` is the CLI-selectable discriminated union (eval | train).
"""

import os
from typing import Annotated, Literal
from urllib.parse import urlparse

from openai import AsyncOpenAI
from pydantic import Field, model_validator
from pydantic_config import BaseConfig
from renderers import RendererConfig

from verifiers.v1.clients.client import Client
from verifiers.v1.clients.eval import EvalClient
from verifiers.v1.clients.train import TrainClient

DEFAULT_PRIME_INFERENCE_URL = "https://api.pinference.ai/api/v1"
PRIME_INFERENCE_HOST = "pinference.ai"
PRIME_TEAM_ID_HEADER = "X-Prime-Team-ID"


class BaseClientConfig(BaseConfig):
    """An OpenAI-compatible endpoint. The API key is read from an env var."""

    base_url: str = DEFAULT_PRIME_INFERENCE_URL
    api_key_var: str = "PRIME_API_KEY"
    headers: dict[str, str] = Field(default_factory=dict)
    """Extra HTTP headers sent on every request."""

    @model_validator(mode="after")
    def apply_prime_config(self) -> "BaseClientConfig":
        if self.api_key_var != "PRIME_API_KEY":
            return self
        prime_base_url = (
            os.environ.get("PRIME_INFERENCE_URL") or DEFAULT_PRIME_INFERENCE_URL
        )
        if "base_url" not in self.model_fields_set:
            self.base_url = prime_base_url
        host = urlparse(self.base_url).hostname or ""
        if host != PRIME_INFERENCE_HOST and not host.endswith(
            f".{PRIME_INFERENCE_HOST}"
        ):
            return self
        team_id = os.environ.get("PRIME_TEAM_ID")
        if team_id:
            self.headers.setdefault(PRIME_TEAM_ID_HEADER, team_id)
        return self


class EvalClientConfig(BaseClientConfig):
    """The default (eval): forward each request to a matching endpoint via `EvalClient`."""

    type: Literal["eval"] = "eval"


class TrainClientConfig(BaseClientConfig):
    """Training: a vLLM `/inference/v1/generate` endpoint with client-side tokenization (via
    `TrainClient`), so responses carry token ids + logprobs. Needs a running vLLM engine."""

    type: Literal["train"] = "train"
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


# Discriminated union for a CLI-selectable client (`--client.type eval|train`).
ClientConfig = Annotated[
    EvalClientConfig | TrainClientConfig, Field(discriminator="type")
]


def resolve_client(config: BaseClientConfig) -> Client:
    api_key = os.environ.get(config.api_key_var)
    api_key = api_key or "EMPTY"
    if isinstance(config, TrainClientConfig):
        # The renderer calls a vLLM `/inference/v1/generate` engine through the OpenAI SDK.
        openai = AsyncOpenAI(
            base_url=config.base_url,
            api_key=api_key,
            default_headers=config.headers or None,
        )
        return TrainClient(
            openai,
            pool_size=config.pool_size,
            config=config.renderer,
            renderer_model_name=config.renderer_model_name,
        )
    # The proxy is a raw httpx forwarder; the dialect supplies the auth scheme + upstream path.
    return EvalClient(config.base_url, api_key, headers=config.headers or None)
