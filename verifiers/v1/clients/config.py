"""Client configs: describe OpenAI-compatible endpoints and resolve them to a Client.

A `BaseClientConfig` is one or more OpenAI-compatible endpoints (base_url + API-key env var
+ extra headers) that `resolve_client` turns into a `Client`. Prime team-billing
is handled while resolving each endpoint. Both the eval entrypoint
(its model client) and in-env LLM calls (e.g. a judge reward) build clients from
these — inherit `BaseClientConfig` to get the endpoint/header handling for free.
`ClientConfig` is the CLI-selectable discriminated union (eval | train).
"""

import os
from typing import Annotated, Literal

from openai import AsyncOpenAI
from pydantic import Field
from pydantic_config import BaseConfig
from renderers import RendererConfig

from verifiers.v1.clients.client import Client
from verifiers.v1.clients.eval import EvalClient
from verifiers.v1.clients.train import TrainClient

PRIME_INFERENCE_HOST = "pinference.ai"
PRIME_TEAM_ID_HEADER = "X-Prime-Team-ID"


class BaseClientConfig(BaseConfig):
    """OpenAI-compatible endpoint(s). The API key is read from an env var."""

    base_url: str | Annotated[list[str], Field(min_length=1)] = (
        "https://api.pinference.ai/api/v1"
    )
    """One URL, or URLs assigned round-robin across rollouts."""
    api_key_var: str = "PRIME_API_KEY"
    headers: dict[str, str] = Field(default_factory=dict)
    """Extra HTTP headers sent on every request."""


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


def resolve_client(config: BaseClientConfig, endpoint_idx: int = 0) -> Client:
    """Resolve one configured endpoint."""
    base_urls = (
        [config.base_url] if isinstance(config.base_url, str) else config.base_url
    )
    base_url = base_urls[endpoint_idx % len(base_urls)]
    api_key = os.environ.get(config.api_key_var, "EMPTY")
    headers = dict(config.headers)
    team_id = os.environ.get("PRIME_TEAM_ID")
    if PRIME_INFERENCE_HOST in base_url and team_id:
        headers.setdefault(PRIME_TEAM_ID_HEADER, team_id)
    if isinstance(config, TrainClientConfig):
        # The renderer calls a vLLM `/inference/v1/generate` engine through the OpenAI SDK.
        openai = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            default_headers=headers or None,
        )
        return TrainClient(
            openai,
            pool_size=config.pool_size,
            config=config.renderer,
            renderer_model_name=config.renderer_model_name,
        )
    # The proxy is a raw httpx forwarder; the dialect supplies the auth scheme + upstream path.
    return EvalClient(base_url, api_key, headers=headers or None)
