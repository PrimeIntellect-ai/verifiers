"""Client configs: describe an OpenAI-compatible endpoint.

A `BaseClientConfig` is an OpenAI-compatible endpoint (base_url + API-key env var
+ extra headers); `clients.resolve_client` turns one into a live `Client` — the
interception server builds one per distinct config and shares it across the rollouts
it multiplexes. The default Prime endpoint, API key, and team fall back to
the active Prime CLI config. Both an eval's model client and in-env LLM calls (e.g. a
judge reward) build clients from these. `ClientConfig` is the CLI-selectable discriminated union
(eval | train).
"""

import os
from typing import Annotated, Literal
from urllib.parse import urlparse

from pydantic import Field, model_validator
from pydantic_config import BaseConfig
from renderers import RendererConfig

from verifiers.v1.utils.prime import load_prime_config

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
        prime_config = load_prime_config()
        prime_base_url = (
            os.environ.get("PRIME_INFERENCE_URL")
            or prime_config.get("inference_url")
            or DEFAULT_PRIME_INFERENCE_URL
        )
        if "base_url" not in self.model_fields_set:
            self.base_url = prime_base_url
        host = urlparse(self.base_url).hostname or ""
        if host != PRIME_INFERENCE_HOST and not host.endswith(
            f".{PRIME_INFERENCE_HOST}"
        ):
            return self
        team_id = os.environ.get("PRIME_TEAM_ID") or prime_config.get("team_id")
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
    renderer_model_name: str | None = None
    """Model the tokenizer/renderer pool is built for. Pin to the base model so a LoRA
    adapter name (served only for sampling) never drives tokenizer loading. Falls back to
    the per-request model when None."""
    multiplex: int = Field(256, ge=1)
    """Rollouts that share one renderer (~75-95 MB each): the pool warms one and grows on
    demand, so N concurrent rollouts hold ~N/multiplex tokenizers. A renderer is only busy
    for the render itself (ms against a multi-second turn), so one absorbs many rollouts;
    lower this when rendering is the slow part (very long prompts, frequent bridge misses)."""


# Discriminated union for a CLI-selectable client (`--client.type eval|train`).
ClientConfig = Annotated[
    EvalClientConfig | TrainClientConfig, Field(discriminator="type")
]


def resolve_api_key(config: BaseClientConfig) -> str:
    """The API key for `config`: its env var, falling back to the Prime CLI config for a
    `PRIME_API_KEY`-keyed pinference endpoint. `"EMPTY"` when unset."""
    api_key = os.environ.get(config.api_key_var)
    host = urlparse(config.base_url).hostname or ""
    if (
        not api_key
        and config.api_key_var == "PRIME_API_KEY"
        and (host == PRIME_INFERENCE_HOST or host.endswith(f".{PRIME_INFERENCE_HOST}"))
    ):
        api_key = load_prime_config().get("api_key")
    return api_key or "EMPTY"
