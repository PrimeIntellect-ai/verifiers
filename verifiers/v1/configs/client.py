"""Client configs: describe an OpenAI-compatible endpoint.

A `BaseClientConfig` is an OpenAI-compatible endpoint (base_url + API-key env var
+ extra headers); `clients.resolve_client` turns one into a live `Client` — the
interception server builds one per distinct config and shares it across the rollouts
it multiplexes. The default Prime endpoint, API key, and team fall back to
the active Prime CLI config, so direct `uv run vf-eval` calls behave like `prime eval`.
Both the eval entrypoint (its model client) and in-env LLM calls (e.g. a judge reward)
build clients from these. `ClientConfig` is the CLI-selectable discriminated union
(eval | train).
"""

import os
from itertools import chain
from typing import Annotated, Literal
from urllib.parse import urlparse

from pydantic import Field, model_validator
from pydantic_config import BaseConfig
from renderers import RendererConfig

from verifiers.v1.utils.prime import load_prime_config

DEFAULT_PRIME_INFERENCE_URL = "https://api.pinference.ai/api/v1"

PRIME_INFERENCE_HOST = "pinference.ai"
PRIME_TEAM_ID_HEADER = "X-Prime-Team-ID"


class ClientTimeoutConfig(BaseConfig):
    """Timeouts (in seconds) for each phase of an HTTP request to the endpoint; `None`
    waits indefinitely. Distinct from the agent's stage timeouts, which bound a rollout."""

    connect: float | None = Field(30.0, gt=0)
    """Time to establish a connection, so an unreachable endpoint still fails fast."""
    read: float | None = Field(None, gt=0)
    """Time between received bytes. Unbounded by default: agentic completions are slow
    and the rollout timeout is the real backstop."""
    write: float | None = Field(None, gt=0)
    """Time between sent bytes."""
    pool: float | None = Field(None, gt=0)
    """Time to acquire a connection from the pool."""


class ProviderRetryConfig(BaseConfig):
    """How long a model call is retried in place when the provider fails it transiently:
    a 5xx, 408 or 429, a connection reset or timeout, or a truncated or malformed
    response. The interception server sends the call again, before anything of it is
    committed to the trace, with exponential backoff (or the provider's `Retry-After`),
    until it succeeds or `budget` seconds have passed since it first failed. Any other
    failure (a 4xx such as an invalid request or a context overflow) is final at once.
    The failure it gives up on reaches the harness marked not retryable."""

    budget: float = Field(900.0, ge=0)
    """Seconds since the call first failed in which it is retried; 0 retries nothing.
    An attempt in flight when it runs out is not cut short."""
    max_backoff: float = Field(60.0, gt=0)
    """Longest wait between attempts, `Retry-After` included."""


class RequestGateConfig(BaseConfig):
    """A per-model cap on concurrent requests to this endpoint, shared by every process on
    the host that uses the same endpoint, API key and model (see `clients.gate`). With
    `adaptive`, the limit drops on provider rejections (429/529) and climbs back to
    `max_concurrent` over `recover_seconds`; `max_concurrent` is never exceeded."""

    max_concurrent: int = Field(ge=1)
    """Most requests in flight at once, per model: the cap a gate starts with. The first
    process to use a gate sets its cap; processes joining a live gate share that cap, and
    an operator can move it (`clients.gate.resize`)."""
    scope: str | None = None
    """Optional deployment/quota identity within an endpoint origin and credential."""
    min_concurrent: int = Field(1, ge=1)
    """Lowest limit the adaptive gate shrinks to."""
    adaptive: bool = True
    """Shrink the limit on 429/529 and grow it back on success; off, the limit stays at
    `max_concurrent`."""
    decrease_factor: float = Field(0.75, gt=0, lt=1)
    """Factor applied to the limit on a rejection."""
    decrease_cooldown: float = Field(2.0, ge=0)
    """Seconds after a decrease in which further rejections do not decrease again: one
    burst of rejections is one signal."""
    recover_seconds: float = Field(60.0, gt=0)
    """Seconds in which a lowered limit climbs back to `max_concurrent`, in a straight
    line from the decrease."""
    max_rejections: int = Field(10, ge=0)
    """Rejections absorbed per request (each waits, then queues again). The next one is
    final: the interception server does not retry it (`ProviderRetryConfig`). 0 absorbs
    none and leaves 429s to those retries, as without a gate."""
    backoff: float = Field(1.0, gt=0)
    """Base wait in seconds after a rejection without `Retry-After`; doubles per rejection."""
    max_backoff: float = Field(60.0, gt=0)
    """Longest wait after a rejection, `Retry-After` included."""
    dir: str | None = None
    """Directory of the host's gate daemon (its socket, log and state snapshots). None =
    a host-wide directory under the system temp dir. Gates are per host: a directory on a
    shared mount does not share a budget across hosts."""

    @model_validator(mode="after")
    def check_bounds(self) -> "RequestGateConfig":
        if self.min_concurrent > self.max_concurrent:
            raise ValueError("min_concurrent exceeds max_concurrent")
        return self


class BaseClientConfig(BaseConfig):
    """An OpenAI-compatible endpoint. The API key is read from an env var."""

    base_url: str = DEFAULT_PRIME_INFERENCE_URL
    api_key_var: str = "PRIME_API_KEY"
    headers: dict[str, str] = Field(default_factory=dict)
    """Extra HTTP headers sent on every request."""
    headers_from_env: dict[str, str] = Field(default_factory=dict)
    """Header names mapped to environment variables, resolved when building a client.
    Set environment values override static headers."""
    timeout: ClientTimeoutConfig = ClientTimeoutConfig()
    retry: ProviderRetryConfig = ProviderRetryConfig()
    """In-place retries of a rollout's model calls the provider failed transiently."""
    request_gate: RequestGateConfig | None = None
    """Per-model concurrent-request gate for every request sent through this endpoint
    config (harness turns, nested agents, judges). None = no gate."""

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
        if (
            "base_url" not in self.model_fields_set
            and self.base_url == DEFAULT_PRIME_INFERENCE_URL
        ):
            self.base_url = prime_base_url
        return self


class EvalClientConfig(BaseClientConfig):
    """The default (eval): forward each request to a matching endpoint via `EvalClient`."""

    type: Literal["eval"] = "eval"


class TrainClientConfig(BaseClientConfig):
    """Training: a vLLM `/inference/v1/generate` endpoint with client-side tokenization (via
    `TrainClient`), so responses carry token IDs and logprobs. Needs a running vLLM engine."""

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


def resolve_headers(config: BaseClientConfig) -> dict[str, str]:
    """Resolve request headers for this destination without storing credentials in config."""
    # Keep original spelling for SDKs that merge default headers case-sensitively.
    names: dict[str, str] = {}
    headers: dict[str, str] = {}
    for name, value in chain(
        config.headers.items(),
        (
            (name, value)
            for name, variable in config.headers_from_env.items()
            if (value := os.environ.get(variable)) is not None
        ),
    ):
        headers[names.setdefault(name.lower(), name)] = value
    host = urlparse(config.base_url).hostname or ""
    if (
        config.api_key_var == "PRIME_API_KEY"
        and (host == PRIME_INFERENCE_HOST or host.endswith(f".{PRIME_INFERENCE_HOST}"))
        and PRIME_TEAM_ID_HEADER.lower() not in names
    ):
        team_id = os.environ.get("PRIME_TEAM_ID") or load_prime_config().get("team_id")
        if team_id:
            headers[PRIME_TEAM_ID_HEADER] = team_id
    return headers
