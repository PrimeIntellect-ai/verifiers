"""Endpoint configuration shared by evaluation, training, and in-env model calls."""

import os
from itertools import chain
from urllib.parse import urlparse

from pydantic import Field, model_validator
from pydantic_config import BaseConfig

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


class ClientConfig(BaseConfig):
    """An OpenAI-compatible endpoint. The API key is read from an env var."""

    base_url: str = DEFAULT_PRIME_INFERENCE_URL
    api_key_var: str = "PRIME_API_KEY"
    headers: dict[str, str] = Field(default_factory=dict)
    """Extra HTTP headers sent on every request."""
    headers_from_env: dict[str, str] = Field(default_factory=dict)
    """Header names mapped to environment variables, resolved when building a client.
    Set environment values override static headers."""
    timeout: ClientTimeoutConfig = ClientTimeoutConfig()

    @model_validator(mode="after")
    def apply_prime_config(self) -> "ClientConfig":
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


def resolve_api_key(config: ClientConfig) -> str:
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


def resolve_headers(config: ClientConfig) -> dict[str, str]:
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
