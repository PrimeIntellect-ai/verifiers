"""Endpoint configuration shared by evaluation, training, and in-env model calls."""

import os
from urllib.parse import urlparse

from pydantic import Field, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.utils.prime import load_prime_config

DEFAULT_PRIME_INFERENCE_URL = "https://api.pinference.ai/api/v1"

PRIME_INFERENCE_HOST = "pinference.ai"
PRIME_TEAM_ID_HEADER = "X-Prime-Team-ID"


class ClientConfig(BaseConfig):
    """An OpenAI-compatible endpoint. The API key is read from an env var."""

    base_url: str = DEFAULT_PRIME_INFERENCE_URL
    api_key_var: str = "PRIME_API_KEY"
    headers: dict[str, str] = Field(default_factory=dict)
    """Extra HTTP headers sent on every request."""

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
