import json
import logging
import os
from pathlib import Path

import httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from verifiers.types import ClientConfig

logger = logging.getLogger(__name__)


def _resolve_client_config_impl(
    config: ClientConfig, ancestors: list[ClientConfig] | None = None
) -> ClientConfig:
    """Resolve possibly nested endpoint configs to a single concrete client config."""
    chain = list(ancestors or [])
    visited = {id(ancestor) for ancestor in chain}
    resolved = config
    while resolved.endpoint_configs:
        resolved_id = id(resolved)
        if resolved_id in visited:
            raise ValueError(
                "Detected cyclic ClientConfig.endpoint_configs while resolving client"
            )
        visited.add(resolved_id)
        chain.append(resolved)
        endpoint_idx = resolved.client_idx % len(resolved.endpoint_configs)
        resolved = resolved.endpoint_configs[endpoint_idx]

    resolved_data = resolved.model_dump(mode="python")
    resolved_fields = set(resolved.model_fields_set)
    for parent in reversed(chain):
        for field_name in ClientConfig.model_fields:
            if field_name == "endpoint_configs":
                continue
            if field_name not in resolved_fields:
                resolved_data[field_name] = getattr(parent, field_name)
                resolved_fields.add(field_name)

    return ClientConfig.model_validate(resolved_data)


def resolve_client_config(config: ClientConfig) -> ClientConfig:
    return _resolve_client_config_impl(config)


def resolve_client_configs(config: ClientConfig) -> list[ClientConfig]:
    """Expand a client config into one or more resolved endpoint configs."""
    if config.endpoint_configs:
        return [
            _resolve_client_config_impl(endpoint, ancestors=[config])
            for endpoint in config.endpoint_configs
        ]
    return [resolve_client_config(config)]


def load_prime_config() -> dict:
    try:
        config_file = Path.home() / ".prime" / "config.json"
        if config_file.exists():
            data = json.loads(config_file.read_text())
            if isinstance(data, dict):
                return data
            logger.warning("Invalid prime config: expected dict")
    except (RuntimeError, json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load prime config: {e}")
    return {}


def _build_headers_and_api_key(config: ClientConfig) -> tuple[dict[str, str], str | None]:
    headers = dict(config.extra_headers)
    api_key = os.getenv(config.api_key_var)

    if config.api_key_var == "PRIME_API_KEY":
        prime_config = load_prime_config()
        if not api_key:
            api_key = prime_config.get("api_key", "")
        team_id = os.getenv("PRIME_TEAM_ID") or prime_config.get("team_id")
        if team_id:
            headers["X-Prime-Team-ID"] = team_id

    return headers, api_key


def _build_http_client(config: ClientConfig, headers: dict[str, str]) -> httpx.AsyncClient:
    timeout = httpx.Timeout(config.timeout, connect=5.0)
    limits = httpx.Limits(
        max_connections=config.max_connections,
        max_keepalive_connections=config.max_keepalive_connections,
    )
    return httpx.AsyncClient(
        limits=limits,
        timeout=timeout,
        headers=headers,
    )


def setup_http_client(config: ClientConfig) -> httpx.AsyncClient:
    """Setup base HTTP client with timeouts, limits, and PRIME headers."""
    resolved_config = resolve_client_config(config)
    headers, _ = _build_headers_and_api_key(resolved_config)
    return _build_http_client(resolved_config, headers)


def _setup_openai_client_from_resolved(config: ClientConfig) -> AsyncOpenAI:
    assert config.client_type == "openai"
    headers, api_key = _build_headers_and_api_key(config)
    return AsyncOpenAI(
        api_key=api_key or "EMPTY",
        base_url=config.api_base_url,
        max_retries=config.max_retries,
        http_client=_build_http_client(config, headers),
    )


def setup_openai_client(config: ClientConfig) -> AsyncOpenAI:
    """Setup an AsyncOpenAI client from config."""
    resolved_config = resolve_client_config(config)
    return _setup_openai_client_from_resolved(resolved_config)


def _setup_anthropic_client_from_resolved(config: ClientConfig) -> AsyncAnthropic:
    assert config.client_type == "anthropic"
    headers, api_key = _build_headers_and_api_key(config)
    return AsyncAnthropic(
        api_key=api_key or "EMPTY",
        base_url=config.api_base_url,
        max_retries=config.max_retries,
        http_client=_build_http_client(config, headers),
    )


def setup_anthropic_client(config: ClientConfig) -> AsyncAnthropic:
    """Setup an AsyncAnthropic client from config."""
    resolved_config = resolve_client_config(config)
    return _setup_anthropic_client_from_resolved(resolved_config)


def setup_client(config: ClientConfig) -> AsyncOpenAI | AsyncAnthropic:
    """Setup the appropriate async client based on config.client_type."""
    resolved_config = resolve_client_config(config)
    if resolved_config.client_type == "openai":
        return _setup_openai_client_from_resolved(resolved_config)
    if resolved_config.client_type == "anthropic":
        return _setup_anthropic_client_from_resolved(resolved_config)
    raise ValueError(f"Unsupported client type: {resolved_config.client_type}")
