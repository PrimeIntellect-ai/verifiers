import json
import logging
import os
from pathlib import Path

import httpx
from httpx import AsyncClient
from openai import AsyncOpenAI

from verifiers.types import ClientConfig

logger = logging.getLogger(__name__)


def _resolve_client_config_impl(config: ClientConfig) -> ClientConfig:
    """Resolve endpoint config overrides onto a concrete client config."""
    if not config.endpoint_configs:
        return ClientConfig.model_validate(config.model_dump(mode="python"))

    endpoint_idx = config.client_idx % len(config.endpoint_configs)
    endpoint_config = config.endpoint_configs[endpoint_idx]

    resolved_data = endpoint_config.model_dump(mode="python")
    resolved_fields = set(endpoint_config.model_fields_set)
    for field_name in ClientConfig.model_fields:
        if field_name == "endpoint_configs":
            continue
        if field_name not in resolved_fields:
            resolved_data[field_name] = getattr(config, field_name)

    return ClientConfig.model_validate(resolved_data)


def resolve_client_config(config: ClientConfig) -> ClientConfig:
    return _resolve_client_config_impl(config)


def resolve_client_configs(config: ClientConfig) -> list[ClientConfig]:
    """Expand a client config into one or more resolved endpoint configs."""
    if config.endpoint_configs:
        expanded_configs: list[ClientConfig] = []
        for endpoint_config in config.endpoint_configs:
            expanded_data = endpoint_config.model_dump(mode="python")
            expanded_fields = set(endpoint_config.model_fields_set)
            for field_name in ClientConfig.model_fields:
                if field_name == "endpoint_configs":
                    continue
                if field_name not in expanded_fields:
                    expanded_data[field_name] = getattr(config, field_name)
            expanded_configs.append(ClientConfig.model_validate(expanded_data))
        return expanded_configs
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


def setup_client(config: ClientConfig) -> AsyncOpenAI:
    """A helper function to setup an AsyncOpenAI client."""
    resolved_config = resolve_client_config(config)

    # Setup timeouts and limits
    http_timeout = httpx.Timeout(resolved_config.timeout, connect=5.0)
    limits = httpx.Limits(
        max_connections=resolved_config.max_connections,
        max_keepalive_connections=resolved_config.max_keepalive_connections,
    )

    headers = resolved_config.extra_headers
    api_key = os.getenv(resolved_config.api_key_var)

    # Fall back to prime config if using PRIME_API_KEY
    if resolved_config.api_key_var == "PRIME_API_KEY":
        prime_config = load_prime_config()
        if not api_key:
            api_key = prime_config.get("api_key", "")
        team_id = os.getenv("PRIME_TEAM_ID") or prime_config.get("team_id")
        if team_id:
            headers = {**resolved_config.extra_headers, "X-Prime-Team-ID": team_id}

    # Setup client
    http_client = AsyncClient(
        limits=limits,
        timeout=http_timeout,
        headers=headers,
    )
    client = AsyncOpenAI(
        base_url=resolved_config.api_base_url,
        api_key=api_key or "EMPTY",
        max_retries=resolved_config.max_retries,
        http_client=http_client,
    )

    return client
