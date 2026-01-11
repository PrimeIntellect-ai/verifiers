import json
import logging
import os
from pathlib import Path

import httpx
from httpx import AsyncClient
from openai import AsyncOpenAI

from verifiers.types import ClientConfig

logger = logging.getLogger(__name__)


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


def setup_client(
    config: ClientConfig,
) -> AsyncOpenAI:
    """
    A helper function to setup an AsyncOpenAI client.
    """
    # Setup timeouts and limits
    http_timeout = httpx.Timeout(config.timeout, connect=5.0)
    limits = httpx.Limits(
        max_connections=config.max_connections,
        max_keepalive_connections=config.max_keepalive_connections,
    )

    headers = config.extra_headers
    api_key = os.getenv(config.api_key_var)

    # Fall back to prime config if using PRIME_API_KEY
    if config.api_key_var == "PRIME_API_KEY":
        prime_config = load_prime_config()
        if not api_key:
            api_key = prime_config.get("api_key", "")
        team_id = os.getenv("PRIME_TEAM_ID") or prime_config.get("team_id")
        if team_id:
            headers = {**config.extra_headers, "X-Prime-Team-ID": team_id}

    # Setup client
    http_client = AsyncClient(
        limits=limits,
        timeout=http_timeout,
        headers=headers,
    )
    client = AsyncOpenAI(
        base_url=config.api_base_url,
        api_key=api_key or "EMPTY",
        max_retries=config.max_retries,
        http_client=http_client,
    )

    return client
