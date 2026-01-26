import json
import logging
import os
from pathlib import Path

import httpx

from verifiers.types import (
    ClientConfig,
)

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


def setup_http_client(config: ClientConfig) -> httpx.AsyncClient:
    """Setup base HTTP client."""

    def setup_limits(
        max_connections: int, max_keepalive_connections: int
    ) -> httpx.Limits:
        return httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
        )

    def setup_timeout(timeout: float) -> httpx.Timeout:
        return httpx.Timeout(timeout, connect=5.0)

    def setup_headers() -> dict[str, str]:
        headers = config.extra_headers
        api_key = os.getenv(config.api_key_var)

        # Fall back to prime config if using PRIME_API_KEY
        if config.api_key_var == "PRIME_API_KEY":
            prime_config = load_prime_config()
            if not api_key:
                api_key = prime_config.get("api_key", "")
            team_id = os.getenv("PRIME_TEAM_ID") or prime_config.get("team_id")
            if team_id:
                headers = {**headers, "X-Prime-Team-ID": team_id}
        return headers

    limits = setup_limits(config.max_connections, config.max_keepalive_connections)
    timeout = setup_timeout(config.timeout)
    headers = setup_headers()
    return httpx.AsyncClient(
        limits=limits,
        timeout=timeout,
        headers=headers,
    )
