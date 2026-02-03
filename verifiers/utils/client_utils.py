import json
import logging
import os
from pathlib import Path

import httpx
from httpx import AsyncClient
from openai import AsyncOpenAI

from verifiers.types import ClientConfig

logger = logging.getLogger(__name__)


class ClientPool:
    """Round-robin pool of client configurations for multi-server inference.

    Used to distribute requests across multiple vLLM servers. Each call to
    get_next_config() returns the next ClientConfig in round-robin order.

    Example:
        pool = ClientPool.from_urls(["http://server1:8000/v1", "http://server2:8000/v1"])

        for group in groups:
            config = pool.get_next_config()  # Round-robin per group
            await env.run_group(group, client=config, ...)
    """

    def __init__(self, configs: list[ClientConfig]):
        if not configs:
            raise ValueError("ClientPool requires at least one ClientConfig")
        self._configs = configs
        self._index = 0

    def get_next_config(self) -> ClientConfig:
        """Get next client config in round-robin order."""
        config = self._configs[self._index % len(self._configs)]
        self._index += 1
        return config

    def __len__(self) -> int:
        return len(self._configs)

    @classmethod
    def from_urls(
        cls,
        urls: list[str],
        api_key_var: str = "PRIME_API_KEY",
        **kwargs,
    ) -> "ClientPool":
        """Create a ClientPool from a list of base URLs.

        Args:
            urls: List of API base URLs (e.g., ["http://server1:8000/v1", ...])
            api_key_var: Environment variable name for API key
            **kwargs: Additional arguments passed to each ClientConfig
        """
        configs = [
            ClientConfig(api_base_url=url, api_key_var=api_key_var, **kwargs)
            for url in urls
        ]
        return cls(configs)

    @classmethod
    def from_config(cls, config: ClientConfig) -> "ClientPool":
        """Create a ClientPool from a ClientConfig.

        If config.api_base_url is a list, creates a pool with one config per URL.
        If config.api_base_url is a single string, creates a single-config pool.
        """
        urls = config.api_base_url
        if isinstance(urls, list):
            # Multi-URL config: create separate ClientConfig for each URL
            configs = [
                ClientConfig(
                    api_base_url=url,
                    api_key_var=config.api_key_var,
                    timeout=config.timeout,
                    max_connections=config.max_connections,
                    max_keepalive_connections=config.max_keepalive_connections,
                    max_retries=config.max_retries,
                    extra_headers=config.extra_headers,
                )
                for url in urls
            ]
            return cls(configs)
        else:
            # Single URL: wrap in pool for uniform interface
            return cls([config])


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
    """A helper function to setup an AsyncOpenAI client.

    Note: config.api_base_url must be a single URL string, not a list.
    For multi-URL configs, use ClientPool.from_config() instead.
    """
    # Validate single URL
    if isinstance(config.api_base_url, list):
        raise ValueError(
            "setup_client() requires a single URL. "
            "For multi-URL configs, use ClientPool.from_config() instead."
        )

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


def resolve_client(config: ClientConfig) -> "ClientConfig | ClientPool":
    """Resolve a ClientConfig to either itself or a ClientPool.

    If config.api_base_url is a list with multiple URLs, returns a ClientPool.
    Otherwise, returns the original config unchanged.
    """
    if isinstance(config.api_base_url, list) and len(config.api_base_url) > 1:
        return ClientPool.from_config(config)
    return config
