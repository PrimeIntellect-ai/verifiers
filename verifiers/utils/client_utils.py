import os

import httpx
from httpx import AsyncClient
from openai import AsyncOpenAI

from verifiers.types import ClientConfig


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

    # Setup client
    http_client = AsyncClient(
        limits=limits,
        timeout=http_timeout,
        headers=config.extra_headers,
    )
    client = AsyncOpenAI(
        base_url=config.api_base_url,
        api_key=os.getenv(config.api_key_var, "EMPTY"),
        max_retries=config.max_retries,
        http_client=http_client,
    )

    return client
