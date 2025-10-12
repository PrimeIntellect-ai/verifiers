import os
from typing import Dict

import httpx
from httpx import AsyncClient
from openai import AsyncOpenAI


def setup_client(
    api_base_url: str,
    api_key_var: str,
    timeout: float = 600.0,  # OAI default, larger value recommended for evals
    max_connections: int = 1000,  # OAI default, larger value recommended for evals
    max_keepalive_connections: int = 100,  # OAI default, larger value recommended for evals
    max_retries: int = 2,  # OAI default, larger value recommended for evals
    extra_headers: Dict[str, str] | None = None,
) -> AsyncOpenAI:
    """
    A helper function to setup an AsyncOpenAI client.
    """
    # Setup timeouts and limits
    http_timeout = httpx.Timeout(timeout, connect=5.0)
    limits = httpx.Limits(
        max_connections=max_connections,
        max_keepalive_connections=max_keepalive_connections,
    )

    # Setup client
    http_client = AsyncClient(
        limits=limits,
        timeout=http_timeout,
        headers=extra_headers,
    )
    client = AsyncOpenAI(
        base_url=api_base_url,
        api_key=os.getenv(api_key_var, "EMPTY"),
        max_retries=max_retries,
        http_client=http_client,
    )

    return client
