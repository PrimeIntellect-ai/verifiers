"""Shared client plumbing: transport defaults and URL/key utilities."""

import re

import httpx2 as httpx
from openai import AsyncOpenAI

from verifiers.v1.configs.client import BaseClientConfig, resolve_api_key

SESSION_ID_HEADER = "X-Session-ID"
"""Pin a rollout's requests to one provider engine for prefix-cache reuse."""

DEFAULT_LIMITS = httpx.Limits(max_connections=1000, max_keepalive_connections=100)
MAX_RETRIES = 0
"""No client-side retries: failures surface to the harness SDK and the trace instead of
being silently reattempted."""

# An API version path segment (`v1`, `v2`, ...) — the only kind `join_url` dedups.
VERSION_SEGMENT = re.compile(r"v\d+")


def build_async_httpx(config: BaseClientConfig) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(**config.timeout.model_dump()), limits=DEFAULT_LIMITS
    )


def build_async_openai(config: BaseClientConfig) -> AsyncOpenAI:
    http_client = build_async_httpx(config)
    # Pass it explicitly: SDK defaults can replace a transport timeout equal to HTTPX's.
    return AsyncOpenAI(
        base_url=config.base_url,
        api_key=resolve_api_key(config),
        default_headers=config.headers or None,
        max_retries=MAX_RETRIES,
        timeout=http_client.timeout,
        http_client=http_client,
    )


def join_url(base_url: str, path: str) -> str:
    """Join `base_url` with a dialect path without repeating the API version segment:
    `.../api/v1` + `/v1/messages` -> `.../api/v1/messages`. Only version-shaped segments
    dedup, so a base ending in `/chat` doesn't swallow `/chat/completions`."""
    head = path.split("/")[1] if path.startswith("/") else ""
    base = base_url.rstrip("/")
    if VERSION_SEGMENT.fullmatch(head) and base.endswith(f"/{head}"):
        base = base[: -len(head) - 1]
    return base + path
