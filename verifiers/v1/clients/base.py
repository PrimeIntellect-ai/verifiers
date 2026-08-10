"""Shared client plumbing: transport defaults and URL/key utilities."""

import re

import httpx
from openai import AsyncOpenAI

from verifiers.v1.configs.client import BaseClientConfig, resolve_api_key

# No read timeout: agentic completions are slow and the rollout timeout is the real
# backstop. The connect bound stays so an unreachable endpoint still fails fast.
DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=None, write=None, pool=None)
DEFAULT_LIMITS = httpx.Limits(max_connections=1000, max_keepalive_connections=100)
MAX_RETRIES = 0
"""No client-side retries: failures surface to the harness SDK and the trace instead of
being silently reattempted."""

# An API version path segment (`v1`, `v2`, ...) — the only kind `join_url` dedups.
VERSION_SEGMENT = re.compile(r"v\d+")


def build_async_openai(config: BaseClientConfig) -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=config.base_url,
        api_key=resolve_api_key(config),
        default_headers=config.headers or None,
        timeout=DEFAULT_TIMEOUT,
        max_retries=MAX_RETRIES,
        http_client=httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, limits=DEFAULT_LIMITS),
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
