from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from verifiers.types import ClientConfig

ANTHROPIC_ORIGINS = frozenset({"https://api.anthropic.com"})


def endpoint_origin(api_base_url: str) -> str | None:
    parsed = urlsplit(api_base_url)
    if not parsed.scheme or not parsed.hostname:
        return None
    scheme = parsed.scheme.lower()
    host = parsed.hostname.lower()
    port = parsed.port
    netloc = host
    if ":" in host:
        netloc = f"[{host}]"
    if port is not None and not (
        (scheme == "https" and port == 443) or (scheme == "http" and port == 80)
    ):
        netloc = f"{netloc}:{port}"
    return f"{scheme}://{netloc}"


def uses_official_anthropic_messages(config: ClientConfig | None) -> bool:
    return (
        config is not None
        and config.client_type == "anthropic_messages"
        and endpoint_origin(config.api_base_url) in ANTHROPIC_ORIGINS
    )


def _cache_control_payload() -> dict[str, str]:
    return {"type": "ephemeral"}


def apply_prompt_cache_to_kwargs(
    *,
    config: ClientConfig | None,
    sampling_args: Mapping[str, Any],
    extra_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    updated_extra_kwargs = dict(extra_kwargs)
    if (
        uses_official_anthropic_messages(config)
        and "cache_control" not in sampling_args
    ):
        updated_extra_kwargs.setdefault("cache_control", _cache_control_payload())
    return updated_extra_kwargs
