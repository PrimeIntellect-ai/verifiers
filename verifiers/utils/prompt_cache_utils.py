from collections.abc import Mapping
from typing import Any, TypeVar
from urllib.parse import urlsplit

from verifiers.types import ClientConfig

NativePromptT = TypeVar("NativePromptT")
NativeToolsT = TypeVar("NativeToolsT")

ANTHROPIC_ORIGINS = frozenset({"https://api.anthropic.com"})


def endpoint_origin(api_base_url: str) -> str | None:
    parsed = urlsplit(api_base_url)
    if not parsed.scheme or not parsed.hostname:
        return None
    scheme = parsed.scheme.lower()
    host = parsed.hostname.lower()
    port = parsed.port
    netloc = host
    if port is not None and not (
        (scheme == "https" and port == 443) or (scheme == "http" and port == 80)
    ):
        netloc = f"{host}:{port}"
    return f"{scheme}://{netloc}"


def uses_official_anthropic_messages(config: ClientConfig | None) -> bool:
    return (
        config is not None
        and config.client_type == "anthropic_messages"
        and endpoint_origin(config.api_base_url) in ANTHROPIC_ORIGINS
    )


def _cache_control_payload() -> dict[str, str]:
    return {"type": "ephemeral"}


def apply_prompt_cache_to_request(
    *,
    config: ClientConfig | None,
    model: str,
    native_prompt: NativePromptT,
    native_tools: NativeToolsT,
    sampling_args: Mapping[str, Any],
    extra_kwargs: Mapping[str, Any],
) -> tuple[NativePromptT, NativeToolsT, dict[str, Any], dict[str, Any]]:
    _ = model
    updated_extra_kwargs = dict(extra_kwargs)
    if uses_official_anthropic_messages(config):
        updated_extra_kwargs.setdefault("cache_control", _cache_control_payload())
    return native_prompt, native_tools, dict(sampling_args), updated_extra_kwargs
