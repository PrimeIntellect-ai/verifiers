import logging
import os
from collections.abc import Mapping
from typing import Any

import httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from verifiers.types import (
    ClientConfig,
    EndpointClientConfig,
)
from verifiers.utils.response_utils import strip_routed_experts_data

logger = logging.getLogger(__name__)


def _merge_endpoint(
    parent: ClientConfig, endpoint: EndpointClientConfig
) -> ClientConfig:
    """Merge parent config fields into an endpoint config, preserving endpoint overrides."""
    merged_data = endpoint.model_dump(mode="python")
    explicitly_set = set(endpoint.model_fields_set)
    for field_name in ClientConfig.model_fields:
        if field_name == "endpoint_configs":
            continue
        if field_name not in explicitly_set:
            merged_data[field_name] = getattr(parent, field_name)
    return ClientConfig.model_validate(merged_data)


def resolve_client_config(config: ClientConfig) -> ClientConfig:
    """Resolve endpoint config overrides onto a concrete client config."""
    if not config.endpoint_configs:
        return ClientConfig.model_validate(config.model_dump(mode="python"))

    endpoint_idx = config.client_idx % len(config.endpoint_configs)
    return _merge_endpoint(config, config.endpoint_configs[endpoint_idx])


def resolve_client_configs(config: ClientConfig) -> list[ClientConfig]:
    """Expand a client config into one or more resolved endpoint configs."""
    if config.endpoint_configs:
        return [_merge_endpoint(config, ep) for ep in config.endpoint_configs]
    return [resolve_client_config(config)]


def _build_headers_and_api_key(
    config: ClientConfig,
) -> tuple[dict[str, str], str | None]:
    headers = dict(config.extra_headers)
    api_key = os.getenv(config.api_key_var)

    if config.api_key_var == "PRIME_API_KEY":
        team_id = os.getenv("PRIME_TEAM_ID")
        if team_id:
            headers["X-Prime-Team-ID"] = team_id

    return headers, api_key


def _build_http_client(
    config: ClientConfig, headers: dict[str, str]
) -> httpx.AsyncClient:
    timeout = httpx.Timeout(config.timeout, connect=config.connect_timeout)
    limits = httpx.Limits(
        max_connections=config.max_connections,
        max_keepalive_connections=config.max_keepalive_connections,
    )
    return httpx.AsyncClient(
        limits=limits,
        timeout=timeout,
        headers=headers,
    )


async def post_chat_completion_with_routed_experts_sidecar(
    client: AsyncOpenAI,
    path: str,
    *,
    body: dict[str, Any],
    extra_headers: Mapping[str, str] | None = None,
) -> ChatCompletion:
    raw_response = await client.post(
        path,
        body=body,
        cast_to=httpx.Response,
        options={"headers": extra_headers} if extra_headers else {},
    )
    stripped, routed_data = strip_routed_experts_data(raw_response.content)
    response = ChatCompletion.model_validate_json(stripped)
    if routed_data is not None:
        choice_extra = response.choices[0].model_extra
        assert choice_extra is not None
        choice_extra["routed_experts"]["data"] = routed_data
    return response


def setup_openai_client(config: ClientConfig) -> AsyncOpenAI:
    """Setup an AsyncOpenAI client from config."""
    resolved_config = resolve_client_config(config)
    headers, api_key = _build_headers_and_api_key(resolved_config)
    return AsyncOpenAI(
        api_key=api_key or "EMPTY",
        base_url=resolved_config.api_base_url,
        max_retries=resolved_config.max_retries,
        http_client=_build_http_client(resolved_config, headers),
    )


def setup_anthropic_client(config: ClientConfig) -> AsyncAnthropic:
    """Setup an AsyncAnthropic client from config."""
    resolved_config = resolve_client_config(config)
    headers, api_key = _build_headers_and_api_key(resolved_config)
    return AsyncAnthropic(
        api_key=api_key or "EMPTY",
        base_url=resolved_config.api_base_url,
        max_retries=resolved_config.max_retries,
        http_client=_build_http_client(resolved_config, headers),
    )
