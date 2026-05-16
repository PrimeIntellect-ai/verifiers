from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlsplit

from verifiers.types import ClientConfig, ClientType

PromptCacheMode = Literal[
    "disabled",
    "implicit",
    "anthropic_top_level",
    "openrouter_anthropic_top_level",
]

OPENAI_CACHE_CLIENT_TYPES: frozenset[ClientType] = frozenset(
    {
        "openai_chat_completions",
        "openai_responses",
    }
)
ANTHROPIC_CACHE_CLIENT_TYPES: frozenset[ClientType] = frozenset(
    {"anthropic_messages"}
)
OPENROUTER_CACHE_CLIENT_TYPES: frozenset[ClientType] = frozenset(
    {
        "openai_chat_completions",
        "openai_responses",
    }
)


@dataclass(frozen=True)
class EndpointIdentity:
    client_type: ClientType
    origin: str
    host: str
    path: str

    @classmethod
    def from_config(cls, config: ClientConfig) -> "EndpointIdentity | None":
        return cls.from_url(config.api_base_url, config.client_type)

    @classmethod
    def from_url(
        cls, api_base_url: str, client_type: ClientType
    ) -> "EndpointIdentity | None":
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
        return cls(
            client_type=client_type,
            origin=f"{scheme}://{netloc}",
            host=host,
            path=parsed.path or "",
        )


@dataclass(frozen=True)
class PromptCachePolicy:
    mode: PromptCacheMode = "disabled"
    prefire_groups: bool = False

    @property
    def enabled(self) -> bool:
        return self.mode != "disabled"


class PromptCacheAdapter:
    prefire_groups = True

    def policy_for(
        self, identity: EndpointIdentity, model: str
    ) -> PromptCachePolicy:
        _ = identity, model
        return PromptCachePolicy(mode="implicit", prefire_groups=self.prefire_groups)


class AnthropicPromptCacheAdapter(PromptCacheAdapter):
    def policy_for(
        self, identity: EndpointIdentity, model: str
    ) -> PromptCachePolicy:
        _ = identity, model
        return PromptCachePolicy(
            mode="anthropic_top_level", prefire_groups=self.prefire_groups
        )


class OpenRouterPromptCacheAdapter(PromptCacheAdapter):
    anthropic_model_prefixes = ("anthropic/",)

    def policy_for(
        self, identity: EndpointIdentity, model: str
    ) -> PromptCachePolicy:
        _ = identity
        if model.startswith(self.anthropic_model_prefixes):
            return PromptCachePolicy(
                mode="openrouter_anthropic_top_level",
                prefire_groups=self.prefire_groups,
            )
        return PromptCachePolicy(mode="implicit", prefire_groups=self.prefire_groups)


@dataclass(frozen=True)
class ProviderSpec:
    provider_id: str
    origins: frozenset[str]
    client_types: frozenset[ClientType]
    prompt_cache: PromptCacheAdapter

    def recognizes(self, identity: EndpointIdentity) -> bool:
        return (
            identity.origin in self.origins
            and identity.client_type in self.client_types
        )


PROVIDER_SPECS: tuple[ProviderSpec, ...] = (
    ProviderSpec(
        provider_id="openai",
        origins=frozenset({"https://api.openai.com"}),
        client_types=OPENAI_CACHE_CLIENT_TYPES,
        prompt_cache=PromptCacheAdapter(),
    ),
    ProviderSpec(
        provider_id="anthropic",
        origins=frozenset({"https://api.anthropic.com"}),
        client_types=ANTHROPIC_CACHE_CLIENT_TYPES,
        prompt_cache=AnthropicPromptCacheAdapter(),
    ),
    ProviderSpec(
        provider_id="openrouter",
        origins=frozenset({"https://openrouter.ai"}),
        client_types=OPENROUTER_CACHE_CLIENT_TYPES,
        prompt_cache=OpenRouterPromptCacheAdapter(),
    ),
)

DISABLED_PROMPT_CACHE_POLICY = PromptCachePolicy()


def infer_provider_spec(config: ClientConfig) -> ProviderSpec | None:
    identity = EndpointIdentity.from_config(config)
    if identity is None:
        return None
    for spec in PROVIDER_SPECS:
        if spec.recognizes(identity):
            return spec
    return None


def resolve_prompt_cache_policy(
    config: ClientConfig | None, model: str
) -> PromptCachePolicy:
    if config is None or not config.prompt_cache:
        return DISABLED_PROMPT_CACHE_POLICY
    identity = EndpointIdentity.from_config(config)
    if identity is None:
        return DISABLED_PROMPT_CACHE_POLICY
    spec = infer_provider_spec(config)
    if spec is None:
        return DISABLED_PROMPT_CACHE_POLICY
    return spec.prompt_cache.policy_for(identity, model)


def should_prefire_prompt_cache_group(
    client_or_config: object, model: str, group_size: int
) -> bool:
    if group_size <= 1:
        return False
    config = client_or_config if isinstance(client_or_config, ClientConfig) else None
    if config is None:
        config = getattr(client_or_config, "config", None)
    if not isinstance(config, ClientConfig):
        return False
    return resolve_prompt_cache_policy(config, model).prefire_groups


def _cache_control_payload() -> dict[str, str]:
    return {"type": "ephemeral"}


def apply_prompt_cache_to_request(
    *,
    config: ClientConfig | None,
    model: str,
    native_prompt: object,
    native_tools: object,
    sampling_args: Mapping[str, Any],
    extra_kwargs: Mapping[str, Any],
) -> tuple[object, object, dict[str, Any], dict[str, Any]]:
    policy = resolve_prompt_cache_policy(config, model)
    updated_sampling_args = dict(sampling_args)
    updated_extra_kwargs = dict(extra_kwargs)
    updated_native_prompt = native_prompt
    if policy.mode == "anthropic_top_level":
        updated_extra_kwargs.setdefault("cache_control", _cache_control_payload())
    elif policy.mode == "openrouter_anthropic_top_level":
        extra_body = updated_sampling_args.get("extra_body")
        if isinstance(extra_body, Mapping):
            extra_body = dict(extra_body)
        else:
            extra_body = {}
        extra_body.setdefault("cache_control", _cache_control_payload())
        updated_sampling_args["extra_body"] = extra_body
    return (
        updated_native_prompt,
        native_tools,
        updated_sampling_args,
        updated_extra_kwargs,
    )
