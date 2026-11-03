"""One env agent's config: who plays the seat, and its per-run caps."""

from typing import Any

from pydantic import (
    BaseModel,
    Field,
    SerializationInfo,
    SerializeAsAny,
    field_serializer,
    model_validator,
)
from pydantic_config import BaseConfig

from verifiers.v1.configs.client import ClientConfig
from verifiers.v1.configs.deployment import DeploymentConfig
from verifiers.v1.configs.harness import HarnessConfig, WireHarnessConfig
from verifiers.v1.configs.retries import RetryConfig
from verifiers.v1.configs.runtime import PrimeConfig, RuntimeConfig
from verifiers.v1.configs.sampling import SamplingConfig
from verifiers.v1.configs.scoring import ScoringConfig
from verifiers.v1.utils.generic import merge_defaults


class TimeoutConfig(BaseConfig):
    """Timeout (in seconds) for different phases of an agent's run. Each unset
    stage takes the task's own timeout for it, else runs with no limit."""

    # one shared budget: task setup + provisioning
    setup: float | None = Field(None, gt=0)
    """Timeout (in seconds) for task/harness setup through session preparation."""
    rollout: float | None = Field(None, gt=0)
    """Timeout (in seconds) for the agent's solve attempt."""
    finalize: float | None = Field(None, gt=0)
    """Timeout (in seconds) for the task + harness finalize hooks."""
    scoring: float | None = Field(None, gt=0)
    """Combined budget (in seconds) for harness metrics and task scoring, including
    isolated verifier setup and retries. Finalization has its own budget."""


class AgentConfig(BaseConfig):
    harness: SerializeAsAny[HarnessConfig] | None = None
    """The agent's program (None = the taskset's default harness)."""
    runtime: RuntimeConfig = PrimeConfig()
    """Runtime configuration for the harness program — the policy each run provisions its box
    from; tool servers choose their placement separately."""

    scoring: ScoringConfig = ScoringConfig()
    """Scoring overrides; unset fields inherit the task’s scoring defaults."""

    deployment: DeploymentConfig = DeploymentConfig()
    """Workspace and service placement, including trust for local Compose tasks."""

    model: str | None = None
    """Model id (None = the run's model, i.e. the policy under evaluation/training)."""
    client: ClientConfig | None = None
    """Endpoint override (None = the run's client, with the run's sampling)."""
    sampling: SamplingConfig | None = None
    """Sampling values merged onto the run's sampling. A seat with its own `client`
    does not inherit the run's sampling: it uses only these values."""

    max_turns: int | None = None
    """Max model turns per run (None = no limit)."""
    max_input_tokens: int | None = None
    """Max input tokens per run (None = no limit)."""
    max_output_tokens: int | None = None
    """Max output tokens per run (None = no limit)."""
    max_total_tokens: int | None = None
    """Max total tokens per run (None = no limit)."""
    max_turn_tokens: int | None = Field(None, gt=0)
    """Max output tokens per model response (None = no limit). Caps the sampling
    `max_tokens` the framework sets on every request, whatever the harness asks for,
    and stops the run once a response hits it. `sampling.max_tokens` alone caps each
    response without stopping the run."""

    timeout: TimeoutConfig = TimeoutConfig()
    retries: RetryConfig = RetryConfig()

    network_notice: bool = True
    """Include the runtime's network restrictions in the agent's prompt."""

    @field_serializer("scoring")
    def _scoring_overrides(
        self, scoring: ScoringConfig, info: SerializationInfo
    ) -> dict:
        # Preserve unset fields across env-server and trace roundtrips so they
        # continue to inherit from the task instead of becoming explicit overrides.
        return scoring.model_dump(mode=info.mode, exclude_unset=True)

    @field_serializer("sampling")
    def _sampling_overrides(self, sampling: SamplingConfig | None) -> dict | None:
        # An unresolved agent carries only overrides across env-server boundaries.
        # resolve_agent fills every field before the config is recorded on a trace.
        return sampling.model_dump(exclude_unset=True) if sampling is not None else None

    @model_validator(mode="before")
    @classmethod
    def _resolve_harness(cls, data):
        """Narrow a pinned `harness` to its concrete config type by `id` (absent
        stays None = the taskset's default). The lazy import keeps class-body
        `AgentConfig()` defaults constructible while this module initializes."""
        if isinstance(data, dict) and data.get("harness") is not None:
            from verifiers.v1.utils.loaders import (
                harness_config_type,
                narrow_plugin_field,
            )

            narrow_plugin_field(data, "harness", harness_config_type, "bash")
        return data


class WireAgentConfig(AgentConfig):
    """Wire form for trace records: parses without resolving the harness plugin,
    so records round-trip anywhere — the knobs stay readable on the extra-allow
    `WireHarnessConfig` (see `WireTaskData`)."""

    harness: SerializeAsAny[WireHarnessConfig] | None = None

    @model_validator(mode="before")
    @classmethod
    def _resolve_harness(cls, data):
        """Override: a record read resolves no plugins."""
        return data


def agent_config_fields(config: BaseModel) -> dict[str, AgentConfig]:
    """Top-level agent configs, in declaration order, keyed by their field names."""
    return {name: value for name, value in config if isinstance(value, AgentConfig)}


def merge_agent_defaults(config: type[BaseModel], data: Any) -> Any:
    """Merge partial agent overrides onto their declared defaults."""
    if isinstance(data, dict):
        for name, field in config.model_fields.items():
            if isinstance(field.default, AgentConfig) and isinstance(
                data.get(name), dict
            ):
                data[name] = merge_defaults(field.default, data[name])
    return data


def resolve_agent(
    spec: AgentConfig,
    *,
    model: str | None = None,
    client: ClientConfig | None = None,
    sampling: SamplingConfig | None = None,
    harness: HarnessConfig | None = None,
) -> AgentConfig:
    """`spec` with what it leaves unset filled from the run's defaults; its own
    sampling values merge over the run's. The one place a seat's identity resolves,
    for configured agent roles.

    The run's sampling belongs to the run's client: it can carry endpoint-specific
    keys (e.g. vLLM's `top_k`) that another provider rejects. A seat that pins its
    own `client` therefore starts from empty sampling."""
    defaults = (
        sampling if sampling is not None and spec.client is None else SamplingConfig()
    )
    overrides = (spec.sampling or SamplingConfig()).model_dump(exclude_unset=True)
    merged = {**defaults.model_dump(), **merge_defaults(defaults, overrides)}
    if spec.max_turn_tokens is not None:
        max_tokens = merged.get("max_tokens")
        merged["max_tokens"] = min(
            max_tokens or spec.max_turn_tokens, spec.max_turn_tokens
        )
    return spec.model_copy(
        update={
            "harness": spec.harness if spec.harness is not None else harness,
            "model": spec.model if spec.model is not None else model,
            "client": spec.client if spec.client is not None else client,
            "sampling": SamplingConfig.model_validate(merged),
        }
    )
