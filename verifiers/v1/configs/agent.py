"""One env agent's config: who plays the seat, and its per-run caps."""

from typing import Literal

from pydantic import SerializeAsAny, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.clients import ClientConfig
from verifiers.v1.configs.harness import HarnessConfig, WireHarnessConfig
from verifiers.v1.configs.retries import RetryConfig
from verifiers.v1.runtimes import PrimeConfig, RuntimeConfig
from verifiers.v1.types import SamplingConfig


class TimeoutConfig(BaseConfig):
    """Timeout (in seconds) for different phases of an agent's run."""

    setup: float | None = None  # one shared budget: task setup + provisioning
    """Timeout (in seconds) for the task + harness setup hooks."""
    rollout: float | None = None
    """Timeout (in seconds) for the agent's solve attempt."""
    on_rollout_timeout: Literal["error", "truncate"] = "error"
    """How to record an expired rollout timeout. ``error`` preserves the current
    failure behavior. ``truncate`` finalizes and scores the partial trajectory like
    a context-limit stop."""
    finalize: float | None = None
    """Timeout (in seconds) for the task + harness finalize hooks."""
    scoring: float | None = None
    """Timeout (in seconds) for the task + harness metrics + scoring hooks."""


class AgentConfig(BaseConfig):
    harness: SerializeAsAny[HarnessConfig] | None = None
    """The agent's program (None = the taskset's default harness)."""
    runtime: RuntimeConfig = PrimeConfig()
    """Runtime for the harness program — the policy each run provisions its box
    from; tool servers choose their placement separately."""

    model: str | None = None
    """Model id (None = the run's model, i.e. the policy under evaluation/training)."""
    client: ClientConfig | None = None
    """Endpoint override (None = the run's client)."""
    sampling: SamplingConfig | None = None
    """Sampling values merged onto the run's sampling."""

    max_turns: int | None = None
    """Max model turns per run (None = no limit)."""
    max_input_tokens: int | None = None
    """Max input tokens per run (None = no limit)."""
    max_output_tokens: int | None = None
    """Max output tokens per run (None = no limit)."""
    max_total_tokens: int | None = None
    """Max total tokens per run (None = no limit)."""

    timeout: TimeoutConfig = TimeoutConfig()
    retries: RetryConfig = RetryConfig()

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
