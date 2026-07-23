"""One env agent's config: who plays the seat, and its per-run caps."""

from pathlib import Path

from pydantic import SerializeAsAny, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.clients import ClientConfig
from verifiers.v1.configs.harness import HarnessConfig, WireHarnessConfig
from verifiers.v1.configs.retries import RetryConfig
from verifiers.v1.runtimes import RuntimeConfig, SubprocessConfig
from verifiers.v1.types import SamplingConfig


class TimeoutConfig(BaseConfig):
    """Per-agent wall-clock timeouts per rollout stage, in seconds (None = no
    limit); each stage falls back to the task's own `TaskTimeout` when unset. An
    interaction's rollout budget is cumulative across its active harness segments
    and pauses while the caller computes the next user turn."""

    setup: float | None = None  # one shared budget: task setup + provisioning
    rollout: float | None = None
    finalize: float | None = None
    scoring: float | None = None


class AgentConfig(BaseConfig):
    """One env agent: who plays it, and its per-run caps. It pins only what
    makes it a different actor; everything unpinned falls back — the model context
    to the run's own, the harness to the taskset's default."""

    harness: SerializeAsAny[HarnessConfig] | None = None
    """The agent's program (None = the taskset's default harness)."""
    runtime: RuntimeConfig = SubprocessConfig()
    """Runtime for the harness program — the policy each run provisions its box
    from; tool servers choose their placement separately."""
    model: str | None = None
    """Model id (None = the run's model, i.e. the policy under evaluation/training)."""
    client: ClientConfig | None = None
    """Endpoint override (None = the run's client)."""
    sampling: SamplingConfig | None = None
    """Sampling override (None = the run's sampling)."""
    replay: Path | None = None
    """Play this seat from a finished run instead of sampling: a run dir (or its
    traces.jsonl). Each task returns that run's saved trace for this seat (a
    single-seat run matches regardless of name), re-stamped with this seat's
    standing — so the other seats iterate against fixed work. A replayed seat
    runs no model and leaves no runtime state: an env that inspects this seat's
    box can't replay it."""
    timeout: TimeoutConfig = TimeoutConfig()
    retries: RetryConfig = RetryConfig()
    """Whole-run retries: rerun this agent's rollout while its trace ends with a
    retryable error (never into a borrowed box)."""
    max_turns: int | None = None
    """Max model turns per run (None = no limit). Framework-enforced (the
    interception server refuses turns past it), so it applies to any harness."""
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_total_tokens: int | None = None
    """Token caps per run (None = no limit); framework-enforced between turns."""

    @model_validator(mode="before")
    @classmethod
    def _resolve_harness(cls, data):
        """Narrow a pinned `harness` to its concrete config type by `id` (absent
        stays None = the taskset's default). The lazy import keeps class-body
        `AgentConfig()` defaults constructible while this module initializes."""
        if isinstance(data, dict) and data.get("harness") is not None:
            from verifiers.v1.loaders import harness_config_type, narrow_plugin_field

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
