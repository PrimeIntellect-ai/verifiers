"""One env agent's config: who plays the seat, and its per-run caps."""

from typing import Any, Literal

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
from verifiers.v1.configs.retries import RetryConfig, RetryRule, StatusCode
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


class RolloutCheckpointConfig(BaseConfig):
    """Filesystem checkpoints of a rollout's box. When an attempt fails with a model
    error that its turn retries did not recover (`AgentConfig.turn_retries`), or its
    owner interrupts it, the rollout snapshots the box's filesystem before the box goes.
    A retry, or another process continuing the run, restores the snapshot: the trace
    keeps its turns up to there, and a new harness process continues in a new box with
    the snapshot's files, told that the box was restored. Background processes,
    sub-agents and in-memory state are gone. Needs a runtime and a harness that set
    `SUPPORTS_FILESYSTEM_CHECKPOINTS`; elsewhere the agent warns once and a retry starts
    over. A failure of the box itself (`SandboxError`) takes no checkpoint: its retry
    starts over in a new box."""

    after: Literal["delete", "keep"] = "delete"
    """What happens to the checkpoint when the rollout's run is final (it succeeded,
    or its retries ran out): `"delete"` deletes it once the box is gone; `"keep"`
    leaves it for its owner (forks, inspection), who must delete it."""
    boxless: bool = False
    """Also checkpoint a harness that runs no code (null) on a runtime without
    snapshots, from the trace alone."""


class MemoryCheckpointConfig(BaseConfig):
    """Memory checkpoints of a rollout's box: like `RolloutCheckpointConfig`, taken on
    the same failures, but the checkpoint also holds the box's processes and memory. A
    restore continues the harness process where it stopped, with its background
    processes, sub-agents and in-memory state; the harness does not notice. Needs a
    runtime and a harness that set `SUPPORTS_MEMORY_CHECKPOINTS`. A memory checkpoint
    that can no longer resume restores its filesystem instead, as a filesystem
    checkpoint does (prime: a memory checkpoint resumes once, on the node that took it).
    Takes precedence over `AgentConfig.checkpoint` where both apply."""

    after: Literal["delete", "keep"] = "delete"
    """As `RolloutCheckpointConfig.after`. Prime bills a kept memory checkpoint as a
    paused sandbox until its owner deletes it."""


class TurnRetryConfig(BaseConfig):
    """Retries of one model call, inside the interception: the harness waits on the
    call and never sees an error that a retry recovered. When the retries run out, the
    rollout checkpoints its box (`AgentConfig.checkpoint`, `.memory_checkpoint`) and
    fails the attempt with the error, for the rollout retries (`AgentConfig.retries`)
    to restore."""

    max_retries: int = Field(5, ge=0)
    """Retries per model call. Zero fails the attempt at the first error."""
    status_code: list[StatusCode] = Field(default_factory=lambda: [408, 429, "5xx"])
    """The provider statuses that retry; a transport failure or timeout counts as 502."""
    initial_delay: float = Field(1.0, gt=0)
    """Seconds before the first retry; each next retry waits twice as long."""
    max_delay: float = Field(30.0, gt=0)
    """Cap on the wait between two retries, in seconds."""


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
    retries: RetryConfig = Field(
        default_factory=lambda: RetryConfig(
            rules=[
                RetryRule(type="SandboxError", max_retries=2),
                RetryRule(type="ProviderError", max_retries=2),
            ]
        )
    )
    """Retries of the whole rollout. By default a failed box (`SandboxError`) starts
    over in a new box, and a model error that the turn retries did not recover
    (`ProviderError`) restores the checkpoint it took, if any."""
    turn_retries: TurnRetryConfig = TurnRetryConfig()
    checkpoint: RolloutCheckpointConfig | None = Field(
        default_factory=RolloutCheckpointConfig
    )
    """Filesystem checkpoints, on by default (`RolloutCheckpointConfig`). None turns
    them off."""
    memory_checkpoint: MemoryCheckpointConfig | None = None
    """Memory checkpoints, off by default (`MemoryCheckpointConfig`)."""

    network_notice: bool = True
    """Include the runtime's network restrictions in the agent's prompt."""

    @field_serializer("scoring")
    def _scoring_overrides(
        self, scoring: ScoringConfig, info: SerializationInfo
    ) -> dict:
        # Preserve unset fields across process and trace roundtrips so they
        # continue to inherit from the task instead of becoming explicit overrides.
        return scoring.model_dump(mode=info.mode, exclude_unset=True)

    @field_serializer("sampling")
    def _sampling_overrides(self, sampling: SamplingConfig | None) -> dict | None:
        # An unresolved agent carries only overrides across process boundaries.
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
