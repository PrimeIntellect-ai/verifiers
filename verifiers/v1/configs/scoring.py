"""Placement and retry policy for task scoring, independent of the agent program."""

from typing import Literal

from pydantic import SerializationInfo, field_serializer
from pydantic_config import BaseConfig

from verifiers.v1.configs.retries import RetryConfig
from verifiers.v1.configs.runtime import NetworkPolicyConfig, RuntimeConfig
from verifiers.v1.utils.generic import merge_defaults


class ScoringConfig(BaseConfig):
    mode: Literal["shared", "isolated"] = "shared"
    """Score in the workspace or restore into a fresh runtime."""
    runtime: RuntimeConfig | None = None
    """Fresh verifier provider/configuration. None uses the agent's provider policy."""
    env: dict[str, str] | None = None
    """Scoring-only variables, merged over task defaults and resolved at scoring time."""
    retries: RetryConfig = RetryConfig(max_retries=2)
    """Retry rules for fresh isolated scoring attempts; shared scoring is not retried."""
    network: NetworkPolicyConfig | None = None
    """The task's network policy while scoring runs, in place of `TaskData.network` and
    combined with the runtime's own rules the same way. None keeps the default: isolated
    scoring runs under `TaskData.network`, shared scoring keeps the agent phase's network,
    and Harbor's shared grading lifts the policy."""
    skip_on_truncated: bool = False
    """Skip task scoring for a truncated trace (`Trace.is_truncated`): it keeps no
    rewards, so its reward is 0."""
    skip_on_timeout: bool = False
    """Skip task scoring for a trace whose stage deadline expired before scoring
    (`Trace.is_timeout`): it keeps no rewards, so its reward is 0."""

    @field_serializer("runtime")
    def serialize_runtime(
        self, runtime: RuntimeConfig | None, info: SerializationInfo
    ) -> dict | None:
        if runtime is None:
            return None
        values = runtime.model_dump(mode=info.mode)
        # verifier_runtime treats an explicitly supplied image as an override of
        # the task's image. Dumping a provider's implicit default would mark it
        # explicit on reload, changing which image scoring uses after transport.
        # Keep explicit values even when they equal the provider default.
        if "image" not in runtime.model_fields_set:
            values.pop("image", None)
        return values


def resolve_scoring(task: ScoringConfig, agent: ScoringConfig) -> ScoringConfig:
    """Apply explicitly configured agent fields over the task's scoring defaults.

    Runtime overrides replace the task runtime. Environment entries merge by key.
    The config defaults supply shared placement and two retries when neither sets them.
    """
    updates = {name: getattr(agent, name) for name in agent.model_fields_set}
    updates["env"] = {**(task.env or {}), **(agent.env or {})}
    updates["retries"] = RetryConfig.model_validate(
        merge_defaults(
            task.retries,
            agent.retries.model_dump(exclude_unset=True)
            if "retries" in agent.model_fields_set
            else {},
        )
    )
    return task.model_copy(update=updates)
