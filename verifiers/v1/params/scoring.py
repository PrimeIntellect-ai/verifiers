"""What a rollout scores with: the agent's scoring overrides over the task's."""

from verifiers.v1.configs.retries import RetryConfig
from verifiers.v1.configs.scoring import ScoringConfig
from verifiers.v1.utils.generic import merge_defaults


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
