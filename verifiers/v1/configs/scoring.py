"""Placement and retry policy for task scoring, independent of the agent program."""

from typing import Literal, Self

from pydantic import Field, SerializationInfo, field_serializer, field_validator
from pydantic_config import BaseConfig

from verifiers.v1.configs.retries import RetryConfig
from verifiers.v1.configs.runtime import NetworkPolicyConfig, RuntimeConfig
from verifiers.v1.utils.generic import merge_defaults

RewardRange = tuple[float | None, float | None]
"""Inclusive (low, high) reward bounds; None leaves a side unbounded."""

SKIPPED_REWARD_PREFIX = "skipped/"
"""Name prefix of the reward a trace gets when its task scoring is skipped; the skip
reason completes the name (`skipped/truncated`)."""


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
    """Skip task scoring for a truncated trace (`Trace.is_truncated`): it gets its
    skip reward (`skip_rewards`) in place of task rewards."""
    skip_on_timeout: bool = False
    """Skip task scoring for a trace whose stage deadline expired before scoring
    (`Trace.is_timeout`): it gets its skip reward (`skip_rewards`) in place of task
    rewards."""
    reward_range: RewardRange = (0.0, 1.0)
    """Inclusive bounds of a trace's total reward (`Trace.reward`), env-recorded
    rewards included. None leaves a side unbounded. A finished episode with a reward
    outside the range fails with an `EnvError`."""
    pass_threshold: float | None = None
    """Lowest reward that counts as a pass (`Trace.passed`). None uses the range's
    upper bound; with an unbounded upper bound, a pass needs an explicit threshold."""
    skip_rewards: dict[str, float] = Field(default_factory=dict)
    """Reward for a trace whose task scoring is skipped, by skip reason:
    `format_violation`, `truncated`, `timeout`, or a `skip_scoring` stop's name. An
    unlisted reason gets the range's lower bound. Each value must lie inside the
    range and below the pass threshold."""

    @field_validator("reward_range")
    @classmethod
    def check_reward_range(cls, value: RewardRange) -> RewardRange:
        lo, hi = value
        if lo is not None and hi is not None and lo >= hi:
            raise ValueError(f"reward_range low {lo} must be below its high {hi}")
        return value

    @property
    def threshold(self) -> float | None:
        """The resolved pass threshold: `pass_threshold`, else the range's high."""
        return (
            self.pass_threshold
            if self.pass_threshold is not None
            else self.reward_range[1]
        )

    def skip_reward(self, reason: str) -> float:
        """The reward for a trace whose task scoring was skipped for `reason`."""
        if reason in self.skip_rewards:
            return self.skip_rewards[reason]
        lo = self.reward_range[0]
        if lo is None:
            raise ValueError(
                f"scoring was skipped ({reason}) under a reward_range without a low "
                f"bound; set scoring.skip_rewards[{reason!r}]"
            )
        return lo

    def check_rewards(self) -> Self:
        """Check the pass threshold and skip rewards against the reward range.

        Call on a resolved policy: task defaults and agent overrides each set only
        part of it, so a partial config can fail checks the merged one passes."""
        lo, hi = self.reward_range
        threshold = self.pass_threshold
        if threshold is not None and (
            (lo is not None and threshold <= lo) or (hi is not None and threshold > hi)
        ):
            raise ValueError(
                f"pass_threshold {threshold} must lie in ({lo}, {hi}] (reward_range)"
            )
        required = [
            reason
            for reason, enabled in (
                ("truncated", self.skip_on_truncated),
                ("timeout", self.skip_on_timeout),
            )
            if enabled and lo is None
        ]
        if missing := [r for r in required if r not in self.skip_rewards]:
            raise ValueError(
                f"reward_range has no low bound, so skipped scoring needs an explicit "
                f"reward: set scoring.skip_rewards for {missing}"
            )
        for reason, value in self.skip_rewards.items():
            if (lo is not None and value < lo) or (hi is not None and value > hi):
                raise ValueError(
                    f"skip_rewards[{reason!r}] = {value} is outside reward_range "
                    f"[{lo}, {hi}]; widen the range to include it"
                )
            if self.threshold is not None and value >= self.threshold:
                raise ValueError(
                    f"skip_rewards[{reason!r}] = {value} would count as a pass "
                    f"(pass threshold {self.threshold})"
                )
        return self

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


def weighted_reward_range(*terms: tuple[float, RewardRange]) -> RewardRange:
    """The range of `sum(weight * reward)` over `(weight, range)` terms, for an env
    that rescales or adds rewards on a trace and restates its `reward_range`."""
    lo: float | None = 0.0
    hi: float | None = 0.0
    for weight, (term_lo, term_hi) in terms:
        if weight == 0:
            continue
        a = None if term_lo is None else term_lo * weight
        b = None if term_hi is None else term_hi * weight
        if weight < 0:
            a, b = b, a
        lo = None if lo is None or a is None else lo + a
        hi = None if hi is None or b is None else hi + b
    return lo, hi


def resolve_scoring(task: ScoringConfig, agent: ScoringConfig) -> ScoringConfig:
    """Apply explicitly configured agent fields over the task's scoring defaults.

    Runtime overrides replace the task runtime. Environment entries and skip rewards
    merge by key. The config defaults supply shared placement and two retries when
    neither sets them. Raises when the merged reward policy is inconsistent.
    """
    updates = {name: getattr(agent, name) for name in agent.model_fields_set}
    updates["env"] = {**(task.env or {}), **(agent.env or {})}
    updates["skip_rewards"] = {**task.skip_rewards, **agent.skip_rewards}
    updates["retries"] = RetryConfig.model_validate(
        merge_defaults(
            task.retries,
            agent.retries.model_dump(exclude_unset=True)
            if "retries" in agent.model_fields_set
            else {},
        )
    )
    return task.model_copy(update=updates).check_rewards()
