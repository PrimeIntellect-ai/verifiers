"""What a `Rollout` is built from: one agent on one task, fully resolved."""

from __future__ import annotations

from dataclasses import dataclass

from verifiers.v1.configs.agent import TimeoutConfig
from verifiers.v1.configs.runtime import RuntimeConfig
from verifiers.v1.configs.scoring import ScoringConfig
from verifiers.v1.params.agent import AgentParams
from verifiers.v1.params.harness import HarnessParams
from verifiers.v1.task import TaskTimeout


@dataclass(frozen=True)
class RolloutTimeouts:
    """Per-stage rollout timeouts, each bounding one rollout stage."""

    setup: float | None = None
    """Timeout (in seconds) for task/harness setup through session preparation."""
    agent: float | None = None
    """Timeout (in seconds) for the agent's solve attempt."""
    finalize: float | None = None
    """Timeout (in seconds) for the task + harness finalize hooks."""
    scoring: float | None = None
    """Combined budget (in seconds) for harness metrics and task scoring, including
    isolated verifier setup and retries. Finalization has its own budget."""

    @classmethod
    def resolve(cls, agent: TimeoutConfig, task: TaskTimeout) -> RolloutTimeouts:
        """Each stage takes the agent's timeout, else the task's."""

        def pick(own: float | None, default: float | None) -> float | None:
            return own if own is not None else default

        return cls(
            setup=pick(agent.setup, task.setup),
            agent=pick(agent.rollout, task.agent),
            finalize=pick(agent.finalize, task.finalize),
            scoring=pick(agent.scoring, task.scoring),
        )


@dataclass(frozen=True)
class RolloutLimits:
    """Per-rollout framework limits (None = no cap), checked before each turn is served.
    The first limit reached refuses the turn — halting any harness, the same mechanism as
    a @stop — and becomes the trace's stop condition. Each caps a trace computed property:
    `max_turns` -> num_turns, `max_input_tokens` -> num_input_tokens, `max_output_tokens` ->
    num_output_tokens, `max_total_tokens` -> num_total_tokens. Token caps are soft by one turn:
    they're checked between turns, so the turn that crosses a cap still completes.
    `max_turn_tokens` caps each response (applied through sampling) and refuses the turn
    after a response that hit it."""

    max_turns: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_total_tokens: int | None = None
    max_turn_tokens: int | None = None

    @classmethod
    def resolve(cls, agent: AgentParams) -> RolloutLimits:
        return cls(
            max_turns=agent.max_turns,
            max_input_tokens=agent.max_input_tokens,
            max_output_tokens=agent.max_output_tokens,
            max_total_tokens=agent.max_total_tokens,
            max_turn_tokens=agent.max_turn_tokens,
        )


@dataclass(frozen=True)
class RolloutParams:
    """Everything one rollout reads, resolved for its task. Live resources it borrows
    (interception, tool servers, a borrowed box) are passed beside it."""

    agent: AgentParams
    harness: HarnessParams
    runtime: RuntimeConfig
    """The box's config: the agent's runtime policy resolved for the task, or the
    borrowed box's own."""
    scoring: ScoringConfig
    """The agent's scoring overrides resolved over the task's defaults."""
    timeouts: RolloutTimeouts
    limits: RolloutLimits
