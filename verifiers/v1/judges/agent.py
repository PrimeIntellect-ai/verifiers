"""A task-authored request for the environment's configured judge agent."""

from collections.abc import Callable
from dataclasses import dataclass

from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.configs.sampling import SamplingConfig
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace


@dataclass
class AgentJudgeRequest:
    task: Task
    model: str | None = None
    sampling: SamplingConfig | None = None
    timeout: float | None = None
    isolated: bool = False
    """Run in a fresh native runtime; task.setup restores its captured inputs."""
    network: NetworkPolicyConfig | None = None
    """Effective execution policy for a fresh workspace; None keeps its config."""
    on_trace: Callable[[Trace], None] | None = None
    """Observe every native attempt, including attempts replaced by retries."""
