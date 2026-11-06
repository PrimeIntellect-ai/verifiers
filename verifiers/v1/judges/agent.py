"""A task-authored request for the environment's configured judge agent."""

from dataclasses import dataclass

from verifiers.v1.configs.sampling import SamplingConfig
from verifiers.v1.task import Task


@dataclass
class AgentJudgeRequest:
    task: Task
    model: str | None = None
    sampling: SamplingConfig | None = None
    timeout: float | None = None
