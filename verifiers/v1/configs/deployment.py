"""Placement of an agent attempt's targets: the harness and its tools."""

from typing import Literal

from pydantic_config import BaseConfig

from verifiers.v1.configs.runtime import PrimeConfig


class DeploymentConfig(BaseConfig):
    mode: Literal["colocated", "split"] = "colocated"
    """`colocated`: the harness runs in the task workspace with its tools. `split`: the
    harness runs on a Prime VM of its own on the rollout's private network, and its
    tools execute in the workspace, which never holds the model credentials."""
    harness: PrimeConfig = PrimeConfig()
    """The harness's VM in split mode. It shares the workspace's labels, region and
    creation rate unless set here, and always the task's network policy."""
    trust_compose: bool = False
    """Permit task-authored Compose privileges and bind mounts on local Docker."""


class ExecutionFailurePolicy(BaseConfig):
    """How a rollout ends when execution stops before the agent finishes: `error` fails
    it, `zero` keeps it as a valid sample with reward zero and skips grading."""

    workspace_loss: Literal["error", "zero"] = "error"
    """Policy for a workspace its provider reports gone while the agent runs, such as
    a VM the agent powered off. A loss attributed to the provider stays an error."""
    agent_timeout: Literal["grade", "error", "zero"] = "grade"
    """Policy for an expired agent deadline, which includes model latency; `grade`
    scores the workspace as the agent left it."""
