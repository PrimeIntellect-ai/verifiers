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
