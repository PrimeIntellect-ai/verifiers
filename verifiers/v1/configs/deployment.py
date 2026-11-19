"""Placement of an agent attempt's targets: the harness and its tools."""

from typing import Literal

from pydantic import Field
from pydantic_config import BaseConfig

from verifiers.v1.configs.runtime import PrimeConfig


class DeploymentConfig(BaseConfig):
    mode: Literal["colocated", "split"] = "colocated"
    """In split mode the harness and task workspace use separate sandboxes."""
    harness: PrimeConfig = PrimeConfig()
    """The harness's Prime VM in split mode. It shares the workspace's labels,
    region and creation rate unless set here, and always the task's network policy.
    Docker uses the evaluator's runtime configuration for its harness container."""
    multiplex: int = Field(8, ge=1, le=32)
    """Concurrent rollouts that share one pooled Prime harness VM (up to Prime's
    32-LAN-per-sandbox ceiling). N concurrent rollouts use ~N/multiplex harness VMs;
    1 = exclusive per active rollout."""
    trust_compose: bool = False
    """Permit task-authored Compose privileges and bind mounts on local Docker."""
