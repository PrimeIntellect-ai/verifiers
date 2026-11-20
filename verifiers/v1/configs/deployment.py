"""Placement of an agent attempt's targets: the harness and its tools."""

from pydantic import Field
from pydantic_config import BaseConfig

from verifiers.v1.configs.runtime import PrimeConfig


class DeploymentConfig(BaseConfig):
    harness: PrimeConfig = PrimeConfig(cpu=2.0, memory=4.0)
    """The harness's Prime VM. It shares the workspace's labels, region and creation
    rate unless set here, and always the task's network policy. Docker uses the
    evaluator's runtime configuration for its harness container."""
    multiplex: int = Field(8, ge=1, le=32)
    """Concurrent rollouts that share one pooled Prime harness VM (up to Prime's
    32-LAN-per-sandbox ceiling). N concurrent rollouts use ~N/multiplex harness VMs;
    1 = exclusive per active rollout."""
    trust_compose: bool = False
    """Permit task-authored Compose privileges and bind mounts on local Docker."""
