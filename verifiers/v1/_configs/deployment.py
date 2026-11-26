"""Placement of an agent attempt's targets: the harness and its tools."""

from typing import Literal

from pydantic_config import BaseConfig


class DeploymentConfig(BaseConfig):
    """Placement of the harness and its tools."""

    mode: Literal["colocated"] = "colocated"
    """Harness and tools share the workspace."""
    trust_compose: bool = False
    """Permit task-authored Compose privileges and bind mounts on local Docker."""
