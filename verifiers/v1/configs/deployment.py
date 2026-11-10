"""Placement of an agent attempt's targets: the harness and its tools."""

from typing import Literal

from pydantic_config import BaseConfig


class DeploymentConfig(BaseConfig):
    mode: Literal["colocated"] = "colocated"
    """Harness and tools share the workspace."""
    trust_compose: bool = False
    """Permit task-authored Compose privileges and bind mounts on local Docker."""
    compose_mode: Literal["native", "vm"] = "native"
    """Prime Compose placement: native service VMs or Docker Compose inside one VM.
    Local Docker always uses Compose."""
