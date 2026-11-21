"""Deployment policy for task-authored services."""

from pydantic_config import BaseConfig

HARNESS_IMAGE = "prime/primeintellect/verifiers-harness:v1"
DOCKER_HARNESS_IMAGE = "ghcr.io/primeintellect-ai/verifiers-harness:v1"


class DeploymentConfig(BaseConfig):
    trust_compose: bool = False
    """Permit task-authored Compose privileges and bind mounts on local Docker."""
