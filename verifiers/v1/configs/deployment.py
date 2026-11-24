"""Deployment policy and per-harness sandbox configuration."""

from pydantic import Field
from pydantic_config import BaseConfig

from verifiers.v1.configs.runtime import PrimeConfig

HARNESS_IMAGE_PREFIX = "prime/primeintellect/verifiers-harness"
DOCKER_HARNESS_IMAGE_PREFIX = "ghcr.io/primeintellect-ai/verifiers-harness"
HARNESS_IMAGE = f"{HARNESS_IMAGE_PREFIX}-base:v1"
DOCKER_HARNESS_IMAGE = f"{DOCKER_HARNESS_IMAGE_PREFIX}-base:v1"


def _default_harness_runtime() -> PrimeConfig:
    cfg = PrimeConfig(image=HARNESS_IMAGE, cpu=2.0, memory=4.0)
    cfg.model_fields_set.clear()
    return cfg


class DeploymentConfig(BaseConfig):
    harness: PrimeConfig = Field(default_factory=_default_harness_runtime)
    """The harness's Prime VM: defaults to the active harness's prepared image at
    2 CPU / 4 GB, inheriting the workspace's placement and the task's network policy."""
    trust_compose: bool = False
    """Permit task-authored Compose privileges and bind mounts on local Docker."""
