"""Placement of an agent attempt's targets: the harness and its tools."""

from pydantic import Field
from pydantic_config import BaseConfig

from verifiers.v1.configs.runtime import PrimeConfig

HARNESS_IMAGE = "prime/primeintellect/verifiers-harness:v1"
"""Shared container image for split harness sandboxes (Python 3.14, Node 24, uv, curl, git, rg, tmux)."""

DOCKER_HARNESS_IMAGE = "localhost/verifiers-harness:v1"

HARNESS_DOCKERFILE = """\
FROM docker.io/library/python:3.14-bookworm
COPY --from=docker.io/library/node:24.22.3-bookworm /usr/local/bin/node /usr/local/bin/node
COPY --from=docker.io/library/node:24.22.3-bookworm /usr/local/lib/node_modules /usr/local/lib/node_modules
COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /uvx /usr/local/bin/
RUN ln -sf /usr/local/bin/node /usr/local/bin/nodejs \\
 && ln -sf /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \\
 && ln -sf /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \\
 && apt-get update -qq \\
 && apt-get install -y -qq --no-install-recommends curl ca-certificates git ripgrep tmux \\
 && rm -rf /var/lib/apt/lists/*
"""
"""Official-base Dockerfile used to build `HARNESS_IMAGE` on Prime and Docker."""


def _default_harness_runtime() -> PrimeConfig:
    cfg = PrimeConfig(image=HARNESS_IMAGE, cpu=2.0, memory=4.0)
    cfg.model_fields_set.clear()
    return cfg


class DeploymentConfig(BaseConfig):
    harness: PrimeConfig = Field(default_factory=_default_harness_runtime)
    """The harness's Prime VM. It shares the workspace's labels, region and creation
    rate unless set here, and always the task's network policy. Docker uses the
    evaluator's runtime configuration for its harness container."""
    multiplex: int = Field(8, ge=1, le=32)
    """Concurrent rollouts that share one pooled Prime harness VM (up to Prime's
    32-LAN-per-sandbox ceiling). N concurrent rollouts use ~N/multiplex harness VMs;
    1 = exclusive per active rollout."""
    trust_compose: bool = False
    """Permit task-authored Compose privileges and bind mounts on local Docker."""
