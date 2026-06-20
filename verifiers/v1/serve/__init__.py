"""Serve a v1 environment over ZMQ.

`EnvServer` loads an environment once and runs rollouts on request by task index;
`EnvClient` is the matching ZMQ client. The orchestrator talks to the server and
never loads the environment itself — the env runtime is independent of it.
"""

from verifiers.v1.serve.client import EnvClient
from verifiers.v1.serve.pool import EnvServerPool, env_config_data, serve_env
from verifiers.v1.serve.server import EnvServer
from verifiers.v1.serve.types import (
    AdvantageBranch,
    HealthRequest,
    HealthResponse,
    InfoRequest,
    InfoResponse,
    ModelRuntimeConfig,
    RunAdvantagesRequest,
    RunAdvantagesResponse,
    RunGroupRequest,
    RunGroupResponse,
    RunRolloutRequest,
    RunRolloutResponse,
    TraceAdvantages,
)

__all__ = [
    "EnvServer",
    "EnvServerPool",
    "serve_env",
    "env_config_data",
    "EnvClient",
    "HealthRequest",
    "HealthResponse",
    "InfoRequest",
    "InfoResponse",
    "ModelRuntimeConfig",
    "RunRolloutRequest",
    "RunRolloutResponse",
    "RunGroupRequest",
    "RunGroupResponse",
    "RunAdvantagesRequest",
    "RunAdvantagesResponse",
    "TraceAdvantages",
    "AdvantageBranch",
]
