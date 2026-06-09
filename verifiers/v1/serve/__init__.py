"""Serve a v1 environment over ZMQ.

`EnvServer` loads an environment once and runs rollouts on request by task index;
`EnvClient` is the matching ZMQ client. The orchestrator talks to the server and
never loads the environment itself — the env runtime is independent of it.
"""

from verifiers.v1.serve.client import EnvClient
from verifiers.v1.serve.server import EnvServer
from verifiers.v1.serve.types import (
    HealthRequest,
    HealthResponse,
    InfoRequest,
    InfoResponse,
    RunGroupRequest,
    RunGroupResponse,
    RunRolloutRequest,
    RunRolloutResponse,
)

__all__ = [
    "EnvServer",
    "EnvClient",
    "HealthRequest",
    "HealthResponse",
    "InfoRequest",
    "InfoResponse",
    "RunRolloutRequest",
    "RunRolloutResponse",
    "RunGroupRequest",
    "RunGroupResponse",
]
