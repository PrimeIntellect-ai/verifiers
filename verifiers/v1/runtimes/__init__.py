"""Execution runtimes for harnesses.

Each runtime decides WHERE the program runs and HOW it reaches the host
interception server: subprocess (local), docker (local container), or prime /
modal (remote sandbox). They share the `Runtime` contract, so the Environment is
runtime-agnostic. `RuntimeConfig` is the discriminated config union and
`make_runtime` builds the runtime matching a config; `RuntimePool` reuses
`persistent` runtimes across rollouts.
"""

from verifiers.v1.runtimes.base import (
    HOST,
    BaseRuntimeConfig,
    ProgramResult,
    Runtime,
    host_endpoint,
    reachable_url,
    register,
)
from verifiers.v1.runtimes.docker import DockerConfig, DockerRuntime
from verifiers.v1.runtimes.factory import (
    RuntimeConfig,
    make_runtime,
    runtime_is_local,
)
from verifiers.v1.runtimes.modal import ModalConfig, ModalRuntime
from verifiers.v1.runtimes.pool import RuntimePool
from verifiers.v1.runtimes.prime import PrimeConfig, PrimeRuntime
from verifiers.v1.runtimes.subprocess import SubprocessConfig, SubprocessRuntime

__all__ = [
    "ProgramResult",
    "Runtime",
    "BaseRuntimeConfig",
    "RuntimeConfig",
    "RuntimePool",
    "make_runtime",
    "runtime_is_local",
    "host_endpoint",
    "reachable_url",
    "register",
    "HOST",
    "SubprocessConfig",
    "SubprocessRuntime",
    "DockerConfig",
    "DockerRuntime",
    "PrimeConfig",
    "PrimeRuntime",
    "ModalConfig",
    "ModalRuntime",
]
