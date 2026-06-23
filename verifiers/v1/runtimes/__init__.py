"""Execution runtimes for harnesses.

Each runtime decides WHERE the program runs and HOW it reaches the host
interception server: subprocess (local), docker (local container), or prime /
modal (remote sandbox). They share the `Runtime` contract, so the Environment is
runtime-agnostic. `RuntimeConfig` is the discriminated config union and
`make_runtime` builds the runtime matching a config.
"""

from typing import Annotated

from pydantic import Field

from verifiers.v1.runtimes.base import (
    HOST,
    ProgramResult,
    Runtime,
    host_endpoint,
    reachable_url,
    register,
)
from verifiers.v1.runtimes.docker import DockerConfig, DockerRuntime
from verifiers.v1.runtimes.modal import ModalConfig, ModalRuntime
from verifiers.v1.runtimes.prime import PrimeConfig, PrimeRuntime
from verifiers.v1.runtimes.subprocess import SubprocessConfig, SubprocessRuntime

RuntimeConfig = Annotated[
    SubprocessConfig | DockerConfig | PrimeConfig | ModalConfig,
    Field(discriminator="type"),
]


def _runtime_cls(config: RuntimeConfig) -> type[Runtime]:
    if isinstance(config, PrimeConfig):
        return PrimeRuntime
    if isinstance(config, ModalConfig):
        return ModalRuntime
    if isinstance(config, DockerConfig):
        return DockerRuntime
    return SubprocessRuntime


def make_runtime(config: RuntimeConfig, name: str | None = None) -> Runtime:
    runtime = _runtime_cls(config)(config, name)
    register(runtime)
    return runtime


def runtime_is_local(config: RuntimeConfig) -> bool:
    """Whether this is a local runtime rather than a provider sandbox. Config-specific network
    restrictions do not change this topology; use `runtime_reaches_host_locally` for host URLs."""
    return _runtime_cls(config).is_local


def runtime_reaches_host_locally(config: RuntimeConfig) -> bool:
    """Whether this runtime config reaches a host service at localhost rather than by tunnel."""
    return _runtime_cls(config).config_reaches_host_locally(config)


__all__ = [
    "ProgramResult",
    "Runtime",
    "RuntimeConfig",
    "make_runtime",
    "runtime_is_local",
    "runtime_reaches_host_locally",
    "host_endpoint",
    "reachable_url",
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
