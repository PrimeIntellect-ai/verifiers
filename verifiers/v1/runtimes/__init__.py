"""Execution runtimes for harnesses."""

from typing import Annotated

from pydantic import Field

from verifiers.v1.runtimes.base import (
    BaseRuntimeInfo,
    ProgramResult,
    Runtime,
    register,
)
from verifiers.v1.runtimes.docker import DockerConfig, DockerRuntime, DockerRuntimeInfo
from verifiers.v1.runtimes.modal import ModalConfig, ModalRuntime, ModalRuntimeInfo
from verifiers.v1.runtimes.prime import PrimeConfig, PrimeRuntime, PrimeRuntimeInfo
from verifiers.v1.runtimes.subprocess import (
    SubprocessConfig,
    SubprocessRuntime,
    SubprocessRuntimeInfo,
)

RuntimeConfig = Annotated[
    SubprocessConfig | DockerConfig | PrimeConfig | ModalConfig,
    Field(discriminator="type"),
]

RuntimeInfo = Annotated[
    SubprocessRuntimeInfo | DockerRuntimeInfo | PrimeRuntimeInfo | ModalRuntimeInfo,
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
    """Whether this runtime runs locally rather than in a provider sandbox."""
    return _runtime_cls(config).is_local


def runtime_reaches_host_locally(config: RuntimeConfig) -> bool:
    """Whether this config reaches host services at localhost rather than through a tunnel."""
    return runtime_is_local(config) and getattr(config, "network_access", True)


__all__ = [
    "ProgramResult",
    "Runtime",
    "RuntimeConfig",
    "RuntimeInfo",
    "BaseRuntimeInfo",
    "make_runtime",
    "runtime_is_local",
    "runtime_reaches_host_locally",
    "SubprocessConfig",
    "SubprocessRuntime",
    "SubprocessRuntimeInfo",
    "DockerConfig",
    "DockerRuntime",
    "DockerRuntimeInfo",
    "PrimeConfig",
    "PrimeRuntime",
    "PrimeRuntimeInfo",
    "ModalConfig",
    "ModalRuntime",
    "ModalRuntimeInfo",
]
