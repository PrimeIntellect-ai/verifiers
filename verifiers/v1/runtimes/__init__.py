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
    """Whether a runtime of this config exchanges host-local URLs without a public
    tunnel, read off the runtime class without provisioning one."""
    return _runtime_cls(config).is_local


__all__ = [
    "ProgramResult",
    "Runtime",
    "RuntimeConfig",
    "RuntimeInfo",
    "BaseRuntimeInfo",
    "make_runtime",
    "runtime_is_local",
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
