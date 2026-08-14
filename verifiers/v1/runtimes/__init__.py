from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from pydantic import Field

from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.runtimes.base import (
    BaseRuntimeInfo,
    ProgramResult,
    Runtime,
    RuntimeProcess,
    register,
)
from verifiers.v1.runtimes.docker import DockerConfig, DockerRuntime, DockerRuntimeInfo
from verifiers.v1.runtimes.modal import ModalConfig, ModalRuntime, ModalRuntimeInfo
from verifiers.v1.runtimes.prime import (
    PrimeConfig,
    PrimeRuntime,
    PrimeRuntimeInfo,
    set_base_sandbox_labels,
)
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


def make_runtime(
    config: RuntimeConfig,
    name: str | None = None,
    env: dict[str, str] | None = None,
) -> Runtime:
    runtime = _runtime_cls(config)(config, name)
    runtime.env = dict(env or {})
    register(runtime)
    return runtime


@asynccontextmanager
async def provision_runtime(
    config: RuntimeConfig,
    name: str | None = None,
    env: dict[str, str] | None = None,
) -> AsyncIterator[Runtime]:
    """Provision a box from `config` and tear it down on exit.

    `start()` sits inside the `try`: a failed start may already hold a paid sandbox, so
    it has to reach `stop()` (which is safe on a partially-started runtime)."""
    runtime = make_runtime(config, name, env)
    try:
        await runtime.start()
        yield runtime
    finally:
        await runtime.stop()


def runtime_is_local(config: RuntimeConfig) -> bool:
    """Whether a runtime of this config exchanges host-local URLs without a public
    tunnel, read off the runtime class without provisioning one."""
    return _runtime_cls(config).is_local


__all__ = [
    "BaseRuntimeInfo",
    "DockerConfig",
    "DockerRuntime",
    "DockerRuntimeInfo",
    "ModalConfig",
    "ModalRuntime",
    "ModalRuntimeInfo",
    "NetworkPolicyConfig",
    "PrimeConfig",
    "PrimeRuntime",
    "PrimeRuntimeInfo",
    "ProgramResult",
    "Runtime",
    "RuntimeConfig",
    "RuntimeInfo",
    "RuntimeProcess",
    "SubprocessConfig",
    "SubprocessRuntime",
    "SubprocessRuntimeInfo",
    "make_runtime",
    "provision_runtime",
    "runtime_is_local",
    "set_base_sandbox_labels",
]
