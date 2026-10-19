from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from pydantic import Field

from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.runtimes.apptainer import (
    ApptainerBackend,
    ApptainerConfig,
    ApptainerRuntimeInfo,
)
from verifiers.v1.runtimes.base import (
    BaseRuntimeInfo,
    ProgramResult,
    RuntimeBackend,
    RuntimeProcess,
    register,
)
from verifiers.v1.runtimes.docker import (
    DockerBackend,
    DockerConfig,
    DockerRuntimeInfo,
    PodmanBackend,
    PodmanConfig,
    PodmanRuntimeInfo,
)
from verifiers.v1.runtimes.e2b import E2BBackend, E2BConfig, E2BRuntimeInfo
from verifiers.v1.runtimes.modal import ModalBackend, ModalConfig, ModalRuntimeInfo
from verifiers.v1.runtimes.prime import (
    PrimeBackend,
    PrimeConfig,
    PrimeRuntimeInfo,
    set_base_sandbox_labels,
)
from verifiers.v1.runtimes.runtime import Runtime
from verifiers.v1.runtimes.subprocess import (
    SubprocessBackend,
    SubprocessConfig,
    SubprocessRuntimeInfo,
)

RuntimeConfig = Annotated[
    SubprocessConfig
    | DockerConfig
    | PodmanConfig
    | ApptainerConfig
    | PrimeConfig
    | ModalConfig
    | E2BConfig,
    Field(discriminator="type"),
]

RuntimeInfo = Annotated[
    SubprocessRuntimeInfo
    | DockerRuntimeInfo
    | PodmanRuntimeInfo
    | ApptainerRuntimeInfo
    | PrimeRuntimeInfo
    | ModalRuntimeInfo
    | E2BRuntimeInfo,
    Field(discriminator="type"),
]


def _backend_cls(config: RuntimeConfig) -> type[RuntimeBackend]:
    return {
        "subprocess": SubprocessBackend,
        "docker": DockerBackend,
        "podman": PodmanBackend,
        "apptainer": ApptainerBackend,
        "prime": PrimeBackend,
        "modal": ModalBackend,
        "e2b": E2BBackend,
    }[config.type]


def make_backend(config: RuntimeConfig, name: str | None = None) -> RuntimeBackend:
    backend = _backend_cls(config)(config, name)
    register(backend)
    return backend


@asynccontextmanager
async def provision_backend(
    config: RuntimeConfig,
    name: str | None = None,
    env: dict[str, str] | None = None,
) -> AsyncIterator[RuntimeBackend]:
    """Provision a box from `config` and tear it down on exit.

    `start()` sits inside the `try`: a failed start may already hold a paid sandbox, so
    it has to reach `stop()` (which is safe on a partially-started runtime)."""
    backend = make_backend(config, name)
    backend.env = dict(env or {})
    try:
        await backend.start()
        yield backend
    finally:
        await backend.stop()


@asynccontextmanager
async def provision_runtime(
    config: RuntimeConfig,
    name: str | None = None,
    env: dict[str, str] | None = None,
) -> AsyncIterator[Runtime]:
    """Lend an execution view; the context owns provisioning and cleanup."""
    async with provision_backend(config, name, env) as backend:
        yield backend.bind()


def runtime_is_local(config: RuntimeConfig) -> bool:
    """Whether a runtime of this config exchanges host-local URLs without a public
    tunnel, read off the runtime class without provisioning one."""
    return _backend_cls(config).is_local


__all__ = [
    "ApptainerBackend",
    "ApptainerConfig",
    "ApptainerRuntimeInfo",
    "BaseRuntimeInfo",
    "DockerBackend",
    "DockerConfig",
    "DockerRuntimeInfo",
    "E2BBackend",
    "E2BConfig",
    "E2BRuntimeInfo",
    "ModalBackend",
    "ModalConfig",
    "ModalRuntimeInfo",
    "NetworkPolicyConfig",
    "PodmanBackend",
    "PodmanConfig",
    "PodmanRuntimeInfo",
    "PrimeBackend",
    "PrimeConfig",
    "PrimeRuntimeInfo",
    "ProgramResult",
    "Runtime",
    "RuntimeBackend",
    "RuntimeConfig",
    "RuntimeInfo",
    "RuntimeProcess",
    "SubprocessBackend",
    "SubprocessConfig",
    "SubprocessRuntimeInfo",
    "make_backend",
    "provision_backend",
    "provision_runtime",
    "runtime_is_local",
    "set_base_sandbox_labels",
]
