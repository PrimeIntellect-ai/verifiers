from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from pydantic import Field

from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.runtimes.backends.custom import (
    CustomRuntimeConfig,
    CustomRuntimeInfo,
    import_backend,
)
from verifiers.v1.runtimes.backends.docker import (
    DockerBackend,
    DockerConfig,
    DockerRuntimeInfo,
)
from verifiers.v1.runtimes.backends.e2b import E2BBackend, E2BConfig, E2BRuntimeInfo
from verifiers.v1.runtimes.backends.modal import (
    ModalBackend,
    ModalConfig,
    ModalRuntimeInfo,
)
from verifiers.v1.runtimes.backends.prime import (
    PrimeBackend,
    PrimeConfig,
    PrimeRuntimeInfo,
    set_base_sandbox_labels,
)
from verifiers.v1.runtimes.backends.subprocess import (
    SubprocessBackend,
    SubprocessConfig,
    SubprocessRuntimeInfo,
)
from verifiers.v1.runtimes.base import (
    BaseRuntimeInfo,
    ProgramResult,
    RuntimeBackend,
    RuntimeProcess,
    register,
)
from verifiers.v1.runtimes.runtime import Runtime

RuntimeConfig = Annotated[
    SubprocessConfig
    | DockerConfig
    | PrimeConfig
    | ModalConfig
    | E2BConfig
    | CustomRuntimeConfig,
    Field(discriminator="type"),
]

RuntimeInfo = Annotated[
    SubprocessRuntimeInfo
    | DockerRuntimeInfo
    | PrimeRuntimeInfo
    | ModalRuntimeInfo
    | E2BRuntimeInfo
    | CustomRuntimeInfo,
    Field(discriminator="type"),
]


def backend_cls(config: RuntimeConfig) -> type[RuntimeBackend]:
    """The backend class for `config`, whose class attributes declare its capabilities."""
    if isinstance(config, CustomRuntimeConfig):
        return import_backend(config.import_path)
    return {
        "subprocess": SubprocessBackend,
        "docker": DockerBackend,
        "prime": PrimeBackend,
        "modal": ModalBackend,
        "e2b": E2BBackend,
    }[config.type]


def make_backend(config: RuntimeConfig, name: str | None = None) -> RuntimeBackend:
    backend = backend_cls(config)(config, name)
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
    return backend_cls(config).is_local


__all__ = [
    "BaseRuntimeInfo",
    "CustomRuntimeConfig",
    "CustomRuntimeInfo",
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
    "backend_cls",
    "make_backend",
    "provision_backend",
    "provision_runtime",
    "runtime_is_local",
    "set_base_sandbox_labels",
]
