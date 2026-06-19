"""The runtime config union and the factory that builds a runtime from a config.

Split out from the package `__init__` so submodules (e.g. the runtime pool) can build runtimes
without importing the package itself (a cycle). `__init__` re-exports these.
"""

from typing import Annotated

from pydantic import Field

from verifiers.v1.runtimes.base import Runtime, register
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
    """Whether a runtime of this config shares the host network (so a program inside it reaches a
    host service at localhost, no tunnel) — read off the runtime class, without provisioning one.
    The interception pool / rollout use it to decide whether to tunnel their host port via
    `host_endpoint`."""
    return _runtime_cls(config).is_local
