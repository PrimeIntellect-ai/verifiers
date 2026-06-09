"""Execution runtimes for harnesses.

Each runtime decides WHERE the program runs and HOW it reaches the host
interception server: subprocess (local), docker (local container), or prime /
modal (remote sandbox). They share the `Runtime` contract, so the Environment is
runtime-agnostic. `RuntimeConfig` is the discriminated config union and
`make_runtime` builds the runtime matching a config.
"""

from typing import Annotated

from pydantic import Field

from verifiers.v1.runtimes.base import ProgramResult, Runtime, register
from verifiers.v1.runtimes.docker import DockerConfig, DockerRuntime
from verifiers.v1.runtimes.modal import ModalConfig, ModalRuntime
from verifiers.v1.runtimes.prime import PrimeConfig, PrimeRuntime
from verifiers.v1.runtimes.subprocess import SubprocessConfig, SubprocessRuntime

RuntimeConfig = Annotated[
    SubprocessConfig | DockerConfig | PrimeConfig | ModalConfig,
    Field(discriminator="type"),
]


def make_runtime(config: RuntimeConfig) -> Runtime:
    if isinstance(config, PrimeConfig):
        runtime: Runtime = PrimeRuntime(config)
    elif isinstance(config, ModalConfig):
        runtime = ModalRuntime(config)
    elif isinstance(config, DockerConfig):
        runtime = DockerRuntime(config)
    else:
        runtime = SubprocessRuntime(config)
    register(runtime)
    return runtime


__all__ = [
    "ProgramResult",
    "Runtime",
    "RuntimeConfig",
    "make_runtime",
    "SubprocessConfig",
    "SubprocessRuntime",
    "DockerConfig",
    "DockerRuntime",
    "PrimeConfig",
    "PrimeRuntime",
    "ModalConfig",
    "ModalRuntime",
]
