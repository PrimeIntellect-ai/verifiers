"""Execution runtimes for harnesses.

Each runtime decides WHERE the program runs and HOW it reaches the host
interception server: subprocess (local), docker (local container), or prime
(remote sandbox). They share the `Runtime` contract, so the Environment is
runtime-agnostic. `RuntimeConfig` is the discriminated config union and
`make_runtime` builds the runtime matching a config.
"""

from typing import Annotated

from pydantic import Field

from verifiers.v1.runtimes.base import ProgramResult, Runtime
from verifiers.v1.runtimes.docker import DockerConfig, DockerRuntime
from verifiers.v1.runtimes.prime import PrimeConfig, PrimeRuntime
from verifiers.v1.runtimes.subprocess import SubprocessConfig, SubprocessRuntime

RuntimeConfig = Annotated[
    SubprocessConfig | DockerConfig | PrimeConfig, Field(discriminator="type")
]


def make_runtime(config: RuntimeConfig) -> Runtime:
    if isinstance(config, PrimeConfig):
        return PrimeRuntime(config)
    if isinstance(config, DockerConfig):
        return DockerRuntime(config)
    return SubprocessRuntime(config)


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
]
