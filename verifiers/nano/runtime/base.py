"""The runtime contract: provision execution, run the program, tear down.

A runtime decides WHERE the program runs and HOW it reaches the host interception
server. Concrete runtimes live alongside this base; agents and the Environment
depend only on this contract, so they stay runtime-agnostic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ProgramResult:
    exit_code: int
    stdout: str
    stderr: str


class Runtime(ABC):
    @abstractmethod
    async def start(self, port: int) -> str:
        """Provision execution; return the base URL the program should use to
        reach the host interception server listening on `port`."""

    async def stop(self) -> None:
        """Tear down any provisioned resources. Default no-op."""

    @abstractmethod
    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        """Run `argv` (with the interception env vars `env`) to completion."""
