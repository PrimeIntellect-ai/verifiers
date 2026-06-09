"""The client abstraction: turn a prompt into a `Response`.

Collapsed from v1's 4-typevar generic ABC with five conversion hooks to a single
abstract method. Each concrete client owns its own wire translation internally.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from verifiers.v1.types import Messages, Response, SamplingConfig, Tool


class Client(ABC):
    @abstractmethod
    async def get_response(
        self,
        prompt: Messages,
        model: str,
        sampling_args: SamplingConfig,
        tools: list[Tool] | None = None,
    ) -> Response:
        """Run one completion, translating to/from this client's wire format."""

    async def close(self) -> None:
        """Release any underlying resources. Default no-op."""


@dataclass(frozen=True)
class RolloutContext:
    """The collaborators a single rollout needs (client + model + sampling), bundled
    so harnesses hold no rollout state. Built by the Environment."""

    client: Client
    model: str
    sampling: SamplingConfig
