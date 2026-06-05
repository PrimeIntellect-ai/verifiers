"""The user simulator for multi-turn rollouts.

A `User` produces the next user message(s) given the transcript; returning an
empty list ends the conversation. Same API as v1's user-facing `User`, minus the
bindings/objects/artifacts/registry machinery.
"""

from typing import Generic, TypeVar

from pydantic_config import BaseConfig

from verifiers.v2.task import Task
from verifiers.v2.transcript import Transcript
from verifiers.v2.types import Messages, UserMessage


class UserConfig(BaseConfig):
    pass


UserConfigT = TypeVar("UserConfigT", bound=UserConfig)


class User(Generic[UserConfigT]):
    def __init__(self, config: UserConfigT) -> None:
        self.config = config

    async def get_response(
        self, task: Task, transcript: Transcript, messages: Messages
    ) -> list[UserMessage]:
        """Return the next user message(s); an empty list ends the conversation."""
        return []
