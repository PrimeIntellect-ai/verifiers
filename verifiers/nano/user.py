"""The user simulator for multi-turn rollouts.

A `User` supplies the scripted user turns that follow the first instruction — one
message per subsequent turn, an empty list meaning single-turn. These are passed
to the agent program up front (only the model endpoint is intercepted for now);
dynamic, model-driven users await a future user-interception endpoint.
"""

from typing import Generic, TypeVar

from pydantic_config import BaseConfig

from verifiers.nano.task import Task


class UserConfig(BaseConfig):
    pass


UserConfigT = TypeVar("UserConfigT", bound=UserConfig)


class User(Generic[UserConfigT]):
    def __init__(self, config: UserConfigT) -> None:
        self.config = config

    async def follow_ups(self, task: Task) -> list[str]:
        """The scripted user turns after the first; empty means single-turn."""
        return []
