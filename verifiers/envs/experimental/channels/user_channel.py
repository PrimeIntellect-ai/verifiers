from __future__ import annotations

from collections.abc import Awaitable
from typing import TYPE_CHECKING, Protocol, TypeAlias

from verifiers.types import Messages, State

from verifiers.envs.experimental.channels.channel import (
    Channel,
    ChannelConfig,
    ChannelContext,
    single_config,
)

if TYPE_CHECKING:
    from verifiers.envs.experimental.resources import Resources
    from verifiers.envs.experimental.task import Task

UserResponse: TypeAlias = Messages | None


class User(Protocol):
    def respond(
        self, task: Task, state: State, resources: Resources
    ) -> UserResponse | Awaitable[UserResponse]: ...


def resolve_user(
    configs: list[ChannelConfig], context: ChannelContext
) -> dict[str, object]:
    config = single_config("user", configs)
    if config is None:
        return {}
    return {"user": config}


user_channel = Channel(
    name="user",
    outputs=("user",),
    resolve_fn=resolve_user,
)
