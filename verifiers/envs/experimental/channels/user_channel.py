from __future__ import annotations

from collections.abc import Awaitable
from typing import TYPE_CHECKING, Protocol, TypeAlias, runtime_checkable

from verifiers.types import Messages, State

from verifiers.envs.experimental.channels.channel import (
    Channel,
    ChannelConfig,
    ChannelContext,
    ResourcePatch,
    single_config,
)

if TYPE_CHECKING:
    from verifiers.envs.experimental.resources import Resources
    from verifiers.envs.experimental.task import Task

UserResponse: TypeAlias = Messages | None


@runtime_checkable
class User(Protocol):
    def respond(
        self, task: Task, state: State, resources: Resources
    ) -> UserResponse | Awaitable[UserResponse]: ...


def resolve_user(
    configs: list[ChannelConfig], context: ChannelContext
) -> ResourcePatch:
    config = single_config("user", configs)
    if config is None:
        return ResourcePatch()
    return ResourcePatch(objects={"user": config})


user_channel = Channel(
    name="user",
    outputs={"user": User},
    resolve_fn=resolve_user,
)
