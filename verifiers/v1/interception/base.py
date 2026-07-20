"""The interception contract: hand each rollout a slot on a host interception server.

Three shapes, picked by `InterceptionConfig` type (see `make_interception`): a single
`InterceptionServer`, a fixed `StaticInterceptionPool`, or an on-demand
`ElasticInterceptionPool`. From the outside they behave the same: start/stop (or the async
context manager wrapping them) bound the lifecycle — one eval, one topology run, or one
agent `.run` — and each rollout `acquire`s a slot and frees it. An `Interception` can be
shared: whoever entered it owns the lifecycle; borrowers only `acquire`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

from pydantic_config import BaseConfig

if TYPE_CHECKING:
    from verifiers.v1.session import RolloutSession


class BaseInterceptionConfig(BaseConfig):
    """Base for the interception types — the discriminated union's common type. Per-type
    fields live on the subclasses (server's `tunnel`, static's `servers`, elastic's
    `multiplex`)."""


@dataclass(frozen=True)
class Slot:
    """One rollout's interception registration.

    `base_url` is universally reachable and `secret` authenticates the harness/tool/user
    servers. `cancel` closes admission before cancelling slot handlers, so a rollout timeout
    cannot race a newly accepted request.
    """

    base_url: str
    secret: str
    cancel: Callable[[], Awaitable[None]]


class Interception(ABC):
    """How rollouts reach the host interception server. `start` brings the servers up (or
    arms lazy growth), `stop` tears every server (+ its tunnel) down via `stack` — LIFO,
    even if one teardown fails; `async with` wraps the two. Each rollout `acquire`s a slot
    and frees it on exit."""

    def __init__(self) -> None:
        self.stack = AsyncExitStack()

    @abstractmethod
    async def start(self) -> None:
        """Bring the interception up; resources land on `stack` so `stop` frees them."""

    async def stop(self) -> None:
        await self.stack.aclose()

    async def __aenter__(self) -> Self:
        try:
            await self.start()
        except BaseException:
            # unwind whatever `start` already put on the stack
            await self.stop()
            raise
        return self

    async def __aexit__(self, *exc) -> None:
        await self.stop()

    @abstractmethod
    def acquire(self, session: RolloutSession) -> AbstractAsyncContextManager[Slot]:
        """Register `session` on a server (bringing one up if needed) and yield its `Slot`;
        free it on exit."""
