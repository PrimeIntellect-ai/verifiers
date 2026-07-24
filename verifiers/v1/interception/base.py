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
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from typing import TYPE_CHECKING, Self

from pydantic_config import BaseConfig

if TYPE_CHECKING:
    from verifiers.v1.session import RolloutSession


class BaseInterceptionConfig(BaseConfig):
    """Base for the interception types — the discriminated union's common type. Per-type
    fields live on the subclasses (server's `tunnel`, static's `servers`, elastic's
    `multiplex`)."""


# (base_url, secret): the interception server's reachable base URL for this rollout, and the
# bearer the harness/tool servers authenticate with. The harness reaches the model at
# `{base_url}/v1`; tool servers reach this rollout's shared state at `{base_url}/state`
# + `/task`. `base_url` is universally reachable — the interception is exposed (tunnel)
# whenever any consumer is remote.
Slot = tuple[str, str]


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
