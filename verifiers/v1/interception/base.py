"""The interception contract: hand each rollout a slot on a host interception server.

Two shapes, picked by `InterceptionConfig` type (`make_interception`): `InterceptionPool` (prime)
grows servers — one behind its own tunnel per `multiplex` rollouts — to stay under the prime_tunnel
creation cap; a single `InterceptionServer` (custom) is one bring-your-own-endpoint server every
rollout shares. Both own their servers' lifecycle on an `AsyncExitStack` and expose one `acquire`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from verifiers.v1.interception.server import RolloutSession

# (base_url, secret): the interception server's reachable base URL for this rollout, and the bearer
# the harness/tool/user servers authenticate with. The harness reaches the model at `{base_url}/v1`;
# tool/user servers reach this rollout's shared state at `{base_url}/state` + `/task`. `base_url` is
# universally reachable — the interception is exposed (tunnel) whenever any consumer is remote.
Slot = tuple[str, str]


class Interception(ABC):
    """How an eval's rollouts reach the host interception server. Entered once for the run — it owns
    every server's lifecycle on `_stack` — and torn down on exit; each rollout `acquire`s a slot and
    frees it. Concrete shapes: `InterceptionPool` (prime, multiplexed) / `InterceptionServer` (one
    server, the custom case)."""

    def __init__(self) -> None:
        self._stack = AsyncExitStack()

    async def __aenter__(self) -> Self:
        await self._stack.__aenter__()
        return self

    async def __aexit__(self, *exc) -> None:
        # tears down every server (+ its tunnel) on `_stack`, LIFO, even if one teardown fails
        await self._stack.aclose()

    @abstractmethod
    def acquire(self, session: RolloutSession) -> AbstractAsyncContextManager[Slot]:
        """Register `session` on a server (bringing one up if needed) and yield its `Slot`; free it
        on exit."""
