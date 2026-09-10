"""Live boxes kept between `Agent.provision(..., reuse=key)` contexts — a cache the env
owns for the length of `serving()`. A box is a cache entry, never a correctness
dependency: a miss provisions a fresh box exactly as `provision_runtime` does, and every
box is torn down with the pool (or by the atexit backstop on a hard exit)."""

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self

from pydantic import Field
from pydantic_config import BaseConfig

from verifiers.v1.runtimes.base import Runtime

if TYPE_CHECKING:
    # Annotation-only: `RuntimeConfig` is the package's union, and the package
    # imports this module.
    from verifiers.v1.runtimes import RuntimeConfig

logger = logging.getLogger(__name__)


class RuntimePoolConfig(BaseConfig):
    """The idle budget of the boxes kept between `Agent.provision(task, reuse=key)` contexts."""

    ttl: float = Field(600, gt=0)
    """Seconds an idle box is kept after its `provision` context closes, then it is stopped."""
    max_idle: int | None = Field(16, ge=1)
    """Idle boxes kept at once (the oldest is stopped first); None = no cap. Leased boxes
    are bounded by the caller's own concurrency, not here."""


@dataclass
class _Idle:
    runtime: Runtime
    config: "RuntimeConfig"
    since: float


@dataclass
class _Gate:
    """A key's lock and the number of leases holding or waiting on it."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


class RuntimePool:
    """Live boxes kept between `Agent.provision(..., reuse=key)` contexts: one box per
    key, reused while its config still matches, it is alive, and its idle time is under
    `ttl`. A key is exclusive — a second `lease` of a leased key waits for the first to
    end, so a box never hosts two rollouts at once (leasing a key inside its own context
    therefore deadlocks). Idle boxes are held strongly, so `cleanup_at_exit` still frees
    them on a hard exit; `async with pool:` runs the TTL sweeper and stops every idle box
    on exit."""

    def __init__(self, config: RuntimePoolConfig | None = None) -> None:
        self.config = config or RuntimePoolConfig()
        self._idle: dict[str, _Idle] = {}
        self._locks: dict[str, _Gate] = {}
        self._closed = False
        self._sweeper: asyncio.Task[None] | None = None

    async def __aenter__(self) -> Self:
        self._sweeper = asyncio.create_task(self._sweep())
        return self

    async def __aexit__(self, *exc) -> None:
        await self.stop()

    async def stop(self) -> None:
        """Stop every idle box now; a lease still live stops its box on release. Idempotent."""
        self._closed = True
        if self._sweeper is not None:
            self._sweeper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._sweeper
            self._sweeper = None
        await self._evict(list(self._idle))

    async def discard(self, key: str) -> None:
        """Stop the box under `key` now (after its live lease, if any, ends); a no-op if none."""
        async with self._lock(key):
            await self._evict([key])

    @asynccontextmanager
    async def lease(
        self, key: str, config: "RuntimeConfig", env: Mapping[str, str]
    ) -> AsyncIterator[Runtime]:
        """The seam `Agent.provision` rides: the idle box under `key` when its config equals
        `config`, it is alive, and its idle time is under `ttl`, else a fresh box started
        from `config` (a stale one stopped first), with `env` as its environment. A normal
        exit parks the box idle under `key`; an exception (a cancellation included), a box
        the caller already `stop()`ped, or a closed pool tears it down instead — the
        outcome `provision_runtime`'s `finally: stop()` gives."""
        # Lazy: the package imports this module.
        from verifiers.v1.runtimes import make_runtime

        async with self._lock(key):
            runtime = await self._take(key, config, env)
            if runtime is None:
                runtime = make_runtime(config)
                runtime.env = dict(env)
                try:
                    await runtime.start()
                except BaseException:
                    await runtime.stop()
                    raise
            try:
                yield runtime
            except BaseException:
                await runtime.stop()
                raise
            if runtime.stopped:  # the caller tore it down itself: not kept
                return
            if self._closed:
                await runtime.stop()
                return
            self._idle[key] = _Idle(runtime, config, time.monotonic())
            await self._trim()

    @asynccontextmanager
    async def _lock(self, key: str) -> AsyncIterator[None]:
        """Exclusive (FIFO) use of `key`; its gate is dropped once no lease holds or waits on it."""
        gate = self._locks.setdefault(key, _Gate())
        gate.users += 1
        try:
            async with gate.lock:
                yield
        finally:
            gate.users -= 1
            if gate.users == 0:
                del self._locks[key]

    def _expired(self, entry: _Idle) -> bool:
        return time.monotonic() - entry.since >= self.config.ttl

    async def _take(
        self, key: str, config: "RuntimeConfig", env: Mapping[str, str]
    ) -> Runtime | None:
        """The idle box under `key` if it still fits (probed under `env`), else None (a
        stale one is stopped)."""
        entry = self._idle.pop(key, None)
        if entry is None:
            return None
        # The probe runs under this lease's env, not whatever the last one left behind.
        entry.runtime.env = dict(env)
        try:
            fits = (
                entry.config == config
                and not entry.runtime.stopped
                and not self._expired(entry)
                and await entry.runtime.alive()
            )
        except BaseException:  # a cancelled probe must not leak the popped box
            await entry.runtime.stop()
            raise
        if fits:
            logger.info("runtime pool: reusing box %s for %r", entry.runtime.name, key)
            return entry.runtime
        logger.info("runtime pool: replacing box %s for %r", entry.runtime.name, key)
        await entry.runtime.stop()
        return None

    async def _trim(self) -> None:
        """Stop the oldest idle boxes past `max_idle`."""
        cap = self.config.max_idle
        if cap is None or len(self._idle) <= cap:
            return
        oldest = sorted(self._idle, key=lambda key: self._idle[key].since)
        await self._evict(oldest[: len(self._idle) - cap])

    async def _evict(self, keys: list[str]) -> None:
        """Stop the idle boxes under `keys` (each popped first, so a concurrent lease
        provisions fresh instead of racing the teardown); unknown keys are skipped."""
        entries = [self._idle.pop(key) for key in keys if key in self._idle]
        results = await asyncio.gather(
            *(entry.runtime.stop() for entry in entries), return_exceptions=True
        )
        for entry, result in zip(entries, results):
            if isinstance(result, Exception):
                logger.warning(
                    "runtime pool: failed to stop box %s: %s",
                    entry.runtime.name,
                    result,
                )

    async def _sweep(self) -> None:
        """Stop expired idle boxes on a period; the platform's idle timeout is the backstop."""
        while True:
            await asyncio.sleep(min(self.config.ttl / 4, 30))
            expired = [key for key, entry in self._idle.items() if self._expired(entry)]
            await self._evict(expired)


__all__ = ["RuntimePool", "RuntimePoolConfig"]
