"""Multiplexed pool of split Prime harness VMs shared across concurrent rollouts."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import shlex
from collections import Counter, defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from verifiers.v1.configs.runtime import PrimeConfig
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import make_backend
from verifiers.v1.runtimes.base import register
from verifiers.v1.utils.aio import run_shielded
from verifiers.v1.utils.logging import clip, get_logger, isolated

if TYPE_CHECKING:
    from verifiers.v1.runtimes.backends.prime import PrimeBackend

logger = get_logger(__name__)

_current_pool: ContextVar[HarnessPool | None] = ContextVar(
    "vf_harness_pool", default=None
)


def current_harness_pool() -> HarnessPool | None:
    return _current_pool.get()


def _pool_key(config: PrimeConfig) -> str:
    canonical = config.model_copy(
        update={
            "workdir": "/app",
            "labels": [],
            "creates_per_min": None,
        }
    )
    return json.dumps(canonical.model_dump(mode="json"), sort_keys=True)


@dataclass
class _PooledHarness:
    key: str
    backend: PrimeBackend
    boot_task: asyncio.Task[None]
    load: int = 0
    healthy: bool = True
    workdir_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    egress_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    route_counts: Counter[str] = field(default_factory=Counter)
    applied_sig: tuple | None = None


class HarnessLease:
    """One rollout's concurrent slot on a pooled `_PooledHarness` (`VM A`)."""

    def __init__(self, pool: HarnessPool, entry: _PooledHarness, workdir: str) -> None:
        self.pool = pool
        self.entry = entry
        self.backend = entry.backend
        self.workdir = workdir or "/app"
        self._lan_task: asyncio.Task[None] | None = None
        self._registered_hosts: list[str] = []
        self._released = False

    async def start(
        self, network_source: asyncio.Task[str] | str | None, tags: list[str]
    ) -> None:
        """Wait for the shared `VM A` boot, enroll this rollout's LAN, and ensure `workdir`."""
        from verifiers.v1.runtimes.backends.prime import join_lan

        try:
            await asyncio.shield(self.entry.boot_task)
        except BaseException:
            self.entry.healthy = False
            raise

        if network_source is not None:

            async def enroll() -> None:
                network_id = (
                    network_source
                    if isinstance(network_source, str)
                    else await asyncio.shield(network_source)
                )
                await join_lan(self.backend, network_id, tags)

            self._lan_task = asyncio.create_task(enroll())

        tasks = []
        if self.workdir not in self.backend._created_workdirs:

            async def ensure_workdir() -> None:
                async with self.entry.workdir_lock:
                    if self.workdir in self.backend._created_workdirs:
                        return
                    await self.backend.bind().run(
                        ["sh", "-c", f"mkdir -p {shlex.quote(self.workdir)}"], {}
                    )
                    self.backend._created_workdirs.add(self.workdir)

            tasks.append(ensure_workdir())

        if (
            self.entry.load == 1
            and self.entry.applied_sig is not None
            and not self.backend.cache.mcp_sources
            and self.backend._egress_enforced
        ):

            async def reopen_for_setup() -> None:
                async with self.entry.egress_lock:
                    if self.entry.load == 1 and self.backend._egress_enforced:
                        await self.backend.prepare_execution(None)
                        self.entry.applied_sig = None

            tasks.append(reopen_for_setup())

        if tasks:
            await asyncio.gather(*tasks)

    async def wait_lan_ready(self) -> None:
        if self._lan_task is not None:
            try:
                await asyncio.shield(self._lan_task)
            except Exception as e:
                raise SandboxError(f"prime LAN enrollment failed: {e}") from e

    async def prepare_execution(self, routes: list[str] | None) -> None:
        hosts = list(
            dict.fromkeys(
                h for h in (urlsplit(route).hostname for route in (routes or ())) if h
            )
        )
        async with self.entry.egress_lock:
            if not self._registered_hosts and hosts:
                self._registered_hosts = hosts
                for host in hosts:
                    self.entry.route_counts[host] += 1
            if not self.backend.network_restricted:
                return
            active_hosts = tuple(sorted(self.entry.route_counts))
            sig = (
                active_hosts if self.backend.config.allow != ["*"] else (),
                tuple(self.backend.config.allow),
                tuple(self.backend.config.block),
            )
            if self.entry.applied_sig == sig:
                return
            union_routes = [f"https://{host}" for host in active_hosts]
            await self.backend.prepare_execution(union_routes)
            self.entry.applied_sig = sig

    async def release(self, network_id: str | None, *, healthy: bool = True) -> None:
        if self._released:
            return
        self._released = True
        from verifiers.v1.runtimes.controller import current_job

        if self._lan_task is not None and not self._lan_task.done():
            self._lan_task.cancel()
            await asyncio.gather(self._lan_task, return_exceptions=True)
        self._lan_task = None

        detached = True
        if (
            network_id
            and self.backend._client is not None
            and self.backend.info.id is not None
            and not self.backend.stopped
        ):
            try:
                if job := current_job.get():
                    await job.launcher.call(
                        "leave_network",
                        network_id=network_id,
                        sandbox_id=self.backend.info.id,
                    )
                else:
                    await self.backend._client.remove_lan_member(
                        network_id, self.backend.info.id
                    )
            except Exception as error:
                detached = False
                logger.warning(
                    "Failed to detach warm prime harness %s from LAN %s - %s: %s",
                    self.backend.info.id,
                    network_id,
                    type(error).__name__,
                    clip(str(error)),
                    exc_info=logger.isEnabledFor(logging.DEBUG),
                )

        if self._registered_hosts:
            async with self.entry.egress_lock:
                for host in self._registered_hosts:
                    remaining = self.entry.route_counts[host] - 1
                    if remaining > 0:
                        self.entry.route_counts[host] = remaining
                    else:
                        self.entry.route_counts.pop(host, None)
                self._registered_hosts = []

        stop_backend: PrimeBackend | None = None
        async with self.pool._lock:
            if (
                not healthy
                or not detached
                or self.backend.stopped
                or (self.entry.boot_task.done() and self.backend._client is None)
            ):
                self.entry.healthy = False
            self.entry.load = max(0, self.entry.load - 1)
            if (self.pool._closed or not self.entry.healthy) and self.entry.load == 0:
                bucket = self.pool._entries.get(self.entry.key)
                if bucket and self.entry in bucket:
                    bucket.remove(self.entry)
                stop_backend = self.backend
        if stop_backend is not None:
            await stop_backend.stop()


class HarnessPool:
    """Pool and multiplex split Prime harness VMs (`A`) across concurrent rollouts."""

    def __init__(self) -> None:
        self._entries: dict[str, list[_PooledHarness]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._closed = False

    async def acquire(
        self, config: PrimeConfig, name: str, *, multiplex: int = 8
    ) -> HarnessLease:
        from verifiers.v1.runtimes.backends.prime import PrimeBackend

        key = _pool_key(config)
        workdir = config.workdir or "/app"
        to_stop: list[PrimeBackend] = []
        selected: _PooledHarness | None = None

        async with self._lock:
            bucket = self._entries[key]
            for entry in list(bucket):
                if (
                    not entry.healthy
                    or entry.backend.stopped
                    or (entry.boot_task.done() and entry.backend._client is None)
                ):
                    entry.healthy = False
                    if entry.load == 0:
                        bucket.remove(entry)
                        to_stop.append(entry.backend)
            candidates = [
                entry
                for entry in bucket
                if entry.healthy
                and entry.load < multiplex
                and not (
                    entry.load > 0
                    and entry.applied_sig is not None
                    and entry.backend.config.allow != ["*"]
                    and not entry.backend.cache.mcp_sources
                )
            ]
            if candidates:
                # Prefer already-booted VMs with the lightest active load.
                selected = min(
                    candidates,
                    key=lambda e: (0 if e.boot_task.done() else 1, e.load),
                )
                selected.load += 1
                logger.debug(
                    "Multiplexing prime harness %s load=%d/%d",
                    selected.backend.info.id or selected.backend.name,
                    selected.load,
                    multiplex,
                )
            else:
                base_cfg = config.model_copy(update={"workdir": "/app"})
                backend = make_backend(base_cfg, name)
                assert isinstance(backend, PrimeBackend)
                register(backend)
                with isolated():
                    boot_task = asyncio.create_task(backend.start())
                selected = _PooledHarness(
                    key=key, backend=backend, boot_task=boot_task, load=1
                )
                bucket.append(selected)
                total = sum(len(items) for items in self._entries.values())
                logger.info(
                    "Grew harness pool to %d VM(s) multiplex=%d", total, multiplex
                )

        if to_stop:
            await asyncio.gather(*(b.stop() for b in to_stop), return_exceptions=True)
        assert selected is not None
        return HarnessLease(self, selected, workdir)

    async def aclose(self) -> None:
        """Stop all pooled harness VMs once the eval exits."""
        async with self._lock:
            self._closed = True
            entries = [entry for bucket in self._entries.values() for entry in bucket]
            self._entries.clear()
        boot_tasks = [e.boot_task for e in entries if not e.boot_task.done()]
        for task in boot_tasks:
            task.cancel()
        if boot_tasks:
            with contextlib.suppress(Exception):
                await asyncio.gather(*boot_tasks, return_exceptions=True)
        if entries:
            await asyncio.gather(
                *(e.backend.stop() for e in entries), return_exceptions=True
            )


@asynccontextmanager
async def harness_pool_scope(
    pool: HarnessPool | None = None,
) -> AsyncIterator[HarnessPool]:
    """Bind a `HarnessPool` for the current async context, tearing down an owned pool on exit."""
    existing = _current_pool.get()
    if existing is not None:
        yield existing
        return
    owned = pool is None
    active = pool or HarnessPool()
    token = _current_pool.set(active)
    try:
        yield active
    finally:
        _current_pool.reset(token)
        if owned:
            await run_shielded(active.aclose())
