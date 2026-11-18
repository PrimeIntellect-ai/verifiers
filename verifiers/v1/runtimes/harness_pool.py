"""Warm pool of split Prime harness VMs shared across rollouts in an eval."""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

from verifiers.v1.configs.runtime import PrimeConfig
from verifiers.v1.runtimes import make_backend
from verifiers.v1.utils.aio import run_shielded
from verifiers.v1.utils.logging import clip, get_logger

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
            "workdir": None,
            "allow": ["*"],
            "block": [],
            "labels": [],
            "creates_per_min": None,
        }
    )
    return json.dumps(canonical.model_dump(mode="json"), sort_keys=True)


class HarnessPool:
    """Pool idle split Prime harness VMs (`A`) across rollouts on one event loop."""

    def __init__(self) -> None:
        self._idle: dict[str, list[PrimeBackend]] = defaultdict(list)
        self._closed = False

    def acquire(self, config: PrimeConfig, name: str) -> tuple[PrimeBackend, bool]:
        """Return `(backend, reused)` for `config`."""
        from verifiers.v1.runtimes.backends.prime import PrimeBackend

        key = _pool_key(config)
        bucket = self._idle[key]
        while bucket:
            backend = bucket.pop()
            if backend.stopped or backend._client is None or backend.info.id is None:
                continue
            backend.config = config.model_copy(
                update={"workdir": config.workdir or "/app"}
            )
            backend.info.workdir = backend.config.workdir
            backend.info.allow = config.allow
            backend.info.block = config.block
            backend.lan = None
            backend._lan_pending = None
            backend._lan_sync_task = None
            logger.debug("Reusing warm prime harness %s", backend.info.id)
            return backend, True
        backend = make_backend(config, name)
        assert isinstance(backend, PrimeBackend)
        return backend, False

    async def prepare_reused(self, backend: PrimeBackend) -> None:
        """Reset workdir, egress, and LAN membership on a leased warm harness VM."""
        from verifiers.v1.runtimes.backends.prime import _resolve_and_join_lan

        if backend.lan is not None or backend._lan_pending is not None:
            backend._lan_sync_task = asyncio.create_task(_resolve_and_join_lan(backend))
        tasks = []
        workdir = backend.config.workdir or "/app"
        if workdir not in backend._created_workdirs:

            async def ensure_workdir() -> None:
                await backend.bind().run(
                    ["sh", "-c", f"mkdir -p {shlex.quote(workdir)}"], {}
                )
                backend._created_workdirs.add(workdir)

            tasks.append(ensure_workdir())
        if backend._egress_enforced:
            tasks.append(backend.prepare_execution(None))
        if tasks:
            await asyncio.gather(*tasks)

    async def release(
        self,
        backend: PrimeBackend,
        network_id: str | None,
        *,
        healthy: bool = True,
    ) -> None:
        """Detach `backend` from `network_id` and return it to the pool, or stop it."""
        from verifiers.v1.runtimes.controller import current_job

        if backend._lan_sync_task is not None and not backend._lan_sync_task.done():
            backend._lan_sync_task.cancel()
            await asyncio.gather(backend._lan_sync_task, return_exceptions=True)
        backend._lan_sync_task = None
        backend.lan = None
        backend._lan_pending = None
        detached = True
        if network_id and backend._client is not None and backend.info.id is not None:
            try:
                if job := current_job.get():
                    await job.launcher.call(
                        "leave_network",
                        network_id=network_id,
                        sandbox_id=backend.info.id,
                    )
                else:
                    await backend._client.remove_lan_member(network_id, backend.info.id)
            except Exception as error:
                detached = False
                logger.warning(
                    "Failed to detach warm prime harness %s from LAN %s - %s: %s",
                    backend.info.id,
                    network_id,
                    type(error).__name__,
                    clip(str(error)),
                    exc_info=logger.isEnabledFor(logging.DEBUG),
                )
        if (
            self._closed
            or not healthy
            or not detached
            or backend.stopped
            or backend._client is None
            or backend.info.id is None
        ):
            await backend.stop()
            return
        self._idle[_pool_key(backend.config)].append(backend)

    async def aclose(self) -> None:
        """Stop all idle pooled harness VMs concurrently."""
        self._closed = True
        backends = [backend for bucket in self._idle.values() for backend in bucket]
        self._idle.clear()
        if backends:
            await asyncio.gather(
                *(backend.stop() for backend in backends), return_exceptions=True
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
