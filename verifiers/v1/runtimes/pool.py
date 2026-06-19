"""An eval-level pool of persistent runtimes.

When a runtime is `persistent`, the rollout takes one from this pool instead of provisioning
its own: an idle runtime of the same config is reused, its per-rollout workspace `reset`, and
returned to the pool on release — torn down only when the pool closes (the end of the eval /
training run). So expensive provisioning (container / sandbox create) and warm in-runtime
workers are paid once across many rollouts, not once per rollout.

Runtimes are pooled per resolved config (`RuntimeConfig.model_dump_json()`), so a taskset with
per-task images keeps a separate warm pool per image; the common case (one config for every
rollout, e.g. gsm8k subprocess) is a single pool.
"""

import asyncio
import logging

from verifiers.v1.runtimes.base import Runtime
from verifiers.v1.runtimes.factory import RuntimeConfig, make_runtime

logger = logging.getLogger(__name__)


class RuntimePool:
    def __init__(self) -> None:
        self._idle: dict[str, list[Runtime]] = {}
        self._all: list[Runtime] = []
        self._lock = asyncio.Lock()
        self._created = 0

    @staticmethod
    def _key(config: RuntimeConfig) -> str:
        return config.model_dump_json()

    async def acquire(self, config: RuntimeConfig) -> Runtime:
        """Return a started runtime for `config` — an idle one if available, else a freshly
        provisioned one — with its per-rollout workspace reset, ready for a new rollout."""
        key = self._key(config)
        async with self._lock:
            idle = self._idle.get(key)
            runtime = idle.pop() if idle else None
        if runtime is None:
            async with self._lock:
                self._created += 1
                name = f"vf-pool-{self._created}"
            runtime = make_runtime(config, name=name)
            async with self._lock:
                self._all.append(
                    runtime
                )  # tracked before start, so aclose frees a failed start
            await runtime.start()
        await runtime.reset()
        return runtime

    async def release(self, runtime: Runtime) -> None:
        """Return a runtime to the idle set for reuse (it stays alive — no teardown)."""
        key = self._key(runtime.config)
        async with self._lock:
            self._idle.setdefault(key, []).append(runtime)

    async def aclose(self) -> None:
        """Tear down every pooled runtime — the end of the eval / training run."""
        async with self._lock:
            runtimes = list(self._all)
            self._all.clear()
            self._idle.clear()
        if runtimes:
            logger.info(
                "runtime pool: tearing down %d persistent runtime(s)", len(runtimes)
            )
        await asyncio.gather(*(rt.stop() for rt in runtimes), return_exceptions=True)

    async def __aenter__(self) -> "RuntimePool":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()
