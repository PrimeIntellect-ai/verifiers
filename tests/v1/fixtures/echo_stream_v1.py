"""echo-stream: the echo phrases, arriving over time — a feed that never drains.

A fixture streaming taskset for the v1 e2e suite: `stream()` yields an echo task every
`gap` seconds, cycling the phrases for as long as it is read, as a live feed would. A run
over it ends only where the caller says — `vf eval -n` takes the next `n` and stops — so
the runner's windowed pull and its bounded end are exercised end to end (a pull past the
`n`th would show up as an extra episode). Resolved by id `echo-stream-v1`.
"""

import asyncio
import itertools
from collections.abc import AsyncIterator

from echo_v1 import EchoConfig, EchoTask, EchoTaskset

import verifiers.v1 as vf


class EchoStreamConfig(EchoConfig):
    gap: float = 0.5
    """Seconds before each task appears."""


class EchoStreamTaskset(vf.Taskset[EchoTask, EchoStreamConfig]):
    def load(self) -> list[EchoTask]:
        return []  # nothing to list up front: the tasks arrive through `stream()`

    async def stream(self) -> AsyncIterator[EchoTask]:
        phrases = EchoTaskset(self.config).load()
        for idx, task in enumerate(itertools.cycle(phrases)):
            await asyncio.sleep(self.config.gap)
            yield EchoTask(task.data.model_copy(update={"idx": idx}), self.config.task)


__all__ = ["EchoStreamTaskset"]
