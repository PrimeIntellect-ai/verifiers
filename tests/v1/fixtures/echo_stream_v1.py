"""echo-stream: the echo phrases, arriving over time.

A fixture streaming taskset for the v1 e2e suite: `stream()` yields each echo task after
a short wait, as a feed would, so the runner's windowed pull (and its end when the stream
drains) is exercised end to end. Resolved by id `echo-stream-v1`.
"""

import asyncio
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
        for task in EchoTaskset(self.config).load():
            await asyncio.sleep(self.config.gap)
            yield task


__all__ = ["EchoStreamTaskset"]
