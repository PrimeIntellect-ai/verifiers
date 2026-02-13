import asyncio
import logging

import aiohttp

logger = logging.getLogger(__name__)


class Heartbeat:
    """Sends periodic HTTP GET heartbeats to a monitoring URL."""

    def __init__(self, url: str, interval: float = 30.0):
        self.url = url
        self.interval = interval
        self._task: asyncio.Task | None = None

    async def _run(self):
        async with aiohttp.ClientSession() as session:
            while True:
                await asyncio.sleep(self.interval)
                try:
                    async with session.get(self.url, timeout=aiohttp.ClientTimeout(total=5)):
                        pass
                except Exception as e:
                    logger.warning(f"Heartbeat failed: {e}")

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
