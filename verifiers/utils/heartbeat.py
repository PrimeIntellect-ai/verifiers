import asyncio
import logging

import aiohttp

logger = logging.getLogger(__name__)


class Heartbeat:
    """Sends a heartbeat GET request to a monitoring URL on each beat() call."""

    def __init__(self, url: str):
        self.url = url
        self._in_flight = False

    async def beat(self):
        if self._in_flight:
            return
        self._in_flight = True
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.url, timeout=aiohttp.ClientTimeout(total=5)
                ):
                    pass
        except Exception as e:
            logger.warning(f"Heartbeat failed: {e}")
        finally:
            self._in_flight = False
