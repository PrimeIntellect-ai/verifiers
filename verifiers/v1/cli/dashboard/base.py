"""The shared dashboard engine: a rich `Live` view on a fixed refresh tick.

The eval and validate dashboards each build their own frame; `live_view` just drives whichever
`render` it's given — refreshing on a timer and drawing a final frame on exit.
"""

import asyncio
import contextlib
from collections.abc import Callable

from rich.console import Group
from rich.live import Live


@contextlib.asynccontextmanager
async def live_view(render: Callable[[], Group]):
    """Refresh `render()` every 0.25s until the block exits, then draw one final frame."""
    with Live(render(), auto_refresh=False) as live:

        async def loop() -> None:
            while True:
                await asyncio.sleep(0.25)
                live.update(render(), refresh=True)

        task = asyncio.create_task(loop())
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            live.update(render(), refresh=True)
