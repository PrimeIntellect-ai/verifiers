"""The shared dashboard engine: a rich `Live` view on a fixed refresh tick.

The eval and validate dashboards each build their own frame; `live_view` just drives whichever
`render` it's given — refreshing on a timer and drawing a final frame on exit. When given an
`on_key` handler it also reads left/right arrow presses from the terminal (see `_key_reader`).
"""

import asyncio
import contextlib
import os
import sys
from collections.abc import Callable, Iterator

from rich.console import Group
from rich.live import Live

try:  # POSIX-only terminal control; absent on Windows, where key reading is skipped.
    import termios
    import tty

    _HAS_TTY = True
except ImportError:
    _HAS_TTY = False

# Arrow-key escape sequences → direction, covering both the normal (`ESC [`) and application
# (`ESC O`) cursor-key modes a terminal might send.
_ARROWS = {
    b"\x1b[C": "right",
    b"\x1bOC": "right",
    b"\x1b[D": "left",
    b"\x1bOD": "left",
}


@contextlib.contextmanager
def _key_reader(
    on_key: Callable[[str], None] | None, refresh: Callable[[], None]
) -> Iterator[None]:
    """Dispatch left/right arrow presses to `on_key`, redrawing immediately after each so the
    view feels instant rather than waiting for the next refresh tick. A no-op unless `on_key` is
    given and stdin is a real terminal. Puts stdin in cbreak mode (char-at-a-time, no echo, Ctrl+C
    still interrupts) and reads it via the event loop — no extra thread — restoring the terminal on
    exit."""
    if on_key is None or not _HAS_TTY or not sys.stdin.isatty():
        yield
        return
    fd = sys.stdin.fileno()
    loop = asyncio.get_running_loop()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    def _read() -> None:
        try:
            data = os.read(fd, 32)
        except OSError:
            return
        fired = False
        i = 0
        while i < len(data):  # a burst may hold several presses; scan the whole chunk
            if key := _ARROWS.get(data[i : i + 3]):
                on_key(key)
                fired = True
                i += 3
            else:
                i += 1
        if fired:
            refresh()

    loop.add_reader(fd, _read)
    try:
        yield
    finally:
        loop.remove_reader(fd)
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


@contextlib.asynccontextmanager
async def live_view(
    render: Callable[[], Group], on_key: Callable[[str], None] | None = None
):
    """Refresh `render()` every 0.25s until the block exits, then draw one final frame. When
    `on_key` is given, left/right arrow presses are dispatched to it (and redraw at once)."""
    with Live(render(), auto_refresh=False) as live:

        async def loop() -> None:
            while True:
                await asyncio.sleep(0.25)
                live.update(render(), refresh=True)

        task = asyncio.create_task(loop())
        with _key_reader(on_key, lambda: live.update(render(), refresh=True)):
            try:
                yield
            finally:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                live.update(render(), refresh=True)
