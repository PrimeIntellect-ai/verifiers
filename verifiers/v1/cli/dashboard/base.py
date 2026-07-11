"""Shared live-dashboard rendering."""

import asyncio
import contextlib
import os
import sys
from collections.abc import Callable, Iterator

from rich.console import Group
from rich.live import Live

from verifiers.v1.utils.interrupt import cleaning_up

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
    """Read arrow keys and restore terminal state on exit."""
    if on_key is None or not _HAS_TTY or not sys.stdin.isatty():
        yield
        return
    fd = sys.stdin.fileno()
    loop = asyncio.get_running_loop()
    old = termios.tcgetattr(fd)
    # carries a partial escape across reads — cbreak can split a key's 3 bytes over callbacks
    buf = bytearray()

    def _read() -> None:
        try:
            data = os.read(fd, 32)
        except OSError:
            return
        if not data:
            return
        buf.extend(data)
        fired = False
        i = 0
        while i < len(buf):  # a chunk may hold several presses; scan the whole buffer
            if key := _ARROWS.get(bytes(buf[i : i + 3])):
                on_key(key)
                fired = True
                i += 3
            elif buf[i] == 0x1B and len(buf) - i < 3:
                break  # incomplete escape at the tail — hold it for the next read
            else:
                i += 1
        del buf[:i]
        if fired:
            refresh()

    # setcbreak inside the try so the finally always restores the terminal, even if it (or
    # add_reader) raises — otherwise stdin could be left in cbreak with echo off after we exit.
    try:
        tty.setcbreak(fd)
        loop.add_reader(fd, _read)
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
        stopping = False

        async def loop() -> None:
            while True:
                try:
                    await asyncio.sleep(0.25)
                except asyncio.CancelledError:
                    # On Ctrl-C, asyncio cancels every task at once. Keep refreshing through
                    # graceful shutdown so the "cleaning up" banner stays live during teardown;
                    # only really stop when live_view itself tears down (`stopping`), else the
                    # dashboard would freeze for the whole (possibly slow) container teardown.
                    if stopping or not cleaning_up():
                        raise
                live.update(render(), refresh=True)

        task = asyncio.create_task(loop())
        with _key_reader(on_key, lambda: live.update(render(), refresh=True)):
            try:
                yield
            finally:
                stopping = True
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                live.update(render(), refresh=True)
