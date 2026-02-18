"""
Shared utilities for Rich-based terminal displays.

Provides common infrastructure for EvalDisplay and GEPADisplay:
- Terminal control detection and handling
- Screen mode support (normal vs alternate screen buffer)
- Echo disable/restore for TUI mode
- Wait-for-exit key handling with escape sequence draining
- Log capture and display
"""

import asyncio
import io
import logging
import os
import sys
import threading
from collections import deque
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def make_aligned_row(left: Text, right: Text) -> Table:
    """Create a row with left-aligned and right-aligned content."""
    table = Table.grid(expand=True)
    table.add_column(justify="left", ratio=1)
    table.add_column(justify="right")
    table.add_row(left, right)
    return table


def format_numeric(value: float | int | str) -> str:
    if isinstance(value, float):
        if value == int(value):
            return str(int(value))
        if abs(value) < 0.01:
            return f"{value:.4f}"
        return f"{value:.3f}"
    return str(value)


# Suppress tokenizers parallelism warning (only prints when env var is unset)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

# Check for unix-specific terminal control modules
try:
    import select  # noqa: F401
    import termios  # noqa: F401
    import tty  # noqa: F401

    HAS_TERMINAL_CONTROL = True
except ImportError:
    HAS_TERMINAL_CONTROL = False


def is_tty() -> bool:
    """Check if stdout is a TTY (terminal)."""
    return sys.stdout.isatty()


class DisplayLogHandler(logging.Handler):
    """Custom log handler that captures log records for display."""

    def __init__(self, max_lines: int = 3) -> None:
        super().__init__()
        self.logs: deque[str] = deque(maxlen=max_lines)
        self.setFormatter(logging.Formatter("%(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if record.name.endswith(".stdout") or record.name.endswith(".stderr"):
                msg = record.getMessage()
            else:
                msg = self.format(record)
            self.logs.append(msg)
        except Exception:
            pass


class _FDToLogger(threading.Thread):
    """Background reader that forwards a file descriptor's output to a logger."""

    def __init__(
        self, fd: int, logger: logging.Logger, level: int, encoding: str | None
    ) -> None:
        super().__init__(daemon=True)
        self._fd = fd
        self._logger = logger
        self._level = level
        self._encoding = encoding or "utf-8"
        self._buffer = ""

    def run(self) -> None:
        try:
            while True:
                try:
                    data = os.read(self._fd, 1024)
                except OSError:
                    break
                if not data:
                    break
                text = data.decode(self._encoding, errors="replace").replace("\r", "\n")
                combined = f"{self._buffer}{text}"
                lines = combined.split("\n")
                self._buffer = lines.pop() if lines else ""
                for line in lines:
                    if line:
                        self._logger.log(self._level, line)
        finally:
            if self._buffer:
                self._logger.log(self._level, self._buffer)
                self._buffer = ""
            try:
                os.close(self._fd)
            except OSError:
                pass


class BaseDisplay:
    """
    Base class for Rich-based terminal displays.

    Provides shared infrastructure for screen mode toggling, terminal echo handling,
    and wait-for-exit functionality. Subclasses should implement `_render()` to
    return the Rich renderable for the display.

    Args:
        screen: If True, use alternate screen buffer with echo handling (TUI mode).
                If False, refresh in-place without screen hijacking (default mode).
        refresh_per_second: How often to refresh the display.
    """

    def __init__(self, screen: bool = False, refresh_per_second: int = 4) -> None:
        self.screen = screen
        self.refresh_per_second = refresh_per_second
        self.console = Console()
        self._live: Live | None = None
        self._old_terminal_settings: list | None = None
        self._log_handler = DisplayLogHandler(max_lines=3)
        self._old_handler_levels: dict[logging.Handler, int] = {}
        self._old_datasets_level: int | None = None
        self._old_stdout = None
        self._old_stderr = None
        self._old_stdout_fd: int | None = None
        self._old_stderr_fd: int | None = None
        self._console_file: io.TextIOWrapper | None = None
        self._stdout_thread: _FDToLogger | None = None
        self._stderr_thread: _FDToLogger | None = None

    def _render(self) -> Any:
        """
        Render the display content. Subclasses must implement this.

        Returns:
            A Rich renderable (Layout, Group, Panel, etc.)
        """
        raise NotImplementedError("Subclasses must implement _render()")

    def refresh(self) -> None:
        """Refresh the display with current content."""
        if self._live:
            self._live.update(self._render())

    def get_log_hint(self) -> Text | None:
        """Return an optional hint for viewing full logs."""
        return Text("full logs: --debug", style="dim")

    def _make_log_panel(self) -> Panel:
        """Create a panel showing recent log messages with placeholder lines."""
        max_lines = self._log_handler.logs.maxlen or 3
        log_text = Text(no_wrap=True, overflow="ellipsis")

        # Fill with actual logs or placeholder lines
        logs_list = list(self._log_handler.logs)
        for i in range(max_lines):
            if i > 0:
                log_text.append("\n")
            if i < len(logs_list):
                log_text.append(logs_list[i], style="dim")
            else:
                log_text.append(" ", style="dim")  # placeholder line

        subtitle = self.get_log_hint()
        if subtitle is None:
            return Panel(log_text, title="[dim]Logs[/dim]", border_style="dim")
        return Panel(
            log_text,
            title="[dim]Logs[/dim]",
            subtitle=subtitle,
            subtitle_align="center",
            border_style="dim",
        )

    def start(self) -> None:
        """Start the live display."""
        # Suppress datasets progress bars (e.g. from .map())
        from datasets import disable_progress_bar

        disable_progress_bar()

        # Suppress console output from existing handlers but capture logs for display
        logger = logging.getLogger("verifiers")

        # Preserve original streams for Rich rendering before capturing stdout/stderr
        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        self._old_stdout_fd = os.dup(1)
        self._old_stderr_fd = os.dup(2)
        self._console_file = io.TextIOWrapper(
            os.fdopen(self._old_stdout_fd, "wb", closefd=False),
            encoding=getattr(self._old_stdout, "encoding", "utf-8"),
            write_through=True,
        )
        self.console = Console(file=self._console_file, force_terminal=True)
        for handler in logger.handlers:
            self._old_handler_levels[handler] = handler.level
            handler.setLevel(logging.CRITICAL)

        # Also suppress datasets logger (prints tokenizers warning)
        datasets_logger = logging.getLogger("datasets")
        self._old_datasets_level = datasets_logger.level
        datasets_logger.setLevel(logging.ERROR)

        # Add our handler to capture logs for display panel
        self._log_handler.setLevel(logging.INFO)
        logger.addHandler(self._log_handler)

        # Capture stdout/stderr at the FD level so stray prints don't corrupt the live display
        stdout_r, stdout_w = os.pipe()
        stderr_r, stderr_w = os.pipe()
        os.dup2(stdout_w, 1)
        os.close(stdout_w)
        os.dup2(stderr_w, 2)
        os.close(stderr_w)
        self._stdout_thread = _FDToLogger(
            stdout_r,
            logger.getChild("stdout"),
            logging.INFO,
            getattr(self._old_stdout, "encoding", None),
        )
        self._stderr_thread = _FDToLogger(
            stderr_r,
            logger.getChild("stderr"),
            logging.ERROR,
            getattr(self._old_stderr, "encoding", None),
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

        # Disable terminal echo in screen mode to prevent scroll/arrow keys from displaying
        if self.screen and HAS_TERMINAL_CONTROL and sys.stdin.isatty():
            import termios

            fd = sys.stdin.fileno()
            self._old_terminal_settings = termios.tcgetattr(fd)
            new_settings = termios.tcgetattr(fd)
            # Disable echo (ECHO flag in lflags)
            new_settings[3] = new_settings[3] & ~termios.ECHO
            termios.tcsetattr(fd, termios.TCSADRAIN, new_settings)

        # In non-TUI mode, clamp vertical overflow so oversized renders don't
        # scroll and smear repeated frames in-place.
        vertical_overflow = "visible" if self.screen else "ellipsis"

        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=self.refresh_per_second,
            screen=self.screen,
            transient=False,  # Keep final display visible in scrollback
            vertical_overflow=vertical_overflow,
        )
        self._live.start()

    def stop(self) -> None:
        """Stop the live display and restore terminal settings."""
        if self._live:
            self._live.stop()
            self._live = None

        # Restore stdout/stderr file descriptors (ends pipe, unblocks readers)
        if self._old_stdout_fd is not None:
            os.dup2(self._old_stdout_fd, 1)
        if self._old_stderr_fd is not None:
            os.dup2(self._old_stderr_fd, 2)

        # Join reader threads
        if self._stdout_thread is not None:
            self._stdout_thread.join(timeout=0.5)
            self._stdout_thread = None
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=0.5)
            self._stderr_thread = None

        # Restore datasets progress bar
        from datasets import enable_progress_bar

        enable_progress_bar()

        # Remove our log handler and restore original handler levels
        logger = logging.getLogger("verifiers")
        logger.removeHandler(self._log_handler)
        for handler, level in self._old_handler_levels.items():
            handler.setLevel(level)
        self._old_handler_levels.clear()

        # Restore datasets logger level
        if self._old_datasets_level is not None:
            datasets_logger = logging.getLogger("datasets")
            datasets_logger.setLevel(self._old_datasets_level)
            self._old_datasets_level = None

        # Restore stdout/stderr
        if self._old_stdout is not None:
            sys.stdout = self._old_stdout
            self._old_stdout = None
        if self._old_stderr is not None:
            sys.stderr = self._old_stderr
            self._old_stderr = None
        if self._console_file is not None:
            # Redirect console back to original stdout before closing temp stream
            self.console = Console(file=sys.stdout, force_terminal=sys.stdout.isatty())
            try:
                self._console_file.flush()
                self._console_file.close()
            finally:
                self._console_file = None
        if self._old_stdout_fd is not None:
            os.close(self._old_stdout_fd)
            self._old_stdout_fd = None
        if self._old_stderr_fd is not None:
            os.close(self._old_stderr_fd)
            self._old_stderr_fd = None

        # Restore terminal settings
        if self._old_terminal_settings is not None:
            import termios

            fd = sys.stdin.fileno()
            termios.tcsetattr(fd, termios.TCSADRAIN, self._old_terminal_settings)
            self._old_terminal_settings = None

    async def wait_for_exit(self) -> None:
        """
        Wait for user to press a key to exit.

        Only used in screen mode (--tui). Handles:
        - q/Q to exit
        - Enter to exit
        - Escape to exit
        - Drains escape sequences from mouse/scroll events
        """
        if not HAS_TERMINAL_CONTROL or not sys.stdin.isatty():
            # On Windows or non-tty, just wait for a simple input
            await asyncio.get_event_loop().run_in_executor(None, input)
            return

        # These imports are guaranteed to exist when HAS_TERMINAL_CONTROL is True
        import select as select_module
        import termios as termios_module
        import tty as tty_module

        # Save terminal settings
        fd = sys.stdin.fileno()
        old_settings = termios_module.tcgetattr(fd)

        def drain_escape_sequence() -> None:
            """Consume remaining chars of an escape sequence (mouse events, etc)."""
            while select_module.select([sys.stdin], [], [], 0.01)[0]:
                sys.stdin.read(1)

        try:
            # Use cbreak mode (not raw) - allows single char input without corrupting display
            tty_module.setcbreak(fd)

            # Wait for key press in a non-blocking way
            while True:
                # Small delay to keep display responsive
                await asyncio.sleep(0.1)

                # Use select to check for input without blocking
                if select_module.select([sys.stdin], [], [], 0)[0]:
                    char = sys.stdin.read(1)

                    # Handle escape sequences (mouse scroll, arrow keys, etc)
                    if char == "\x1b":
                        # Check if more chars follow (escape sequence vs standalone Esc)
                        if select_module.select([sys.stdin], [], [], 0.05)[0]:
                            # Escape sequence - drain it and ignore
                            drain_escape_sequence()
                            continue
                        else:
                            # Standalone Escape key - exit
                            break

                    # Exit on q, Q, or enter
                    if char in ("q", "Q", "\r", "\n"):
                        break
        finally:
            # Restore terminal settings
            termios_module.tcsetattr(fd, termios_module.TCSADRAIN, old_settings)

    async def __aenter__(self) -> "BaseDisplay":
        """Async context manager entry - start the display."""
        self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - stop the display."""
        self.stop()

    def __enter__(self) -> "BaseDisplay":
        """Sync context manager entry - start the display."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Sync context manager exit - stop the display."""
        self.stop()
