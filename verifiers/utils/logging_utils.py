import json
import logging
import sys
from contextlib import contextmanager
from datetime import datetime, timezone

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from verifiers.errors import Error
from verifiers.types import ErrorInfo, Messages
from verifiers.utils.error_utils import ErrorChain

LOGGER_NAME = "verifiers"

_seen_once_keys: set[tuple[str, str]] = set()


def log_once(logger: logging.Logger, level: int, msg: str) -> None:
    """Log a message only once per (logger name, message) pair for the process lifetime."""
    key = (logger.name, msg)
    if key in _seen_once_keys:
        return
    _seen_once_keys.add(key)
    logger.log(level, msg)


def warning_once(logger: logging.Logger, msg: str) -> None:
    """Shorthand for ``log_once(logger, logging.WARNING, ...)``."""
    log_once(logger, logging.WARNING, msg)


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def setup_logging(
    level: str | None = "INFO",
    log_format: str | None = None,
    date_format: str | None = None,
    log_file: str | None = None,
    console_logging: bool = True,
    file_logging: bool = False,
    json_logging: bool = False,
) -> None:
    """
    Setup basic logging configuration for the verifiers package.

    Args:
        level: The logging level to use. If None, logging is disabled. Defaults to "INFO".
        log_format: Custom log format string. If None, uses default format.
        date_format: Custom date format string. If None, uses default format.
        log_file: Path to a log file. Required when file_logging is True.
        console_logging: Whether to log to stderr. Defaults to True.
        file_logging: Whether to log to a file. Defaults to False.
        json_logging: If True, output logs as JSON. Defaults to False.
    """
    if json_logging:
        formatter = JsonFormatter()
    else:
        if log_format is None:
            log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        if date_format is None:
            date_format = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

    logger = logging.getLogger(LOGGER_NAME)

    # remove any existing handlers to avoid duplicates
    logger.handlers.clear()

    if level is None:
        logger.propagate = False
        return

    log_level = getattr(logging, level.upper())
    logger.setLevel(log_level)

    if console_logging:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        logger.addHandler(console_handler)

    if file_logging:
        if log_file is None:
            raise ValueError("log_file must be specified when file_logging is True")
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    # prevent the logger from propagating messages to the root logger
    logger.propagate = False

    # when json_logging, also configure the root logger so environment code
    # using logging.getLogger(__name__) emits JSON too
    if json_logging:
        root = logging.getLogger()
        root.handlers = [
            h for h in root.handlers if not isinstance(h.formatter, JsonFormatter)
        ]
        root.setLevel(log_level)
        root_handler = logging.StreamHandler(sys.stderr)
        root_handler.setFormatter(formatter)
        root_handler.setLevel(log_level)
        root.addHandler(root_handler)


@contextmanager
def log_level(level: str | int):
    """
    Context manager to temporarily set the verifiers logger to a new log level.
    Useful for temporarily silencing verifiers logging.

    with log_level("DEBUG"):
        # verifiers logs at DEBUG level here
        ...
    # reverts to previous level
    """
    logger = logging.getLogger(LOGGER_NAME)
    prev_level = logger.level
    new_level = level if isinstance(level, int) else getattr(logging, level.upper())
    logger.setLevel(new_level)
    try:
        yield
    finally:
        logger.setLevel(prev_level)


def quiet_verifiers():
    """Context manager to temporarily silence verifiers logging by setting WARNING level."""
    return log_level("WARNING")


def print_prompt_completions_sample(
    prompts: list[Messages],
    completions: list[Messages],
    errors: list[Error | ErrorInfo | None],
    rewards: list[float],
    step: int,
    num_samples: int = 1,
) -> None:
    from verifiers.utils.message_utils import format_messages

    def format_error(error: ErrorInfo | BaseException) -> Text:
        out = Text()
        if isinstance(error, BaseException):
            out.append(f"error: {ErrorChain(error)}", style="bold red")
        else:
            out.append(f"error: {error['error_chain_repr']}", style="bold red")
        return out

    console = Console()
    table = Table(show_header=True, header_style="bold white", expand=True)

    table.add_column("Prompt", style="bright_yellow")
    table.add_column("Completion", style="bright_green")
    table.add_column("Reward", style="bold cyan", justify="right")

    reward_values = rewards
    if len(reward_values) < len(prompts):
        reward_values = reward_values + [0.0] * (len(prompts) - len(reward_values))

    samples_to_show = min(num_samples, len(prompts))
    for i in range(samples_to_show):
        prompt = list(prompts)[i]
        completion = list(completions)[i]
        error = errors[i]
        reward = reward_values[i]

        formatted_prompt = format_messages(prompt)
        formatted_completion = format_messages(completion)
        if error is not None:
            formatted_completion += Text("\n\n") + format_error(error)

        table.add_row(formatted_prompt, formatted_completion, Text(f"{reward:.2f}"))
        if i < samples_to_show - 1:
            table.add_section()

    panel = Panel(table, expand=False, title=f"Step {step}", border_style="bold white")
    console.print(panel)


def print_time(time_s: float) -> str:
    """
    Format a time in seconds to a human-readable format:
    - >1d -> Xd Yh
    - >1h -> Xh Ym
    - >1m -> Xm Ys
    - <1s -> Xms
    - Else: Xs
    """
    if time_s >= 86400:  # >1d
        d = time_s // 86400
        h = (time_s % 86400) // 3600
        return f"{d:.0f}d" + (f" {h:.0f}h" if h > 0 else "")
    elif time_s >= 3600:  # >1h
        h = time_s // 3600
        m = (time_s % 3600) // 60
        return f"{h:.0f}h" + (f" {m:.0f}m" if m > 0 else "")
    elif time_s >= 60:  # >1m
        m = time_s // 60
        s = (time_s % 60) // 1
        return f"{m:.0f}m" + (f" {s:.0f}s" if s > 0 else "")
    elif time_s < 1:  # <1s
        ms = time_s * 1e3
        return f"{ms:.0f}ms"
    else:
        return f"{time_s:.0f}s"


def truncate(s: str, limit: int = 200) -> str:
    """Truncate a string to a given length."""
    return (s[:limit] + "...") if len(s) > limit else s
