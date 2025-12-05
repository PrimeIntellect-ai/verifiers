"""
Structlog-based logging utilities for verifiers.

This module provides:
- setup_logging(): Configure structlog with selective verbosity
- get_logger(): Get a bound structlog logger
- log_context(): Context manager for adding fields to all logs in scope

Usage:
    from verifiers.utils.logging_utils import setup_logging, get_logger, log_context

    setup_logging()  # Call once at startup

    logger = get_logger(__name__)
    logger.info("Processing", _print=True)  # Always shown
    logger.info("Debug detail")  # Hidden unless VF_LOG_ALL set

    with log_context(env_id="gsm8k", model="gpt-4"):
        logger.info("In context")  # Includes env_id and model
"""

import functools
import json
import logging
import os
import sys
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Generator

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from structlog.contextvars import bound_contextvars
from structlog.typing import EventDict

from verifiers.types import Messages

# Environment variable to control verbose logging
ENV_VF_LOG_ALL = "VF_LOG_ALL"


def _rename_field(
    old: str, new: str, logger: logging.Logger, name: str, event_dict: EventDict
) -> EventDict:
    """Rename a field in the event dict (used in foreign_pre_chain)."""
    del logger, name
    if value := event_dict.get(old):
        event_dict[new] = value
        del event_dict[old]
    return event_dict


def _remove_fields_except(
    to_keep: list[str], logger: logging.Logger, name: str, event_dict: EventDict
) -> EventDict:
    """Keep only specified fields in the event dict, plus remove internal _ fields."""
    del logger, name
    for key in list(event_dict.keys()):
        if key not in to_keep or key.startswith("_"):
            del event_dict[key]
    return event_dict


class PrintOrWarningFilter(logging.Filter):
    """
    Filter that shows:
    - All WARNING and above
    - INFO messages marked with _print=True
    - All INFO if VF_LOG_ALL is set

    This allows selective verbosity: most INFO logs are hidden unless
    explicitly marked as important with _print=True.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # Always show warnings and errors
        if record.levelno > logging.INFO:
            return True

        # Show all INFO if VF_LOG_ALL is set
        if os.environ.get(ENV_VF_LOG_ALL) and record.levelno == logging.INFO:
            return True

        # Show INFO only if _print=True
        return isinstance(record.msg, dict) and record.msg.get("_print", False)


def setup_logging(
    level: str = "INFO",
    log_format: str | None = None,
    date_format: str | None = None,
    use_structlog: bool = True,
) -> None:
    """
    Setup logging configuration for the verifiers package.

    Args:
        level: The logging level to use. Defaults to "INFO".
        log_format: Custom log format string (ignored if use_structlog=True).
        date_format: Custom date format string (ignored if use_structlog=True).
        use_structlog: If True, use structlog with rich console output.
    """
    if not use_structlog:
        # Fallback to basic logging
        _setup_basic_logging(level, log_format, date_format)
        return

    # Capture Python warnings as logs
    logging.captureWarnings(True)

    # Configure structlog for stdlib integration
    structlog.configure(
        processors=[
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
    )

    # Remove existing StreamHandlers from root logger
    for handler in logging.getLogger().handlers[:]:
        if isinstance(handler, logging.StreamHandler):
            logging.getLogger().removeHandler(handler)

    # Create handler with structlog formatter
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.MaybeTimeStamper(fmt="iso"),
                functools.partial(
                    _remove_fields_except,
                    ["timestamp", "level", "event", "component", "exc_info"],
                ),
                structlog.dev.ConsoleRenderer(),
            ],
            foreign_pre_chain=[
                structlog.stdlib.add_logger_name,
                functools.partial(_rename_field, "logger", "component"),
                functools.partial(_rename_field, "logger_name", "component"),
                functools.partial(_rename_field, "log", "event"),
                structlog.stdlib.ExtraAdder(),
            ],
        )
    )

    handler.addFilter(PrintOrWarningFilter())

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))


def _setup_basic_logging(
    level: str = "INFO",
    log_format: str | None = None,
    date_format: str | None = None,
) -> None:
    """Fallback basic logging setup."""
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if date_format is None:
        date_format = "%Y-%m-%d %H:%M:%S"

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt=log_format, datefmt=date_format))

    logger = logging.getLogger("verifiers")
    logger.setLevel(level.upper())
    logger.addHandler(handler)
    logger.propagate = False


def get_logger(component: str | None = None, _print: bool = False) -> structlog.stdlib.BoundLogger:
    """
    Get a structlog logger with optional component binding.

    Args:
        component: Module/class name for identification (typically __name__)
        _print: If True, marks all messages from this logger as important

    Returns:
        A bound structlog logger
    """
    kwargs = {}
    if component:
        kwargs["component"] = component
    if _print:
        kwargs["_print"] = True
    return structlog.stdlib.get_logger(**kwargs)


@contextmanager
def log_context(**kwargs) -> Generator[None, None, None]:
    """
    Context manager to bind context variables to all logs within scope.

    Example:
        with log_context(env_id="gsm8k", model="gpt-4"):
            logger.info("Starting rollout")  # Includes env_id and model
    """
    with bound_contextvars(**kwargs):
        yield


__all__ = [
    "setup_logging",
    "get_logger",
    "log_context",
    "bound_contextvars",
    "print_prompt_completions_sample",
]


# ============================================================================
# Rich-based output utilities (keep existing implementation)
# ============================================================================

def print_prompt_completions_sample(
    prompts: list[Messages],
    completions: list[Messages],
    rewards: list[float],
    step: int,
    num_samples: int = 1,
) -> None:
    def _attr_or_key(obj, key: str, default=None):
        val = getattr(obj, key, None)
        if val is not None:
            return val
        if isinstance(obj, Mapping):
            return obj.get(key, default)
        return default

    def _normalize_tool_call(tc):
        src = _attr_or_key(tc, "function") or tc
        name = _attr_or_key(src, "name", "") or ""
        args = _attr_or_key(src, "arguments", {}) or {}
        if not isinstance(args, str):
            try:
                args = json.dumps(args)
            except Exception:
                args = str(args)
        return {"name": name, "args": args}

    def _format_messages(messages) -> Text:
        if isinstance(messages, str):
            return Text(messages)
        out = Text()
        for idx, msg in enumerate(messages):
            if idx:
                out.append("\n\n")
            assert isinstance(msg, dict)
            role = msg.get("role", "")
            content = msg.get("content", "")
            style = "bright_cyan" if role == "assistant" else "bright_magenta"
            out.append(f"{role}: ", style="bold")
            out.append(content, style=style)
            for tc in msg.get("tool_calls") or []:
                payload = _normalize_tool_call(tc)
                out.append(
                    "\n\n[tool call]\n" + json.dumps(payload, indent=2, ensure_ascii=False),
                    style=style,
                )
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
        reward = reward_values[i]
        formatted_prompt = _format_messages(prompt)
        formatted_completion = _format_messages(completion)
        table.add_row(formatted_prompt, formatted_completion, Text(f"{reward:.2f}"))
        if i < samples_to_show - 1:
            table.add_section()

    panel = Panel(table, expand=False, title=f"Step {step}", border_style="bold white")
    console.print(panel)
