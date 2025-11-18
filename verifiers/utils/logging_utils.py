import json
import logging
import sys
from collections.abc import Mapping

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from verifiers.types import Messages


def setup_logging(
    level: str = "INFO",
    log_format: str | None = None,
    date_format: str | None = None,
) -> None:
    """
    Setup basic logging configuration for the verifiers package.

    Args:
        level: The logging level to use. Defaults to "INFO".
        log_format: Custom log format string. If None, uses default format.
        date_format: Custom date format string. If None, uses default format.
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if date_format is None:
        date_format = "%Y-%m-%d %H:%M:%S"

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt=log_format, datefmt=date_format))

    logger = logging.getLogger("verifiers")
    logger.setLevel(level.upper())
    logger.addHandler(handler)

    # Prevent the logger from propagating messages to the root logger
    logger.propagate = False


def print_prompt_completions_sample(
    prompts: list[Messages],
    completions: list[Messages],
    rewards: list[float],
    step: int,
    num_samples: int = 1,
) -> None:
    def _attr_or_key(obj, key: str, default=None):
        """Return obj.key if present, else obj[key] if Mapping, else default."""
        val = getattr(obj, key, None)
        if val is not None:
            return val
        if isinstance(obj, Mapping):
            return obj.get(key, default)
        return default

    def _normalize_tool_call(tc):
        """Return {"name": ..., "args": ...} from a dict or Pydantic-like object."""
        src = (
            _attr_or_key(tc, "function") or tc
        )  # prefer nested function object if present
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

            for tc in msg.get("tool_calls") or []:  # treat None as empty list
                payload = _normalize_tool_call(tc)
                out.append(
                    "\n\n[tool call]\n"
                    + json.dumps(payload, indent=2, ensure_ascii=False),
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
