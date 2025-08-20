import logging
import sys

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

    # Create a StreamHandler that writes to stderr
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt=log_format, datefmt=date_format))

    # Get the root logger for the verifiers package
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
    num_samples: int = 1,  # Number of samples to display
) -> None:
    console = Console()
    table = Table(show_header=True, header_style="bold white", expand=True)

    # Add columns
    table.add_column("Prompt", style="bright_yellow")
    table.add_column("Completion", style="bright_green")
    table.add_column("Reward", style="bold cyan", justify="right")

    # Get the reward values from the dictionary
    reward_values = rewards

    # Ensure we have rewards for all prompts/completions
    if len(reward_values) < len(prompts):
        # Pad with zeros if we don't have enough rewards
        reward_values = reward_values + [0.0] * (len(prompts) - len(reward_values))

    # Only show the first num_samples samples
    samples_to_show = min(num_samples, len(prompts))

    for i in range(samples_to_show):
        prompt = prompts[i]
        completion = completions[i]
        reward = reward_values[i]

        # Format prompt (can be string or list of dicts)
        formatted_prompt = Text()
        if isinstance(prompt, str):
            formatted_prompt = Text(prompt)
        elif isinstance(prompt, list):
            # For chat format, only show the last message content (typically the user's question)
            if prompt:
                last_message = prompt[-1]
                content = str(last_message.get("content", ""))
                formatted_prompt = Text(content, style="bright_yellow")
            else:
                formatted_prompt = Text("")
        else:
            formatted_prompt = Text(str(prompt))

        # Create a formatted Text object for completion with alternating colors based on role
        formatted_completion = Text()

        if isinstance(completion, dict):
            # Handle single message dict
            role = completion.get("role", "")
            content = completion.get("content", "")
            style = "bright_cyan" if role == "assistant" else "bright_magenta"
            formatted_completion.append(f"{role}: ", style="bold")
            formatted_completion.append(content, style=style)
        elif isinstance(completion, list):
            # Handle list of message dicts
            for i, message in enumerate(completion):
                if i > 0:
                    formatted_completion.append("\n\n")

                role = message.get("role", "")
                content = str(message.get("content", ""))
                tool_calls = message.get("tool_calls", [])

                # Set style based on role
                style = "bright_cyan" if role == "assistant" else "bright_magenta"

                formatted_completion.append(f"{role}: ", style="bold")
                formatted_completion.append(content, style=style)
                for tool_call in tool_calls:
                    formatted_completion.append("\n\n[tool call]", style=style)
                    formatted_completion.append(
                        f"\nname: {tool_call.function.name}",  # type: ignore
                        style=style,
                    )
                    formatted_completion.append(
                        f"\nargs: {tool_call.function.arguments}",  # type: ignore
                        style=style,
                    )
        else:
            # Fallback for string completions
            formatted_completion = Text(str(completion))

        table.add_row(formatted_prompt, formatted_completion, Text(f"{reward:.2f}"))
        if i < samples_to_show - 1:  # Don't add section after last row
            table.add_section()  # Adds a separator between rows

    panel = Panel(table, expand=False, title=f"Step {step}", border_style="bold white")
    console.print(panel)
