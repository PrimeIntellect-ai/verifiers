import json
import logging
import sys
import copy

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from verifiers.types import Messages
from collections.abc import Mapping

import base64
from io import BytesIO
from PIL import Image


def extract_images(obj):
    """
    Extract and decode Base64 images into a list of PIL.Image objects.
    """
    images = []

    def _extract(o):
        if isinstance(o, dict):
            for v in o.values():
                _extract(v)
            if "image_url" in o and isinstance(o["image_url"], dict):
                url = o["image_url"].get("url")
                if isinstance(url, str) and url.startswith("data:image/"):
                    try:
                        header, b64_data = url.split(",", 1)
                        image_data = base64.b64decode(b64_data)
                        image = Image.open(BytesIO(image_data))
                        images.append(image)
                    except Exception:
                        pass
        elif isinstance(o, list):
            for v in o:
                _extract(v)

    _extract(obj)
    return images

def sanitize_and_serialize(obj):
    """
    Sanitize Base64 images and convert nested dict/list to string for WandB.
    """
    if isinstance(obj, dict):
        obj = {k: sanitize_and_serialize(v) for k, v in obj.items()}
        if "image_url" in obj and isinstance(obj["image_url"], dict):
            url = obj["image_url"].get("url")
            if isinstance(url, str) and url.startswith("data:image/"):
                obj["image_url"]["url"] = "<BASE64_IMAGE_REMOVED>"
        return obj
    elif isinstance(obj, list):
        return [sanitize_and_serialize(x) for x in obj]
    else:
        return obj

def serialize_for_wandb(obj):
    sanitized = sanitize_and_serialize(obj)
    return json.dumps(sanitized, ensure_ascii=False)

        
def sanitize_message_for_logging(msg):
    """
    Recursively sanitize a message dict, removing Base64 data from image URLs.
    """
    msg = copy.deepcopy(msg)

    if isinstance(msg, dict):
        for k, v in msg.items():
            if k == "image_url" and isinstance(v, dict) and "url" in v:
                url = v["url"]
                if url.startswith("data:image/"):
                    v["url"] = "<BASE64_IMAGE_REMOVED>"
            else:
                msg[k] = sanitize_message_for_logging(v)

    elif isinstance(msg, list):
        msg = [sanitize_message_for_logging(x) for x in msg]

    return msg


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

            safe_content = sanitize_message_for_logging(content)
            out.append(str(safe_content), style=style)

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
