"""Native harness tool hooks normalized onto the rollout interception API."""

import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from verifiers.v1.types import ToolMessage

if TYPE_CHECKING:
    from verifiers.v1.runtimes import Runtime

TOOL_HOOK_SCRIPT = Path(__file__).with_name("tool_hook.mjs").read_text()
TOOL_HOOK_SOURCE = TOOL_HOOK_SCRIPT.encode()
HERMES_TOOL_HOOK_SOURCE = Path(__file__).with_name("hermes_tool_hook.py").read_bytes()


async def install_tool_hook(
    runtime: "Runtime",
    path: str,
    url: str,
    secret: str,
    source: bytes = TOOL_HOOK_SOURCE,
) -> dict[str, str]:
    """Write a native-hook bridge with one-shot connection credentials."""
    await runtime.write(path, source)
    # The random path cannot be pre-seeded, and shell noclobber makes creation
    # exclusive before any credential bytes are read from stdin.
    credentials_path = f"{path}.{uuid.uuid4().hex}.credentials"
    payload = json.dumps({"url": url, "secret": secret}).encode()
    result = await runtime.run_with_input(
        [
            "sh",
            "-c",
            'umask 077; set -C; head -c "$1" > "$2"',
            "write-tool-credentials",
            str(len(payload)),
            credentials_path,
        ],
        {},
        payload,
    )
    if result.exit_code != 0:
        raise RuntimeError(
            "failed to write tool interception credentials privately: "
            f"{result.stderr.strip()[-500:]}"
        )
    return {"VF_TOOL_INTERCEPTION_CONFIG": credentials_path}


class ToolRewriteCapabilities(BaseModel):
    content: Literal["text", "multimodal"] = "multimodal"
    image_urls: Literal["none", "data", "all"] = "all"
    max_text_utf16_units: int | None = Field(default=None, ge=0)
    allow_empty_text: bool = True
    allow_empty_parts: bool = True
    preserve_single_text_part: bool = True


class ToolHookRequest(BaseModel):
    phase: Literal["before", "after", "after_failure"]
    rewrite: ToolRewriteCapabilities | None
    message: ToolMessage
