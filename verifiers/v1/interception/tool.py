"""Typed wire payload for native harness tool hooks."""

import json
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from verifiers.v1.types import ToolMessage

if TYPE_CHECKING:
    from verifiers.v1.runtimes import Runtime

TOOL_HOOK_SOURCE = Path(__file__).with_name("tool_hook.py").read_text()


async def prepare_tool_hook(runtime: "Runtime") -> str:
    """Install the shared native-hook bridge and return its shell command."""
    return shlex.join(await runtime.prepare_uv_script(TOOL_HOOK_SOURCE))


async def configure_tool_hook(
    runtime: "Runtime",
    path: str,
    url: str | None,
    secret: str,
    adapter: Literal["claude", "codex"],
    events: tuple[str, ...],
) -> dict[str, str]:
    """Install one native hook config and return the bridge environment."""
    if url is None:
        return {}
    handler = {
        "type": "command",
        "command": await prepare_tool_hook(runtime),
        "timeout": 35,
    }
    hooks = {
        "hooks": {event: [{"matcher": "*", "hooks": [handler]}] for event in events}
    }
    await runtime.write(path, json.dumps(hooks).encode())
    return {
        "VF_TOOL_INTERCEPTION_ADAPTER": adapter,
        "VF_TOOL_INTERCEPTION_SECRET": secret,
        "VF_TOOL_INTERCEPTION_URL": url,
    }


class ToolHookRequest(BaseModel):
    phase: Literal["before", "after"]
    can_rewrite: bool
    message: ToolMessage
