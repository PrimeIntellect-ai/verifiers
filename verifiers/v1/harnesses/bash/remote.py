"""Typed Bash tool execution in the task workspace, after host policy gating."""

import asyncio
import json
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, StrictStr, ValidationError

from verifiers.v1.errors import HarnessError
from verifiers.v1.runtimes import Runtime
from verifiers.v1.types import ToolCall, ToolMessage


class _BashArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")
    command: StrictStr


class _EditArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")
    path: StrictStr
    old_str: StrictStr
    new_str: StrictStr


_EDIT_SCRIPT = r"""
import sys
from pathlib import Path

path, old, new = sys.argv[1:]
if not path:
    print("error: 'path' is required")
elif not old:
    print("error: 'old_str' must be a non-empty string")
else:
    target = Path(path)
    if not target.is_absolute():
        target = Path.cwd() / target
    if not target.exists():
        print(f"error: {path} not found")
    else:
        try:
            content = target.read_text()
        except Exception as error:
            print(f"error: could not read {path}: {error}")
        else:
            count = content.count(old)
            if count != 1:
                print(f"error: old_str must appear exactly once in {path} (found {count})")
            else:
                try:
                    target.write_text(content.replace(old, new, 1))
                except Exception as error:
                    print(f"error: could not write {path}: {error}")
                else:
                    print(f"Edited {path}")
"""


@dataclass(frozen=True)
class BashRemoteTools:
    runtime: Runtime
    edit: bool

    async def __call__(self, call: ToolCall) -> ToolMessage:
        if call.name not in ({"bash", "edit"} if self.edit else {"bash"}):
            raise HarnessError(f"split bash cannot execute tool {call.name!r}")
        try:
            arguments = json.loads(call.arguments or "{}")
        except json.JSONDecodeError as error:
            content = f"error: invalid JSON in tool arguments ({error}); resend the call with valid JSON"
        else:
            if not isinstance(arguments, dict):
                content = (
                    "error: tool arguments must be a JSON object, got "
                    f"{type(arguments).__name__}; resend as an object"
                )
            else:
                try:
                    parsed = (
                        _BashArgs if call.name == "bash" else _EditArgs
                    ).model_validate(arguments)
                except ValidationError as error:
                    content = f"error: invalid {call.name} arguments ({error.errors()[0]['msg']})"
                else:
                    try:
                        async with asyncio.timeout(3600):
                            if isinstance(parsed, _BashArgs):
                                result = await self.runtime.run(
                                    ["bash", "-c", parsed.command], {}
                                )
                            else:
                                result = await self.runtime.run(
                                    [
                                        "python3",
                                        "-c",
                                        _EDIT_SCRIPT,
                                        parsed.path,
                                        parsed.old_str,
                                        parsed.new_str,
                                    ],
                                    {},
                                )
                    except TimeoutError:
                        content = "error: tool timed out after 3600 seconds"
                    else:
                        content = result.stdout + result.stderr
                        if call.name == "edit":
                            content = content.removesuffix("\n")
        return ToolMessage(tool_call_id=call.id, name=call.name, content=content)
