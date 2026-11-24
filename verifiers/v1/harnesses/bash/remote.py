"""Typed Bash tool execution in the task workspace, after host policy gating."""

import asyncio
import json
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, StrictStr, ValidationError

from verifiers.v1.errors import HarnessError, SandboxError
from verifiers.v1.harnesses.utils.remote_files import read_text, run_shell
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


async def _edit(runtime: Runtime, args: _EditArgs) -> str:
    if not args.path:
        return "error: 'path' is required"
    if not args.old_str:
        return "error: 'old_str' must be a non-empty string"
    try:
        original = await read_text(runtime, args.path, normalize=True)
    except (SandboxError, OSError, ValueError) as error:
        return f"error: could not read {args.path}: {error}"
    count = original.count(args.old_str)
    if count != 1:
        return f"error: old_str must appear exactly once in {args.path} (found {count})"
    try:
        await runtime.write(
            args.path, original.replace(args.old_str, args.new_str, 1).encode()
        )
    except (SandboxError, OSError, ValueError) as error:
        return f"error: could not write {args.path}: {error}"
    return f"Edited {args.path}"


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
                                result = await run_shell(
                                    self.runtime, parsed.command, {}
                                )
                                content = result.stdout + result.stderr
                            else:
                                content = await _edit(self.runtime, parsed)
                    except TimeoutError:
                        content = "error: tool timed out after 3600 seconds"
        return ToolMessage(tool_call_id=call.id, name=call.name, content=content)
