"""Browser tool orchestration in the harness sandbox."""

import json
from dataclasses import dataclass

from verifiers.v1.errors import HarnessError
from verifiers.v1.runtimes import Runtime
from verifiers.v1.types import ToolCall, ToolMessage


@dataclass
class BrowserRemoteTools:
    runtime: Runtime
    program: list[str]

    async def __call__(self, call: ToolCall) -> ToolMessage:
        if call.name != "browser":
            raise HarnessError(f"browser-use has no built-in tool {call.name!r}")
        try:
            arguments = json.loads(call.arguments)
            if not isinstance(arguments, dict) or not isinstance(
                arguments.get("code"), str
            ):
                raise TypeError("browser code must be a string")
        except (json.JSONDecodeError, TypeError) as error:
            content = f"error: {error}"
        else:
            result = await self.runtime.run(
                [*self.program, f"--execute-code={arguments['code']}"], {}
            )
            content = result.stdout + result.stderr
        return ToolMessage(tool_call_id=call.id, name=call.name, content=content)
