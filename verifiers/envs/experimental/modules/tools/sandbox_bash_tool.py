from __future__ import annotations

import re

from verifiers.types import Tool

from verifiers.envs.experimental.resources import Resources
from verifiers.envs.experimental.task import Task
from verifiers.envs.experimental.modules.tools.sandbox_tool import SandboxTool


class SandboxBashTool(SandboxTool):
    """Bash tool backed by a sandbox declared through the tools channel."""

    def __init__(
        self,
        name: str = "bash",
        command_prefix: str = "",
        working_dir: str | None = None,
        blocked_commands: set[str] | None = None,
        output_limit: int = 10000,
        completion_marker: str | None = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.command_prefix = command_prefix
        self.working_dir = working_dir
        self.blocked_commands = set(blocked_commands or ())
        self.output_limit = output_limit
        self.completion_marker = completion_marker

    def schema(self) -> Tool:
        return Tool(
            name=self.name,
            description="Execute a bash command in the task sandbox.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Bash command to execute.",
                    }
                },
                "required": ["command"],
            },
        )

    async def __call__(
        self,
        command: str,
        task: Task,
        state,
        resources: Resources,
    ) -> str:
        blocked = self.blocked_command(command)
        if blocked is not None:
            return (
                f"Bash command '{blocked}' is not allowed. "
                "Please use a different command or tool."
            )
        full_command = f"{self.command_prefix} {command}".strip()
        exit_code, output = await self.execute_command(
            full_command,
            task,
            state,
            resources,
            timeout=self.command_timeout,
            working_dir=self.working_dir,
        )
        if self.completion_marker and self.completion_marker in output:
            state["agent_signaled_done"] = True
            state["is_completed"] = True
        if exit_code == -1:
            return output
        return self.format_output(exit_code, output)

    def blocked_command(self, command: str) -> str | None:
        for segment in re.split(r"&&|\|\||;|\|", command):
            first = segment.strip().split()[0] if segment.strip() else ""
            if first in self.blocked_commands:
                return first
        return None

    def format_output(self, exit_code: int, output: str) -> str:
        if len(output) <= self.output_limit:
            return f"<returncode>{exit_code}</returncode>\n<output>\n{output}</output>"
        half = self.output_limit // 2
        head = output[:half]
        tail = output[-half:]
        elided = len(output) - self.output_limit
        return (
            f"<returncode>{exit_code}</returncode>\n"
            "<warning>\nThe output of your last command was too long.\n"
            "Use a more selective command.\n</warning>\n"
            f"<output_head>\n{head}\n</output_head>\n"
            f"<elided_chars>\n{elided} characters elided\n</elided_chars>\n"
            f"<output_tail>\n{tail}\n</output_tail>"
        )
