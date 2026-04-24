from __future__ import annotations

import re
import tempfile
from pathlib import Path

from verifiers.types import State, Tool

from verifiers.envs.experimental.channels import SandboxResources
from verifiers.envs.experimental.resources import Resources
from verifiers.envs.experimental.task import Task
from verifiers.envs.experimental.modules.tools.sandbox_tool import SandboxTool


class SandboxEditTool(SandboxTool):
    """Exact string-replacement tool for files inside a sandbox."""

    def __init__(
        self,
        name: str = "edit_via_str_replace",
        root_dir: str = "",
        context_lines: int = 3,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.root_dir = root_dir.rstrip("/")
        self.context_lines = context_lines

    def schema(self) -> Tool:
        return Tool(
            name=self.name,
            description="Replace one exact string occurrence in a sandbox file.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to edit, relative to the task root.",
                    },
                    "old_str": {
                        "type": "string",
                        "description": "Exact existing text to replace.",
                    },
                    "new_str": {
                        "type": "string",
                        "description": "Replacement text.",
                    },
                },
                "required": ["path", "old_str", "new_str"],
            },
        )

    async def __call__(
        self,
        path: str,
        old_str: str,
        new_str: str,
        task: Task,
        state: State,
        resources: Resources,
        encoding: str = "utf-8",
    ) -> str:
        runtime = resources.require("sandbox_runtime", SandboxResources)
        sandbox_id = await self.ensure_sandbox(task, state, resources)
        target = self.target_path(path)
        result = await runtime.with_retry(runtime.client.read_file)(sandbox_id, target)
        text = result.content
        occurrences = [match.start() for match in re.finditer(re.escape(old_str), text)]
        if not occurrences:
            return (
                f"No replacement performed: old_str did not appear verbatim in {path}."
            )
        if len(occurrences) > 1:
            lines = [text.count("\n", 0, idx) + 1 for idx in occurrences]
            return f"No replacement performed. Multiple occurrences of old_str at lines {lines}."
        start = occurrences[0]
        replacement_line = text.count("\n", 0, start) + 1
        new_content = text[:start] + new_str + text[start + len(old_str) :]
        with tempfile.NamedTemporaryFile("w", delete=False, encoding=encoding) as f:
            f.write(new_content)
            local_path = f.name
        try:
            await runtime.with_retry(runtime.client.upload_file)(
                sandbox_id, target, local_path
            )
        finally:
            Path(local_path).unlink(missing_ok=True)
        return self.render_snippet(path, new_content, replacement_line, new_str)

    def target_path(self, path: str) -> str:
        normalized = path.lstrip("/")
        if not self.root_dir:
            return f"/{normalized}"
        return f"{self.root_dir}/{normalized}"

    def render_snippet(
        self, path: str, content: str, replacement_line: int, new_str: str
    ) -> str:
        lines = content.splitlines()
        snippet_start = max(1, replacement_line - self.context_lines)
        snippet_end = min(
            len(lines),
            replacement_line + self.context_lines + new_str.count("\n"),
        )
        width = len(str(snippet_end))
        snippet = "\n".join(
            f"{idx:>{width}} | {lines[idx - 1]}"
            for idx in range(snippet_start, snippet_end + 1)
        )
        return f"The file {path} has been edited successfully.\n{snippet}"
