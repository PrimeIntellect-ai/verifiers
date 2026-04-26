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
            description=(
                "Safely replace a string in a sandbox file iff it occurs exactly once."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the text file.",
                    },
                    "old_str": {
                        "type": "string",
                        "description": "Old string to replace. This is matched literally and can include newlines.",
                    },
                    "new_str": {
                        "type": "string",
                        "description": 'New string. Use empty string "" to delete.',
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Lines of context in the success snippet.",
                        "default": 3,
                    },
                    "encoding": {
                        "type": "string",
                        "description": "File encoding.",
                        "default": "utf-8",
                    },
                    "backup_suffix": {
                        "type": "string",
                        "description": "If set, write a backup copy before editing.",
                        "default": "",
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": "Do not modify the file; only report what would change.",
                        "default": False,
                    },
                    "expand_tabs": {
                        "type": "boolean",
                        "description": "Expand tabs in file, old_str, and new_str before matching.",
                        "default": False,
                    },
                    "tabsize": {
                        "type": "integer",
                        "description": "Tab size for expand_tabs.",
                        "default": 8,
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
        context_lines: int = 3,
        encoding: str = "utf-8",
        backup_suffix: str = "",
        dry_run: bool = False,
        expand_tabs: bool = False,
        tabsize: int = 8,
    ) -> str:
        runtime = resources.require("sandbox_runtime", SandboxResources)
        sandbox_id = await self.ensure_sandbox(task, state, resources)
        target = self.target_path(path)
        result = await runtime.with_retry(runtime.client.read_file)(sandbox_id, target)
        text = result.content
        base_for_match = text.expandtabs(tabsize) if expand_tabs else text
        old_for_match = old_str.expandtabs(tabsize) if expand_tabs else old_str
        new_for_write = new_str.expandtabs(tabsize) if expand_tabs else new_str
        occurrences = [
            match.start()
            for match in re.finditer(re.escape(old_for_match), base_for_match)
        ]
        if not occurrences:
            return (
                f"No replacement performed: old_str did not appear verbatim in {path}."
            )
        if len(occurrences) > 1:
            lines = [base_for_match.count("\n", 0, idx) + 1 for idx in occurrences]
            return f"No replacement performed. Multiple occurrences of old_str at lines {lines}."
        start = occurrences[0]
        replacement_line = base_for_match.count("\n", 0, start) + 1
        new_content = (
            base_for_match[:start]
            + new_for_write
            + base_for_match[start + len(old_for_match) :]
        )
        snippet = self.render_snippet(
            new_content, replacement_line, new_for_write, context_lines
        )
        if dry_run:
            return f"[DRY-RUN] Would edit {path}\n{snippet}"
        if backup_suffix:
            await self.upload_text(
                runtime, sandbox_id, f"{target}{backup_suffix}", text, encoding
            )
        await self.upload_text(runtime, sandbox_id, target, new_content, encoding)
        return (
            f"The file {path} has been edited successfully.\n"
            f"{snippet}\n"
            "Review the changes and make sure they are as expected."
        )

    async def upload_text(
        self,
        runtime: SandboxResources,
        sandbox_id: str,
        target: str,
        content: str,
        encoding: str,
    ) -> None:
        with tempfile.NamedTemporaryFile("w", delete=False, encoding=encoding) as f:
            f.write(content)
            local_path = f.name
        try:
            await runtime.with_retry(runtime.client.upload_file)(
                sandbox_id, target, local_path
            )
        finally:
            Path(local_path).unlink(missing_ok=True)

    def target_path(self, path: str) -> str:
        if path.startswith("/"):
            return path
        normalized = path.lstrip("/")
        if not self.root_dir:
            return f"/{normalized}"
        return f"{self.root_dir}/{normalized}"

    def render_snippet(
        self,
        content: str,
        replacement_line: int,
        new_str: str,
        context_lines: int | None = None,
    ) -> str:
        context = self.context_lines if context_lines is None else context_lines
        lines = content.split("\n")
        snippet_start = max(1, replacement_line - context)
        snippet_end = min(
            len(lines),
            replacement_line + context + new_str.count("\n"),
        )
        width = len(str(snippet_end))
        snippet = "\n".join(
            f"{idx:>{width}} | {lines[idx - 1]}"
            for idx in range(snippet_start, snippet_end + 1)
        )
        return snippet
