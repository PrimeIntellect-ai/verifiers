"""cleanup-tool: a rollout's tool server whose `setup_task` holds a resource with a slow
release, like a sandbox it booted. The release writes the file named by
`$CLEANUP_MARKER`, so a test can see whether it ran when the rollout ended."""

import asyncio
import os
from pathlib import Path

import verifiers.v1 as vf


class CleanupToolset(vf.Toolset[vf.ToolsetConfig]):
    TOOL_PREFIX = "cleanup"

    async def setup_task(self, task) -> None:
        marker = Path(os.environ["CLEANUP_MARKER"])

        async def release() -> None:
            await asyncio.sleep(1)  # as a remote delete would take
            marker.write_text("released")

        self._exit_stack.push_async_callback(release)

    @vf.tool
    def ping(self) -> str:
        """Answer pong."""
        return "pong"


class CleanupTask(vf.Task):
    @classmethod
    def toolsets(cls, config: vf.TaskConfig) -> list[vf.Toolset]:
        return [CleanupToolset(vf.ToolsetConfig())]


if __name__ == "__main__":
    CleanupToolset.run()
