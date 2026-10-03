"""The grader's tool onto the candidates' own boxes, each restored on first use from
the checkpoint its rollout took when the agent finished (`agent.checkpoint_on_finish`)."""

import asyncio
import contextlib
import time
from collections import OrderedDict

from pydantic import Field

import verifiers.v1 as vf
from verifiers.v1.runtimes import PrimeConfig, Runtime, provision_runtime

OUTPUT_CHARS = 8000
"""Command output is cut to its head and tail beyond this."""
COMMAND_SECONDS = 600


class WorkspacesConfig(vf.ToolsetConfig):
    max_open: int = 4
    """Candidate boxes open at once; opening one more closes the least recently used."""


class GradeState(vf.State):
    restores: list[float] = Field(default_factory=list)
    """Seconds each candidate box took to restore, in order."""


def cut(text: str, limit: int = OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return (
        f"{text[:half]}\n[... {len(text) - 2 * half} chars elided ...]\n{text[-half:]}"
    )


class Workspaces(vf.Toolset[WorkspacesConfig, GradeState]):
    TOOL_PREFIX = None

    async def setup(self) -> None:
        self.boxes: OrderedDict[str, tuple[Runtime, contextlib.AsyncExitStack]] = (
            OrderedDict()
        )
        self.locks: dict[str, asyncio.Lock] = {}
        self.restores: list[float] = []
        self._exit_stack.push_async_callback(self.close_all)

    async def setup_task(self, data) -> None:
        """`data` is the rollout's `GroupGradeData`."""
        self.checkpoints = {c.label: c.checkpoint for c in data.candidates}
        self.env = data.box_env
        inner = vf.WireTaskData.model_validate(data.inner)
        # A restored box keeps the checkpoint's disk.
        resources = inner.resources.model_dump(exclude_none=True, exclude={"disk"})
        self.box = PrimeConfig(allow=[], workdir=inner.workdir, **resources)

    async def close_all(self) -> None:
        await asyncio.gather(
            *(stack.aclose() for _, stack in self.boxes.values()),
            return_exceptions=True,
        )
        self.boxes.clear()

    async def open(self, label: str) -> Runtime:
        async with self.locks.setdefault(label, asyncio.Lock()):
            if label in self.boxes:
                self.boxes.move_to_end(label)
                return self.boxes[label][0]
            while len(self.boxes) >= self.config.max_open:
                _, (_, stack) = self.boxes.popitem(last=False)
                await stack.aclose()
            config = self.box.model_copy(update={"checkpoint": self.checkpoints[label]})
            started = time.monotonic()
            stack = contextlib.AsyncExitStack()
            try:
                runtime = await stack.enter_async_context(
                    provision_runtime(config, env=self.env)
                )
                await runtime.prepare_execution([])
            except BaseException:
                await stack.aclose()
                raise
            self.restores.append(time.monotonic() - started)
            self.boxes[label] = (runtime, stack)
            return runtime

    @vf.tool
    async def candidate_shell(self, label: str, command: str) -> str:
        """Run a shell command in candidate `label`'s own box: its filesystem exactly as
        the candidate left it when it finished, in the task's working directory, with
        the network blocked. The first call for a candidate restores its box (10-30 s);
        later calls reuse it, so changes persist. Returns the exit code and the output
        (head and tail beyond 8000 characters)."""
        if label not in self.checkpoints:
            return f"unknown candidate {label!r}; labels: {', '.join(self.checkpoints)}"
        if self.checkpoints[label] is None:
            return f"{label} has no box to restore; grade it from its files"
        try:
            runtime = await self.open(label)
        except Exception as e:  # noqa: BLE001 - the grader falls back to the files
            return f"could not restore {label}'s box ({e}); grade it from its files"
        finally:
            # One server per rollout: its own list is the whole truth.
            self.state.restores = list(self.restores)
        try:
            async with asyncio.timeout(COMMAND_SECONDS):
                result = await runtime.run(["sh", "-c", command], {})
        except TimeoutError:
            return f"timed out after {COMMAND_SECONDS} s"
        output = cut(
            result.stdout + (f"\n[stderr]\n{result.stderr}" if result.stderr else "")
        )
        return f"exit code {result.exit_code}\n{output}"


if __name__ == "__main__":
    Workspaces.run()
