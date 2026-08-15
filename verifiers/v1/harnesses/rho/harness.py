"""Rho: pi's four tools plus run_code — services, sub-LLM calls, and subagents live in code."""

import asyncio
import contextlib
import io
import json
import tarfile
from pathlib import Path

from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.harness import Harness
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace
from verifiers.v1.types import Messages

PROGRAM_SOURCE = (Path(__file__).resolve().parent / "program.py").read_text()

NATIVE_TOOLS = ("read", "write", "edit", "bash", "run_code")

RUNTIME_DIAGNOSTICS = "/tmp/.rho/runtime.json"
"""Where the program leaves its stats (mirrors program.py); picked up onto the trace.
Per-segment, last-write-wins on resume — cumulative truth lives on the trace itself."""


class RhoHarnessConfig(HarnessConfig):
    subagents: bool = False
    """Expose `agent(...)` inside run_code. Subagents run sequentially with a blank
    context at depth 1; every subagent turn spends the same trace turn budget."""

    compact_tool: bool = False
    """Add a zero-param `compact` tool to the surface: it schedules a context checkpoint
    before the next work turn (Codex `new_context` precedent). Enabling it also
    discloses the current context-token figure each turn — a model can only time
    compaction well if it can see the clock."""

    context_budget_tokens: int = 150_000
    """Prompt-token threshold that triggers checkpoint compaction. A training knob, not
    just a safety net: compacting earlier than the physical limit is how the skill gets
    practiced. 0 disables (tests only)."""


class RhoHarness(Harness[RhoHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    # A resumed segment relaunches on the accreted conversation: the kernel namespace
    # is fresh (stated to the model), files and the tools tree persist on the runtime.
    SUPPORTS_RESUME = True
    EXECUTES_CODE = True
    NEEDS_CONTAINER = True

    def tool_surface(self) -> list[str]:
        # Seat shaping goes through the base `disabled_tools`: a ptc-style service
        # seat is disabled_tools=["read", "write", "edit"]. `--tools` is the one
        # channel for what the seat offers — compact rides it like everything else.
        tools = [
            name
            for name in NATIVE_TOOLS
            if name not in (self.config.disabled_tools or [])
        ]
        if self.config.compact_tool:
            tools.append("compact")
        return tools

    async def setup(self, runtime: Runtime) -> None:
        await runtime.prepare_uv_script(PROGRAM_SOURCE, self.config.resolved_env)

    async def run(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
        messages: Messages | None = None,
    ) -> None:
        # Stage the task's `inputs/` tree into the workspace — the file-based taskset
        # convention — on the FIRST segment only: a resumed segment's workspace is the
        # model's work in progress, and re-staging would clobber it. One tarball, one
        # upload, one extraction: a per-file loop costs 2N gateway round-trips on the
        # prime runtime, minutes of startup for large trees. (The tarball is built in
        # memory — an accepted limit of the convention; inputs are task-sized.)
        if messages is None:
            task_dir = Path(raw) if (raw := getattr(data, "dir", "") or "") else None
            if task_dir is not None and (inputs := task_dir / "inputs").is_dir():
                buffer = io.BytesIO()

                def build_tar() -> None:
                    with tarfile.open(fileobj=buffer, mode="w") as tar:
                        tar.add(inputs, arcname="inputs")

                await asyncio.to_thread(build_tar)
                archive = f".vf-inputs-{trace.id}.tar"
                await runtime.write(archive, buffer.getvalue())
                result = await runtime.run(
                    ["sh", "-c", f"tar -xf {archive} && rm -f {archive}"], {}
                )
                if result.exit_code != 0:
                    raise RuntimeError(
                        f"inputs staging failed (exit {result.exit_code}): "
                        f"{result.stderr.strip()[-300:]}"
                    )
        try:
            await super().run(
                ctx, trace, runtime, endpoint, secret, mcp_urls, data, messages
            )
        finally:
            # Diagnostics must never mask the actual rollout result.
            with contextlib.suppress(Exception):
                result = await runtime.run(["cat", RUNTIME_DIAGNOSTICS], {})
                if result.exit_code == 0 and result.stdout.strip():
                    trace.info["rho"] = json.loads(result.stdout)

    async def launch(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
    ) -> ProgramResult:
        system_prompt, prompt = self.resolve_prompt(data)
        args = [
            f"--base-url={endpoint}",
            f"--api-key={secret}",
            f"--model={ctx.model}",
            f"--system-prompt={system_prompt or ''}",
            "--tools=" + ",".join(self.tool_surface()),
            f"--context-budget-tokens={self.config.context_budget_tokens}",
        ]
        if isinstance(prompt, str) or prompt is None:
            args.append(f"--prompt={prompt or ''}")
        else:
            # A resumed segment's prompt is the accreted conversation; hand Messages
            # over through a file (base64 images can exceed exec limits).
            path = f".vf-initial-messages-{trace.id}.json"
            await runtime.write(
                path, json.dumps([message_to_wire(m) for m in prompt]).encode()
            )
            args.append(f"--initial-messages-file={path}")
        # One effort channel: the framework's sampling config (which interception
        # overlays on every call anyway); no rho-only knob to fight it.
        if ctx.sampling.reasoning_effort:
            args.append(f"--effort={ctx.sampling.reasoning_effort}")
        if self.config.subagents:
            args.append("--subagents")
        # One cap, owned by the framework: the box spends the budget it is measured
        # against instead of walking into a refused call. A resumed segment gets the
        # REMAINING allowance — the trace has already spent turns against the cap.
        if trace.agent is not None and trace.agent.config.max_turns:
            args.append(
                f"--max-turns={max(0, trace.agent.config.max_turns - trace.num_turns)}"
            )
        if mcp_urls:
            args.append(
                "--mcp-config="
                + json.dumps(
                    {"mcpServers": {n: {"url": u} for n, u in mcp_urls.items()}}
                )
            )
        program = await runtime.prepare_uv_script(
            PROGRAM_SOURCE, self.config.resolved_env
        )
        return await runtime.run_program([*program, *args], self.config.resolved_env)
