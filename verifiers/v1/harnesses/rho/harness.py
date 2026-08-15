"""Rho: pi's four tools plus run_code — services, sub-LLM calls, and subagents live in code."""

import asyncio
import contextlib
import io
import json
import tarfile
from pathlib import Path

from verifiers.v1.acp import ACPConfig, ACPHarness
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

PROGRAM_SOURCE = (Path(__file__).resolve().parent / "program.py").read_text()

NATIVE_TOOLS = ("read", "write", "edit", "bash", "run_code")

RUNTIME_DIAGNOSTICS = "/tmp/.rho/runtime.json"
"""Where the program leaves its stats (mirrors program.py); picked up onto the trace at
cleanup. The program rewrites it after every segment — one process per rollout, so the
counters are cumulative."""


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


class RhoHarness(ACPHarness[RhoHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    # One live process per rollout: later turns land in the same conversation, so the
    # kernel namespace, named agent() sessions, and the transcript file all persist.
    SUPPORTS_RESUME = True
    EXECUTES_CODE = True
    NEEDS_CONTAINER = True

    def tool_surface(self) -> list[str]:
        # Seat shaping goes through the base `disabled_tools`: a ptc-style service
        # seat is disabled_tools=["read", "write", "edit"]. RHO_TOOLS is the one
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
        await super().setup(runtime)
        await runtime.prepare_uv_script(PROGRAM_SOURCE, self.config.resolved_env)

    async def prepare_acp(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
    ) -> ACPConfig:
        system_prompt, prompt = self.resolve_prompt(data)
        # Stage the task's `inputs/` tree — the file-based taskset convention. A session
        # starts once per rollout, so this always runs on a fresh workspace. One tarball,
        # one upload, one extraction: a per-file loop costs 2N gateway round-trips on
        # the prime runtime.
        if (raw := getattr(data, "dir", None)) and (
            inputs := Path(raw) / "inputs"
        ).is_dir():
            buffer = io.BytesIO()

            def build_tar() -> None:
                with tarfile.open(fileobj=buffer, mode="w") as tar:
                    tar.add(inputs, arcname="inputs")

            await asyncio.to_thread(build_tar)
            archive = f".vf-inputs-{trace.id}.tar"
            await runtime.write(archive, buffer.getvalue())
            staged = await runtime.run(
                ["sh", "-c", f"tar -xf {archive} && rm -f {archive}"], {}
            )
            if staged.exit_code != 0:
                raise RuntimeError(
                    f"inputs staging failed (exit {staged.exit_code}): "
                    f"{staged.stderr.strip()[-300:]}"
                )
        env = {
            **self.config.resolved_env,
            # The program pops RHO_* on startup, so the secret never reaches bash
            # children (the kernel builds its env from an allowlist regardless).
            "RHO_BASE_URL": endpoint,
            "RHO_API_KEY": secret,
            "RHO_MODEL": ctx.model,
            "RHO_TOOLS": ",".join(self.tool_surface()),
            "RHO_CONTEXT_BUDGET_TOKENS": str(self.config.context_budget_tokens),
        }
        # One effort channel: the framework's sampling config (which interception
        # overlays on every call anyway); no rho-only knob to fight it.
        if ctx.sampling.reasoning_effort:
            env["RHO_EFFORT"] = ctx.sampling.reasoning_effort
        if self.config.subagents:
            env["RHO_SUBAGENTS"] = "1"
        # One cap, owned by the framework: the box spends the budget it is measured
        # against instead of walking into a refused call. The budget spans the whole
        # session — every segment's turns draw on it — and the session opens on a
        # fresh trace, so the full allowance is the remaining allowance.
        if trace.agent is not None and trace.agent.config.max_turns:
            env["RHO_MAX_TURNS"] = str(trace.agent.config.max_turns)
        program = await runtime.prepare_uv_script(
            PROGRAM_SOURCE, self.config.resolved_env
        )
        # The task system prompt rides session_meta, NOT ACPConfig.system_prompt: the
        # runner folds the latter into the first prompt's text blocks, while rho builds
        # its own system message at session/new.
        return ACPConfig(
            env=env,
            command=program,
            prompt=prompt,
            session_meta={"system_prompt": system_prompt} if system_prompt else None,
        )

    async def cleanup(self, trace: Trace, runtime: Runtime) -> None:
        # Diagnostics ride the rollout's end — the session is closed by now, but the
        # runtime outlives it (the RLM harness relies on the same ordering). Failures
        # must never mask teardown.
        with contextlib.suppress(Exception):
            trace.info["rho"] = json.loads(await runtime.read(RUNTIME_DIAGNOSTICS))
