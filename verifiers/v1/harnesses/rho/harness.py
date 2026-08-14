"""Rho: pi's four tools plus run_code — services, sub-LLM calls, and subagents live in code."""

import json
from pathlib import Path

from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

PROGRAM_SOURCE = (Path(__file__).resolve().parent / "program.py").read_text()

NATIVE_TOOLS = ("read", "write", "edit", "bash", "run_code")


class RhoHarnessConfig(HarnessConfig):
    edit: bool = True
    """Offer the `edit` tool (exact once-only string replacement, batched)."""

    run_code: bool = True
    """Offer `run_code` (persistent Python kernel; service tools, completion(), agent()).
    Off is a pure-pi seat. Finer cuts go through the base `disabled_tools`."""

    subagents: bool = False
    """Expose `agent(...)` inside run_code. Subagents run sequentially with a blank
    context at depth 1; every subagent turn spends the same trace turn budget."""

    max_subagents: int = 16
    """Spawn cap per rollout, so exhaustion is a legible error instead of a
    mid-fleet refusal."""

    compact_tool: bool = False
    """Offer a zero-param `compact` tool that schedules a context checkpoint before the
    next work turn (Codex `new_context` precedent). Enabling it also disclosed the
    current context-token figure each turn — a model can only time compaction well if
    it can see the clock."""

    effort: str = ""
    """Reasoning effort forwarded to the model (empty = provider default)."""

    context_budget_tokens: int = 150_000
    """Prompt-token threshold that triggers checkpoint compaction. A training knob, not
    just a safety net: compacting earlier than the physical limit is how the skill gets
    practiced. 0 disables (tests only)."""

    history_file: bool = True
    """Write the transcript a checkpoint replaces to a greppable file named in the
    framing message. Recoverable history licenses tighter summaries — the checkpoint
    carries decisions and state, and raw data stays a grep away instead of rotting in
    context. Off for seats where compaction-as-skill is the curriculum."""

    max_compactions: int = 8
    """Checkpoint compactions allowed per rollout; past the cap the transcript grows
    uncompacted."""

    disclose_budget: bool = False
    """Append the remaining-turn-budget line to each turn's last tool result. Off for
    solver seats: a visible clock is pacing information the task never granted."""


class RhoHarness(Harness[RhoHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_RESUME = False
    EXECUTES_CODE = True
    NEEDS_CONTAINER = True

    def tool_surface(self) -> list[str]:
        tools = [name for name in NATIVE_TOOLS if name not in (self.config.disabled_tools or [])]
        if not self.config.edit and "edit" in tools:
            tools.remove("edit")
        if not self.config.run_code and "run_code" in tools:
            tools.remove("run_code")
        return tools

    async def setup(self, runtime: Runtime) -> None:
        await runtime.prepare_uv_script(PROGRAM_SOURCE, self.config.resolved_env)

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
        system_prompt, prompt = self.resolve_text_prompt(data)
        args = [
            f"--base-url={endpoint}",
            f"--api-key={secret}",
            f"--model={ctx.model}",
            f"--system-prompt={system_prompt or ''}",
            f"--prompt={prompt or ''}",
            "--tools=" + ",".join(self.tool_surface()),
        ]
        if self.config.effort:
            args.append(f"--effort={self.config.effort}")
        if self.config.subagents:
            args += ["--subagents", f"--max-subagents={self.config.max_subagents}"]
        if self.config.compact_tool:
            args.append("--compact-tool")
        args.append(f"--context-budget-tokens={self.config.context_budget_tokens}")
        args.append("--history-file" if self.config.history_file else "--no-history-file")
        args.append(f"--max-compactions={self.config.max_compactions}")
        # One cap, owned by the framework: the box spends the budget it is measured
        # against instead of walking into a refused call.
        if trace.agent is not None and trace.agent.config.max_turns:
            args.append(f"--max-turns={trace.agent.config.max_turns}")
        if self.config.disclose_budget:
            args.append("--disclose-budget")
        if mcp_urls:
            args.append(
                "--mcp-config="
                + json.dumps({"mcpServers": {n: {"url": u} for n, u in mcp_urls.items()}})
            )
        program = await runtime.prepare_uv_script(PROGRAM_SOURCE, self.config.resolved_env)
        return await runtime.run_program([*program, *args], self.config.resolved_env)
