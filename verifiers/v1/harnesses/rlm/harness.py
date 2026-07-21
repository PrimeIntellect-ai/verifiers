"""RLM exposes `RLM_MCP_CONFIG` tools as pre-imported IPython skills."""

import json
import logging
import random
import shlex
from typing import Literal

from pydantic import model_validator

from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.clients import ModelContext
from verifiers.v1.decorators import metric
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

BuiltinSkill = Literal["edit", "search"]

RLM_REPO = "github.com/PrimeIntellect-ai/rlm.git"
# rlm writes its session under $RLM_HOME/sessions/<id>/; point it at a workdir-
# relative dir so it stays in the runtime (and is cleaned up with the workdir).
RLM_HOME = ".rlm"
RLM_DIR = "/tmp/vf-rlm"
RLM_BIN = f"{RLM_DIR}/bin/rlm"


class RLMHarnessConfig(HarnessConfig):
    version: str = "main"
    """Git ref (branch, tag, or commit) of rlm to install."""
    max_depth: int = 0
    """Recursion depth rlm may spawn sub-harnesses to (RLM_MAX_DEPTH)."""
    skills: list[BuiltinSkill] = []
    """Built-in rlm skills to enable (RLM_SKILLS), e.g. `["edit"]`; empty enables none.
    The tool set is fixed (ipython); only built-in skills are selectable."""
    summarize_at_tokens: int | tuple[int, int] | None = None
    """Auto-compaction threshold (RLM_SUMMARIZE_AT_TOKENS): compact the context once it grows
    past this many tokens. An int is a fixed threshold; a `(lo, hi)` pair draws a per-group
    threshold (seeded by the task index, so a task's rollouts share one draw and tasks vary).
    `None` disables auto-compaction; ints must be positive."""

    @model_validator(mode="after")
    def validate_limits(self) -> "RLMHarnessConfig":
        value = self.summarize_at_tokens
        if isinstance(value, tuple):
            lo, hi = value
            if lo <= 0 or hi <= 0:
                raise ValueError("`summarize_at_tokens` range bounds must be positive.")
            if lo > hi:
                raise ValueError(
                    "`summarize_at_tokens` range must be (lo, hi) with lo <= hi."
                )
        elif value is not None and value <= 0:
            raise ValueError(
                "`summarize_at_tokens` must be positive, or None to disable."
            )
        return self

    @model_validator(mode="after")
    def reject_disabled_tools(self) -> "RLMHarnessConfig":
        # rlm's only tool is ipython, which must stay enabled, so there's nothing to disable.
        if self.disabled_tools:
            raise ValueError(
                "the rlm harness has a fixed tool set (ipython) and does not support "
                "`disabled_tools`; use `skills` to enable built-in skills instead."
            )
        return self


class RLMHarness(Harness[RLMHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True

    async def setup(self, runtime: Runtime) -> None:
        # install.sh fetches curl/uv itself; add git only when the image lacks it.
        install = (
            "command -v git >/dev/null 2>&1 || "
            "{ apt-get update -qq && apt-get install -y -qq git; } && "
            f"rm -rf /tmp/rlm && git clone https://{RLM_REPO} /tmp/rlm && "
            f"git -C /tmp/rlm checkout {shlex.quote(self.config.version)} && "
            f"UV_INSTALL_DIR={RLM_DIR}/bin UV_TOOL_BIN_DIR={RLM_DIR}/bin "
            f"RLM_CHECKOUT_PATH=/tmp/rlm bash /tmp/rlm/install.sh"
        )
        logger.info("rlm: ensuring rlm is installed (version=%s)", self.config.version)
        ensure = shlex.quote(f"[ -x {RLM_BIN} ] || ({install})")
        guarded = f"mkdir -p {RLM_DIR} && flock {RLM_DIR}/install.lock sh -c {ensure}"
        env = {**self.config.resolved_env, "RLM_HOME": RLM_HOME}
        result = await runtime.run(["sh", "-c", guarded], env)
        if result.exit_code != 0:
            raise RuntimeError(f"rlm install failed: {result.stderr.strip()[-500:]}")

    def summarize_threshold(self, task_idx: int | None) -> str:
        """The `RLM_SUMMARIZE_AT_TOKENS` value: a range draws per-group (seeded by task index —
        0 when unset — so a task's rollouts share one threshold). Always set — "" when disabled —
        so the typed field, not a host var the subprocess runtime would inherit, wins."""
        value = self.config.summarize_at_tokens
        if value is None:
            return ""
        if isinstance(value, tuple):
            lo, hi = value
            return str(random.Random(task_idx or 0).randint(lo, hi))
        return str(value)

    async def launch(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> ProgramResult:
        system_prompt, prompt = self.resolve_prompt(trace.task.data)
        env = {
            **self.config.resolved_env,
            "RLM_BASE_URL": endpoint,
            "RLM_API_KEY": secret,
            "RLM_MODEL": ctx.model,
            "RLM_MAX_DEPTH": str(self.config.max_depth),
            "RLM_HOME": RLM_HOME,
            "RLM_SUMMARIZE_AT_TOKENS": self.summarize_threshold(trace.task.data.idx),
        }
        if system_prompt is not None:
            env["RLM_APPEND_TO_SYSTEM_PROMPT"] = system_prompt
        if self.config.skills:
            env["RLM_SKILLS"] = ",".join(self.config.skills)
        if mcp_urls:
            env["RLM_MCP_CONFIG"] = json.dumps(
                {"mcpServers": {name: {"url": url} for name, url in mcp_urls.items()}}
            )
        return await runtime.run_program([RLM_BIN, "--", prompt], env)

    @metric
    async def rlm(self, trace: Trace, runtime: Runtime) -> dict[str, float]:
        # rlm writes a session meta.json with a rich `metrics` block (compactions,
        # ipython input size, programmatic tool-call counts). There's one top-level
        # session dir (sub-harnesses nest as sub-*/), so the glob matches a single
        # file. Surface its numeric metrics as-is; non-numeric fields (e.g.
        # stop_reason) don't fit the float-only trace metrics, so they're skipped.
        result = await runtime.run(
            ["sh", "-c", f"cat {RLM_HOME}/sessions/*/meta.json"], {}
        )
        if result.exit_code != 0 or not result.stdout.strip():
            return {}
        try:
            meta = json.loads(result.stdout)
        except json.JSONDecodeError:
            return {}
        return {
            key: float(value)
            for key, value in meta.get("metrics", {}).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
