"""The rlm harness: installs the rlm CLI into the runtime and runs the binary.

`RLMHarnessConfig` carries both how to install rlm (repo/branch/token/path) and its
runtime knobs (`max_depth`, `skills`), which rlm reads from `RLM_*` env vars.

A task's MCP tool servers are passed to rlm via `RLM_MCP_CONFIG` (a standard `mcpServers`
URL map); rlm exposes each tool as a pre-imported IPython skill the agent calls
programmatically (`await tools_<name>(...)`), rather than via a native MCP client.
"""

import json
import logging
import shlex
from typing import Literal

from pydantic import model_validator

from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.clients import RolloutContext
from verifiers.v1.decorators import metric
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

RLM_REPO = "github.com/PrimeIntellect-ai/rlm.git"
# rlm writes its session under $RLM_HOME/sessions/<id>/; point it at a workdir-
# relative dir so it stays in the runtime (and is cleaned up with the workdir).
RLM_HOME = ".rlm"
RLM_DIR = "/tmp/vf-rlm"
RLM_BIN = f"{RLM_DIR}/bin/rlm"


class RLMHarnessConfig(HarnessConfig):
    """The rlm CLI harness — how to install rlm and how it should run."""

    version: str = "main"
    """Git ref (branch, tag, or commit) of rlm to install."""
    max_depth: int = 0
    """Recursion depth rlm may spawn sub-harnesses to (RLM_MAX_DEPTH)."""
    skills: list[Literal["edit", "search"]] | None = None
    """Built-in rlm skills to enable (RLM_SKILLS), e.g. `["edit"]`; None enables none.
    Validated against the skills rlm ships (see rlm.skills); the tool set is fixed
    (ipython), only built-in skills are selectable."""

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
        env = {**self.config.env, "RLM_HOME": RLM_HOME}
        result = await runtime.run(["sh", "-c", guarded], env)
        if result.exit_code != 0:
            raise RuntimeError(f"rlm install failed: {result.stderr.strip()[-500:]}")

    async def launch(
        self,
        ctx: RolloutContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> ProgramResult:
        system_prompt, prompt = self.resolve_prompt(trace.task)
        env = {
            **self.config.env,
            "RLM_BASE_URL": endpoint,
            "RLM_API_KEY": secret,
            "RLM_MODEL": ctx.model,
            "RLM_MAX_DEPTH": str(self.config.max_depth),
            "RLM_HOME": RLM_HOME,
        }
        if system_prompt is not None:
            env["RLM_APPEND_TO_SYSTEM_PROMPT"] = system_prompt
        if self.config.skills:
            env["RLM_SKILLS"] = ",".join(self.config.skills)
        if mcp_urls:
            env["RLM_MCP_CONFIG"] = json.dumps(
                {"mcpServers": {name: {"url": url} for name, url in mcp_urls.items()}}
            )
        return await runtime.run_program([RLM_BIN, prompt], env)

    @metric
    async def rlm(self, trace: Trace, runtime: Runtime) -> dict[str, float]:
        # rlm writes a session meta.json with a rich `metrics` block (turns, token
        # stats, compactions, tool-call counts). There's one top-level session dir
        # (sub-harnesses nest as sub-*/), so the glob matches a single file. Surface
        # its numeric metrics under an `rlm_` prefix; non-numeric fields (e.g.
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
            f"rlm_{key}": float(value)
            for key, value in meta.get("metrics", {}).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
