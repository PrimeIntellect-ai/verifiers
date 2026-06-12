"""The rlm harness: installs the rlm CLI into the runtime and runs the binary.

`RLMHarnessConfig` carries both how to install rlm (repo/branch/token/path) and its
runtime knobs (`max_depth`, `tools`), which rlm reads from `RLM_*` env vars.
"""

import json
import logging

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


class RLMHarnessConfig(HarnessConfig):
    """The rlm CLI harness — how to install rlm and how it should run."""

    id: str = "rlm"

    ref: str = "main"
    """Git ref (branch, tag, or commit) of rlm to install."""
    max_depth: int = 0
    """Recursion depth rlm may spawn sub-harnesses to (RLM_MAX_DEPTH)."""
    tools: list[str] | None = None
    """Built-in rlm tools to enable (RLM_TOOLS); None uses rlm's default set."""


class RLMHarness(Harness[RLMHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_TASK_TOOLS = False

    async def launch(
        self,
        ctx: RolloutContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> ProgramResult:
        system_prompt, instruction = self.resolve_prompt(trace.task)
        # rlm reaches the interception server via OPENAI_BASE_URL/API_KEY (its
        # provider precedence falls back to OPENAI_*), and reads RLM_* for itself.
        env = {
            "OPENAI_BASE_URL": f"{endpoint}/v1",
            "OPENAI_API_KEY": secret,
            "RLM_MODEL": ctx.model,
            "RLM_MAX_DEPTH": str(self.config.max_depth),
            "RLM_HOME": RLM_HOME,
        }
        if system_prompt is not None:
            env["RLM_APPEND_TO_SYSTEM_PROMPT"] = system_prompt
        if self.config.tools is not None:
            env["RLM_TOOLS"] = ",".join(self.config.tools)
        # Install rlm onto PATH only if it isn't already there (no runtime-type
        # checks): a no-op where rlm is present, a fresh install otherwise.
        install = (
            "apt-get update -qq && apt-get install -y -qq git curl && "
            f"git clone https://{RLM_REPO} /tmp/rlm && "
            f"git -C /tmp/rlm checkout {self.config.ref} && "
            "UV_INSTALL_DIR=/usr/local/bin UV_TOOL_BIN_DIR=/usr/local/bin "
            "RLM_CHECKOUT_PATH=/tmp/rlm bash /tmp/rlm/install.sh"
        )
        logger.info("rlm: ensuring rlm is installed (ref=%s)", self.config.ref)
        await runtime.run(
            ["sh", "-c", f"command -v rlm >/dev/null 2>&1 || ({install})"], env
        )
        return await runtime.run(["rlm", instruction], env)

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


def load_harness(config: RLMHarnessConfig) -> RLMHarness:
    return RLMHarness(config)
