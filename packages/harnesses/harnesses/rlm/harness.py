"""The rlm harness: installs the rlm CLI into the runtime and runs the binary.

`RLMHarnessConfig` carries both how to install rlm (repo/branch/token/path) and its
runtime knobs (`max_depth`, `tools`), which rlm reads from `RLM_*` env vars.
"""

import json
import logging
import shlex

from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.clients import RolloutContext
from verifiers.v1.decorators import metric
from verifiers.v1.errors import ProgramError
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

    id: str = "rlm"

    version: str = "main"
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
        system_prompt, prompt = self.resolve_prompt(trace.task)
        # rlm reaches the interception server via OPENAI_BASE_URL/API_KEY (its
        # provider precedence falls back to OPENAI_*), and reads RLM_* for itself.
        env = {
            **self.config.env,
            "OPENAI_BASE_URL": endpoint,
            "OPENAI_API_KEY": secret,
            "RLM_MODEL": ctx.model,
            "RLM_MAX_DEPTH": str(self.config.max_depth),
            "RLM_HOME": RLM_HOME,
        }
        if system_prompt is not None:
            env["RLM_APPEND_TO_SYSTEM_PROMPT"] = system_prompt
        if self.config.tools is not None or self.config.disabled_tools:
            tools = self.config.tools if self.config.tools is not None else ["ipython"]
            disabled_tools = set(self.config.disabled_tools or [])
            env["RLM_TOOLS"] = ",".join(
                tool for tool in tools if tool not in disabled_tools
            )
        # Install rlm only if the binary isn't already there (no runtime-type checks): a no-op
        # where it's present, a fresh install otherwise. install.sh fetches curl/uv (and git,
        # via the runtime's package manager) itself when missing, so the only thing we add is
        # git for our pinned checkout — and only when it's absent, so a non-root host with git
        # already present needs no root.
        install = (
            "command -v git >/dev/null 2>&1 || "
            "{ apt-get update -qq && apt-get install -y -qq git; } && "
            f"rm -rf /tmp/rlm && git clone https://{RLM_REPO} /tmp/rlm && "
            f"git -C /tmp/rlm checkout {self.config.version} && "
            f"UV_INSTALL_DIR={RLM_DIR}/bin UV_TOOL_BIN_DIR={RLM_DIR}/bin "
            f"RLM_CHECKOUT_PATH=/tmp/rlm bash /tmp/rlm/install.sh"
        )
        logger.info("rlm: ensuring rlm is installed (version=%s)", self.config.version)
        # Serialize concurrent rollouts that share one runtime (e.g. subprocess on the host),
        # which otherwise race to clone/install into the same /tmp dirs: the first installs, the
        # rest wait on the lock and find the binary already present.
        ensure = shlex.quote(f"[ -x {RLM_BIN} ] || ({install})")
        guarded = f"mkdir -p {RLM_DIR} && flock {RLM_DIR}/install.lock sh -c {ensure}"
        result = await runtime.run(["sh", "-c", guarded], env)
        if result.exit_code != 0:
            raise ProgramError(f"rlm install failed: {result.stderr.strip()[-500:]}")
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
