"""RLM over ACP, with MCP tools exposed as pre-imported IPython skills."""

import json
import logging
import random
import shlex
from typing import Literal

from pydantic import Field, PositiveInt, model_validator

from verifiers.v1.acp import ACPConfig, ACPHarness
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harnesses.rlm.protocol import RLM_LINEAGE_RESOLVER
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace
from verifiers.v1.utils.decorators import metric

logger = logging.getLogger(__name__)

BuiltinSkill = Literal["edit", "search"]

RLM_REPO = "github.com/PrimeIntellect-ai/nano-rlm.git"
RLM_DIR = "/tmp/vf-rlm"
RLM_BIN = f"{RLM_DIR}/bin/rlm"
SKILLS_DIR = "/task/rlm-skills"
RLM_STATE_DIR = ".vf-rlm"


class RLMHarnessConfig(HarnessConfig):
    version: str = Field(default="main", min_length=1)
    """Git ref (branch, tag, or commit) of nano-rlm to install."""
    max_depth: int = 0
    """Recursion depth rlm may spawn sub-harnesses to (RLM_MAX_DEPTH)."""
    builtin_skills: list[BuiltinSkill] = Field(default_factory=list)
    """Built-in rlm skills to enable (RLM_SKILLS), e.g. `["edit"]`; empty enables none.
    The tool set is fixed (ipython); the base `skills` field takes SKILL.md paths."""
    summarize_at_tokens: PositiveInt | tuple[PositiveInt, PositiveInt] | None = None
    """Auto-compaction threshold (RLM_SUMMARIZE_AT_TOKENS): compact the context once it grows
    past this many tokens. An int is a fixed threshold; a `(lo, hi)` pair draws a per-group
    threshold (seeded by the task index, so a task's rollouts share one draw and tasks vary).
    `None` disables auto-compaction; ints must be positive."""

    @model_validator(mode="after")
    def validate_range(self) -> "RLMHarnessConfig":
        value = self.summarize_at_tokens
        if isinstance(value, tuple) and value[0] > value[1]:
            raise ValueError(
                "`summarize_at_tokens` range must be (lo, hi) with lo <= hi."
            )
        return self

    @model_validator(mode="after")
    def reject_disabled_tools(self) -> "RLMHarnessConfig":
        # rlm's only tool is ipython, which must stay enabled, so there's nothing to disable.
        if self.disabled_tools:
            raise ValueError(
                "the rlm harness has a fixed tool set (ipython) and does not support "
                "`disabled_tools`; use `builtin_skills` to enable built-in skills instead."
            )
        return self


class RLMHarness(ACPHarness[RLMHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_SKILLS = True
    LOGICAL_CALL_RESOLVER = RLM_LINEAGE_RESOLVER

    async def setup(self, runtime: Runtime) -> None:
        # Before the installer: install.sh packages the skills it finds.
        await self.install_skills(runtime, SKILLS_DIR)
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
        env = self.config.resolved_env.copy()
        extra_uv_args = env.get("RLM_EXTRA_UV_ARGS", "")
        env["RLM_EXTRA_UV_ARGS"] = f"{extra_uv_args} --with mcp~=1.28".strip()
        result = await runtime.run(["sh", "-c", guarded], env)
        if result.exit_code != 0:
            raise RuntimeError(f"rlm install failed: {result.stderr.strip()[-500:]}")
        await super().setup(runtime)

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

    def _env(
        self,
        ctx: ModelContext,
        trace: Trace,
        endpoint: str,
        secret: str,
        data: TaskData,
        system_prompt: str | None,
    ) -> dict[str, str]:
        env = {
            **self.config.resolved_env,
            "RLM_BASE_URL": endpoint,
            "RLM_API_KEY": secret,
            "RLM_MODEL": ctx.model,
            "RLM_MAX_DEPTH": str(self.config.max_depth),
            "RLM_HOME": self._home(trace),
            "RLM_SUMMARIZE_AT_TOKENS": self.summarize_threshold(data.idx),
        }
        if system_prompt is not None:
            env["RLM_APPEND_TO_SYSTEM_PROMPT"] = system_prompt
        if self.config.builtin_skills:
            env["RLM_SKILLS"] = ",".join(self.config.builtin_skills)
        return env

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
        return ACPConfig(
            env=self._env(ctx, trace, endpoint, secret, data, system_prompt),
            command=[RLM_BIN, "--acp"],
            prompt=prompt,
        )

    @metric
    async def rlm(self, trace: Trace, runtime: Runtime) -> dict[str, float]:
        # RolloutRun closes the harness session before metrics, which finalizes
        # RLM's meta.json while leaving the harness-owned state available here.
        home = shlex.quote(self._home(trace))
        latest = f'cat "$(ls -t {home}/sessions/*/meta.json | head -1)"'
        result = await runtime.run(["sh", "-c", latest], {})
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

    async def cleanup(self, trace: Trace, runtime: Runtime) -> None:
        await runtime.run(["rm", "-rf", f"{RLM_STATE_DIR}/{trace.id}"], {})

    @staticmethod
    def _home(trace: Trace) -> str:
        return f"{RLM_STATE_DIR}/{trace.id}/home"
