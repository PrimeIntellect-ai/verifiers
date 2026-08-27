"""RLM over ACP, with MCP tools exposed as pre-imported IPython skills."""

import logging
import shlex
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.acp import ACPConfig, ACPHarness, ACPTurn, JsonObject
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

BuiltinSkill = Literal["edit", "search"]

RLM_REPO = "github.com/PrimeIntellect-ai/nano-rlm.git"
RLM_DIR = "/tmp/vf-rlm"
RLM_BIN = f"{RLM_DIR}/bin/rlm"
SKILLS_DIR = "/task/rlm-skills"
RLM_STATE_DIR = ".vf-rlm"
RLM_RUNTIME_METADATA_KEY = "ai.prime.rlm/runtime-v1"
RLM_SESSION_METADATA_KEY = "ai.prime.rlm/session-v1"


class _SessionSnapshot(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    session_id: str = Field(pattern=r"^[A-Za-z0-9._:-]{1,128}$")
    metrics: dict[str, int | float]


class CompactionConfig(BaseConfig):
    """Context compaction policy for the RLM agent loop."""

    summarize_at_tokens: PositiveInt | None = None
    """Compact at this token count. When unset, use 90% of the model context window when
    the provider advertises it."""


class RLMHarnessConfig(HarnessConfig):
    version: str = Field(default="ac8fdb0", min_length=1)
    """Git ref (branch, tag, or commit) of nano-rlm to install."""
    max_depth: int = 0
    """Recursion depth RLM may spawn sub-harnesses to."""
    builtin_skills: list[BuiltinSkill] = Field(default_factory=list)
    """Built-in rlm skills to enable (RLM_SKILLS), e.g. `["edit"]`; empty enables none.
    The tool set is fixed (ipython); the base `skills` field takes SKILL.md paths."""
    compaction: CompactionConfig | None = None
    """Context compaction policy. Set an empty config to use automatic thresholds."""

    @model_validator(mode="after")
    def reject_disabled_tools(self) -> "RLMHarnessConfig":
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

    def _runtime_metadata(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        system_prompt: str | None,
    ) -> JsonObject:
        compaction = self.config.compaction
        payload = {
            "session_id": trace.id,
            "model": ctx.model,
            "provider": {
                "base_url": endpoint,
                "api_key": secret,
            },
            "policy": {
                "max_depth": self.config.max_depth,
                "compaction": compaction is not None,
                "summarize_at_tokens": (
                    compaction.summarize_at_tokens if compaction else None
                ),
                "max_concurrent_subagents": max(4, self.config.max_depth),
            },
            "system_prompt_path": None,
            "append_to_system_prompt": system_prompt,
            "skills": list(self.config.builtin_skills),
            "kernel_env": runtime.env,
            "search_api_key": self.config.resolved_env.get("SERPER_API_KEY"),
        }
        return {RLM_RUNTIME_METADATA_KEY: payload}

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
            env={**self.config.resolved_env, "RLM_HOME": self._home(trace)},
            command=[RLM_BIN, "--acp"],
            prompt=prompt,
            session_meta=self._runtime_metadata(
                ctx, trace, runtime, endpoint, secret, system_prompt
            ),
        )

    def acp_turn_result(self, trace: Trace, result: ACPTurn) -> None:
        snapshot = _SessionSnapshot.model_validate(
            result.response_metadata.get(RLM_SESSION_METADATA_KEY)
        )
        if snapshot.session_id != trace.id:
            raise ValueError("RLM session snapshot does not match the rollout")
        trace.record_metrics(snapshot.metrics)

    async def cleanup(self, trace: Trace, runtime: Runtime) -> None:
        await runtime.run(["rm", "-rf", f"{RLM_STATE_DIR}/{trace.id}"], {})

    @staticmethod
    def _home(trace: Trace) -> str:
        return f"{RLM_STATE_DIR}/{trace.id}/home"
