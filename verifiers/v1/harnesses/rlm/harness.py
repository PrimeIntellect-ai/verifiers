"""RLM over ACP, with MCP tools exposed as pre-imported IPython skills."""

import hashlib
import logging
import random
import shlex
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from verifiers.v1.acp import ACPConfig, ACPHarness, ACPTurn, JsonObject
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

BuiltinSkill = Literal["edit", "search"]

RLM_REPO = "github.com/PrimeIntellect-ai/nano-rlm.git"
RLM_CACHE_DIR = "/tmp/vf-rlm"
SKILLS_DIR = "/task/rlm-skills"
RLM_STATE_DIR = ".vf-rlm"
RLM_RUNTIME_METADATA_KEY = "ai.prime.rlm/runtime-v1"
RLM_SESSION_METADATA_KEY = "ai.prime.rlm/session-v1"


class _SessionSnapshot(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    session_id: str = Field(pattern=r"^[A-Za-z0-9._:-]{1,128}$")
    metrics: dict[str, int | float]


class RLMHarnessConfig(HarnessConfig):
    version: str = Field(
        default="e4ad6c590e029b4eda12d73e135d80d3242c3a8e", min_length=1
    )
    """Git ref (branch, tag, or commit) of nano-rlm to install."""
    max_depth: int = 0
    """Recursion depth RLM may spawn sub-harnesses to."""
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
        directory = self._install_dir()
        binary = f"{directory}/bin/rlm"
        checkout = f"{directory}/checkout"
        ready = f"{directory}/.ready"
        # install.sh fetches curl/uv itself; add git only when the image lacks it.
        install = (
            f"rm -f {ready} && "
            "(command -v git >/dev/null 2>&1 || "
            "{ apt-get update -qq && apt-get install -y -qq git; } && "
            f"rm -rf {checkout} && git clone https://{RLM_REPO} {checkout} && "
            f"git -C {checkout} checkout {shlex.quote(self.config.version)} && "
            f"UV_INSTALL_DIR={directory}/bin UV_TOOL_BIN_DIR={directory}/bin "
            f"RLM_CHECKOUT_PATH={checkout} bash {checkout}/install.sh && "
            f"touch {ready})"
        )
        logger.info("rlm: ensuring rlm is installed (version=%s)", self.config.version)
        ensure = shlex.quote(f"[ -f {ready} ] && [ -x {binary} ] || ({install})")
        guarded = (
            f"mkdir -p {directory} && flock {directory}/install.lock sh -c {ensure}"
        )
        env = self.config.resolved_env.copy()
        extra_uv_args = env.get("RLM_EXTRA_UV_ARGS", "")
        env["RLM_EXTRA_UV_ARGS"] = f"{extra_uv_args} --with mcp~=1.28".strip()
        result = await runtime.run(["sh", "-c", guarded], env)
        if result.exit_code != 0:
            raise RuntimeError(f"rlm install failed: {result.stderr.strip()[-500:]}")
        await super().setup(runtime)

    def summarize_threshold(self, task_idx: int | None) -> int | None:
        """Resolve a fixed or per-task compaction threshold."""
        value = self.config.summarize_at_tokens
        if value is None:
            return None
        if isinstance(value, tuple):
            lo, hi = value
            return random.Random(task_idx or 0).randint(lo, hi)
        return value

    def _runtime_metadata(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        data: TaskData,
        system_prompt: str | None,
    ) -> JsonObject:
        payload = {
            "session_id": trace.id,
            "model": ctx.model,
            "provider": {
                "base_url": endpoint,
                "api_key": secret,
            },
            "policy": {
                "max_depth": self.config.max_depth,
                "summarize_at_tokens": self.summarize_threshold(data.idx),
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
            command=[f"{self._install_dir()}/bin/rlm", "--acp"],
            prompt=prompt,
            session_meta=self._runtime_metadata(
                ctx, trace, runtime, endpoint, secret, data, system_prompt
            ),
        )

    def _consume_snapshot(self, trace: Trace, metadata: dict[str, Any]) -> None:
        snapshot = _SessionSnapshot.model_validate(
            metadata.get(RLM_SESSION_METADATA_KEY)
        )
        if snapshot.session_id != trace.id:
            raise ValueError("RLM session snapshot does not match the rollout")
        trace.record_metrics(snapshot.metrics)

    def acp_turn_result(self, trace: Trace, result: ACPTurn) -> None:
        self._consume_snapshot(trace, result.response_metadata)

    def acp_close_result(self, trace: Trace, response_metadata: dict[str, Any]) -> None:
        if RLM_SESSION_METADATA_KEY in response_metadata:
            self._consume_snapshot(trace, response_metadata)

    async def cleanup(self, trace: Trace, runtime: Runtime) -> None:
        await runtime.run(["rm", "-rf", f"{RLM_STATE_DIR}/{trace.id}"], {})

    @staticmethod
    def _home(trace: Trace) -> str:
        return f"{RLM_STATE_DIR}/{trace.id}/home"

    def _install_dir(self) -> str:
        cache_key = hashlib.sha256(self.config.version.encode()).hexdigest()
        return f"{RLM_CACHE_DIR}-{cache_key}"
