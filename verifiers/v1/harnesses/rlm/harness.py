"""RLM over ACP, with MCP tools exposed as pre-imported IPython skills."""

import logging
import random
import shlex
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)

from verifiers.v1.acp import ACPConfig, ACPHarness, JsonObject
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


class _ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _UsageSnapshot(_ContractModel):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class _ProgrammaticToolCallSnapshot(_ContractModel):
    python_total: int = Field(ge=0)
    bash_total: int = Field(ge=0)
    by_tool_python: dict[str, int]
    by_tool_bash: dict[str, int]


class _SupervisorSnapshot(_ContractModel):
    subagent_calls: int = Field(ge=0)
    active_subagent_calls: int = Field(ge=0)


class _LimitsSnapshot(_ContractModel):
    max_depth: int = Field(ge=0)
    max_concurrent_subagents: int = Field(gt=0)
    max_subagent_calls: int = Field(gt=0)
    max_tokens: int | None = Field(default=None, gt=0)
    summarize_at_tokens: int | None = Field(default=None, gt=0)
    max_compactions: int | None = Field(default=None, gt=0)
    max_tool_output_chars: int | None = Field(default=None, gt=0)
    allow_git: bool


class _SessionSnapshot(_ContractModel):
    session_id: str = Field(pattern=r"^[A-Za-z0-9._:-]{1,128}$")
    last_stop_reason: str | None
    model: str = Field(min_length=1)
    turns: int = Field(ge=0)
    usage: _UsageSnapshot
    metrics: dict[str, int | float]
    programmatic_tool_call_stats: _ProgrammaticToolCallSnapshot
    supervisor: _SupervisorSnapshot
    limits: _LimitsSnapshot


class RLMHarnessConfig(HarnessConfig):
    version: str = Field(default="c27f8ea151061e31497a5831fda4c168de6d0587", min_length=1)
    """Git ref (branch, tag, or commit) of nano-rlm to install.

    Pinned: every fresh sandbox installs this ref, and the host ACP client
    is itself pinned, so an unpinned default breaks every new sandbox the
    moment nano-rlm main changes the wire contract. Move the pin
    deliberately, together with the client."""
    max_depth: NonNegativeInt = 0
    """Recursion depth RLM may spawn sub-harnesses to."""
    exec_timeout: PositiveInt = 300
    max_output: int = -1
    max_tokens: PositiveInt | None = None
    max_compactions: PositiveInt | None = None
    max_concurrent_subagents: PositiveInt | None = None
    max_subagent_calls: PositiveInt = 64
    max_tool_output_chars: PositiveInt | None = None
    allow_git: bool = False
    sdk_max_retries: NonNegativeInt = 5
    system_prompt_path: str | None = None
    kernel_env: dict[str, str] = Field(default_factory=dict)
    """Task variables intentionally visible to model-controlled kernel code."""
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

    @field_validator("max_output")
    @classmethod
    def validate_max_output(cls, value: int) -> int:
        if value == 0 or value < -1:
            raise ValueError("must be positive, or -1 to disable truncation")
        return value

    @model_validator(mode="after")
    def validate_concurrency(self) -> "RLMHarnessConfig":
        if (
            self.max_concurrent_subagents is not None
            and self.max_concurrent_subagents < self.max_depth
        ):
            raise ValueError("max_concurrent_subagents must be at least max_depth")
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
        endpoint: str,
        secret: str,
        data: TaskData,
        system_prompt: str | None,
    ) -> JsonObject:
        max_concurrent = self.config.max_concurrent_subagents or max(
            4, self.config.max_depth
        )
        payload = {
            "session_id": trace.id,
            "model": ctx.model,
            "provider": {
                "base_url": endpoint,
                "api_key": secret,
                "headers": {},
                "max_retries": self.config.sdk_max_retries,
            },
            "policy": {
                "max_depth": self.config.max_depth,
                "exec_timeout": self.config.exec_timeout,
                "max_output": self.config.max_output,
                "max_tokens": self.config.max_tokens,
                "summarize_at_tokens": self.summarize_threshold(data.idx),
                "max_compactions": self.config.max_compactions,
                "max_concurrent_subagents": max_concurrent,
                "max_subagent_calls": self.config.max_subagent_calls,
                "max_tool_output_chars": self.config.max_tool_output_chars,
                "allow_git": self.config.allow_git,
            },
            "system_prompt_path": self.config.system_prompt_path,
            "append_to_system_prompt": system_prompt,
            "skills": list(self.config.builtin_skills),
            "kernel_env": self.config.kernel_env,
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
                ctx, trace, endpoint, secret, data, system_prompt
            ),
        )

    def acp_close_metrics(self, trace: Trace, metadata: JsonObject) -> dict[str, float]:
        snapshot = _SessionSnapshot.model_validate(
            metadata.get(RLM_SESSION_METADATA_KEY)
        )
        if snapshot.session_id != trace.id:
            raise ValueError("RLM session snapshot does not match the rollout")
        return {name: float(value) for name, value in snapshot.metrics.items()}

    async def cleanup(self, trace: Trace, runtime: Runtime) -> None:
        await runtime.run(["rm", "-rf", f"{RLM_STATE_DIR}/{trace.id}"], {})

    @staticmethod
    def _home(trace: Trace) -> str:
        return f"{RLM_STATE_DIR}/{trace.id}/home"
