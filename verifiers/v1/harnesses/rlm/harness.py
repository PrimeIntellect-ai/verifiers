"""RLM over ACP, with MCP tools exposed as pre-imported IPython skills."""

import hashlib
import logging
import shlex
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    model_validator,
)
from pydantic_config import BaseConfig

from verifiers.v1.acp import ACPConfig, ACPHarness, ACPTurn, JsonObject
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harnesses.utils.install import ensure_installed
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

BuiltinSkill = Literal["bash", "edit", "fetch", "search"]

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


class CompactionConfig(BaseConfig):
    """Context compaction policy for the RLM agent loop."""

    summarize_at_tokens: PositiveInt | None = None
    """Compact at this token count. When unset, compact when 16k tokens remain below the
    model context window when the provider advertises it."""
    max_compactions: PositiveInt | None = None
    """Compactions per session before the engine stops compacting; `None` =
    nano-rlm's default."""


class RLMHarnessConfig(HarnessConfig):
    version: str = Field(default="f405cd1", min_length=1)
    """Git ref (branch, tag, or commit) of nano-rlm to install. Must know every
    field this harness puts on the wire, i.e. be at least the default ref."""
    max_depth: NonNegativeInt | None = None
    """Recursion depth RLM may spawn sub-agents to; `None` = nano-rlm's default (1).
    Set 0 to disable recursion."""
    builtin_skills: list[BuiltinSkill] = Field(default_factory=list)
    """Built-in rlm skills to enable (RLM_SKILLS), e.g. `["edit"]`; empty enables none.
    The tool set is fixed (ipython); the base `skills` field takes SKILL.md paths."""
    compaction: CompactionConfig | bool | None = None
    """Context compaction: a `[compaction]` section (or `true`) enables it with the
    given settings, `false` disables it (an overflowing session then fails), and
    unset defers to nano-rlm's default (on, automatic thresholds, bounded by its
    default 1M tree-token budget)."""
    max_concurrent_subagents: PositiveInt | None = None
    """Sub-agents running at once per session tree; `None` = nano-rlm's default (4),
    raised to an explicit `max_depth` when needed to keep the policy valid."""
    max_total_turns: PositiveInt | None = None
    """Tree-total turn budget (one turn = one work-loop model call, any engine); every
    engine stops before its next call once spent. `None` = uncapped."""
    max_total_tokens: PositiveInt | None = None
    """Tree-total budget of NEW tokens (completion + uncached prompt) across the session
    tree; once spent every engine stops and no further sub-agents spawn. `None` = unbounded."""
    max_tool_output_bytes: PositiveInt | None = None
    """Byte budget for a single tool result entering the conversation (middle truncation);
    overrides rlm's built-in 20KB default in either direction."""
    append_to_system_prompt: str | None = None
    """Appended to the root engine's system prompt, after the taskset's system prompt."""
    subagent_append_to_system_prompt: str | None = None
    """Append for sub-agent engines that can still delegate (depth >= 1, below
    `max_depth`); unset falls back to `append_to_system_prompt`."""
    leaf_append_to_system_prompt: str | None = None
    """Append for depth-capped leaf engines; unset falls back to the sub-agent append."""

    @model_validator(mode="after")
    def reject_disabled_tools(self) -> "RLMHarnessConfig":
        if self.disabled_tools:
            raise ValueError(
                "the rlm harness has a fixed tool set (ipython) and does not support "
                "`disabled_tools`; use `builtin_skills` to enable built-in skills instead."
            )
        if (
            self.max_depth is not None
            and self.max_concurrent_subagents is not None
            and self.max_concurrent_subagents < self.max_depth
        ):
            raise ValueError(
                "`max_concurrent_subagents` must be at least `max_depth` "
                "(nano-rlm rejects the policy otherwise)."
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
        env = self.config.resolved_env.copy()
        extra_uv_args = env.get("RLM_EXTRA_UV_ARGS", "")
        env["RLM_EXTRA_UV_ARGS"] = f"{extra_uv_args} --with mcp~=1.28".strip()
        await ensure_installed(
            runtime,
            directory=directory,
            ready=f"[ -f {ready} ] && [ -x {binary} ]",
            install=install,
            env=env,
            label="rlm",
        )
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
        max_concurrent_subagents = self.config.max_concurrent_subagents
        if max_concurrent_subagents is None and self.config.max_depth is not None:
            max_concurrent_subagents = max(4, self.config.max_depth)
        policy_knobs: dict[str, Any] = {
            "max_depth": self.config.max_depth,
            "max_concurrent_subagents": max_concurrent_subagents,
            "max_total_turns": self.config.max_total_turns,
            "max_total_tokens": self.config.max_total_tokens,
            "max_tool_output_bytes": self.config.max_tool_output_bytes,
        }
        if isinstance(compaction, bool):
            policy_knobs["compaction"] = compaction
        elif compaction is not None:
            policy_knobs["compaction"] = True
            policy_knobs["summarize_at_tokens"] = compaction.summarize_at_tokens
            policy_knobs["max_compactions"] = compaction.max_compactions
        appends = [
            text
            for text in (system_prompt, self.config.append_to_system_prompt)
            if text
        ]
        payload = {
            "session_id": trace.id,
            "model": ctx.model,
            "provider": {
                "base_url": endpoint,
                "api_key": secret,
            },
            # None = passthrough: the key stays off the wire and nano-rlm's own
            # default applies.
            "policy": {
                key: value for key, value in policy_knobs.items() if value is not None
            },
            "system_prompt_path": None,
            "append_to_system_prompt": "\n\n".join(appends) or None,
            "subagent_append_to_system_prompt": (
                self.config.subagent_append_to_system_prompt
            ),
            "leaf_append_to_system_prompt": self.config.leaf_append_to_system_prompt,
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
                ctx, trace, runtime, endpoint, secret, system_prompt
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
