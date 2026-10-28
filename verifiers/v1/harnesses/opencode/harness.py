"""Run OpenCode 2's native ACP server against interception."""

import json
from pathlib import Path
from typing import Literal

from pydantic import PositiveInt, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.acp import ACPConfig, ACPHarness
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import DialectName, HarnessConfig, PinnedVersion
from verifiers.v1.harnesses.node import NODE_BIN_DIR, NPM_ENV, ensure_node
from verifiers.v1.harnesses.utils.install import ensure_installed, remove_dir
from verifiers.v1.harnesses.utils.replay import drop_blank_text
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace
from verifiers.v1.types import Message

OPENCODE_DIR = "/var/tmp/vf-opencode"
STATE_ROOT = "/tmp/vf-opencode-runs"
KEY_VAR = "VF_OPENCODE_API_KEY"
MCP_READY = (Path(__file__).resolve().parent / "mcp-ready.js").read_bytes()
SHELL_GUARD = (Path(__file__).resolve().parent / "shell-guard.js").read_bytes()
SHELL_CAP = (Path(__file__).resolve().parent / "shell-cap.js").read_bytes()

INSTALL = (
    r"""
set -e
"""
    + NPM_ENV
    + r"""npm install --prefix "$VF_OPENCODE_DIR" --no-audit --no-fund --omit=dev \
    "@opencode/cli@$VF_OPENCODE_VERSION" >/dev/null
"""
)


class OpenCodeCompactionConfig(BaseConfig):
    """OpenCode's auto-compaction, keyed to the model's context window."""

    context_window: PositiveInt
    """The model's context window in tokens (OpenCode's model `limit.context`, and its
    `limit.output`, so a request's `max_completion_tokens` is the room left in the
    window rather than OpenCode's default 32,000)."""
    summarize_at_tokens: PositiveInt | None = None
    """Compact once the conversation reaches this many tokens; must be below
    `context_window`. When unset, OpenCode compacts max(10%, 16k) tokens below it."""
    max_output_tokens: PositiveInt | None = None
    """The provider's largest completion (OpenCode's `limit.output`). Unset, the window:
    a request may then ask for more than a provider serving a smaller output limit
    accepts (Nemotron 3.5 Super: a 1,048,576 window, 131,072 output)."""

    @model_validator(mode="after")
    def _below_window(self) -> "OpenCodeCompactionConfig":
        if (
            self.summarize_at_tokens is not None
            and self.summarize_at_tokens >= self.context_window
        ):
            raise ValueError("summarize_at_tokens must be below context_window")
        if (
            self.max_output_tokens is not None
            and self.max_output_tokens > self.context_window
        ):
            raise ValueError("max_output_tokens must not exceed context_window")
        return self


class OpenCodeHarnessConfig(HarnessConfig):
    version: PinnedVersion = "2.0.26"
    """OpenCode 2 release to install, pinned for reproducibility."""
    dialect: DialectName = "chat_completions"
    disabled_tools: list[str] | None = None
    """OpenCode permission actions to deny, such as shell, read, or edit."""
    compaction: OpenCodeCompactionConfig | None = None
    """Auto-compaction. OpenCode compacts only for a model whose context window it
    knows, and the interception model has none: unset, a rollout never compacts and
    ends at the provider's context limit."""
    location_idle_timeout: str | Literal[False] | None = None
    """OpenCode's `experimental.location_idle_timeout` (2.1 and later): how long a
    workspace may go without durable session activity before OpenCode unloads it and
    interrupts its sessions, e.g. "24 hours", or False never to unload. OpenCode 2.0
    unloads after a fixed 60 minutes, and streamed model output is not activity: a
    single model turn longer than an hour (GLM 5.3 planning or writing a large file at
    ~15 tok/s) ended the run "Session interrupted: inactivity". Unset leaves OpenCode's
    default."""
    input_modalities: list[Literal["text", "image", "audio", "video", "pdf"]] | None = (
        None
    )
    """What the model accepts (OpenCode's model `capabilities.input`), e.g. ["text"] for
    a text-only model. OpenCode sends an image or PDF (a file the agent reads, an MCP
    result, an attachment) only when the model accepts it, otherwise the part becomes
    "ERROR: Cannot read ... (this model does not support image input)". Unset, OpenCode
    assumes a custom model accepts text and images, and a text-only provider rejects
    the request (GLM 5.3: "Unsupported input modality: image"; OpenRouter: "No
    endpoints found that support image input"), which ends the rollout."""
    max_shell_output_bytes: PositiveInt | None = 64 * 1024 * 1024
    """The most of one shell command's output OpenCode keeps; the rest is dropped after
    a note. OpenCode stores a command's whole output on the box's disk, and a command
    that never stops printing filled a 5 GB box before its timeout, killing OpenCode's
    session mid-turn. The model sees only the output's tail (~50 KB) either way. None
    keeps everything (as does a shell other than bash or zsh)."""


class OpenCodeHarness(ACPHarness[OpenCodeHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_SKILLS = True

    # OpenCode drops whitespace-only assistant text on replay over Anthropic messages.
    # Recheck on version bumps:
    # https://github.com/anomalyco/opencode/blob/v2.0.26/packages/ai/src/protocols/anthropic-messages.ts#L929
    def replay_view(self, message: Message, dialect: DialectName | None) -> Message:
        if dialect == "anthropic_messages":
            return drop_blank_text(message)
        return message

    async def setup(self, runtime: Runtime) -> None:
        await ensure_node(runtime)
        directory = f"{OPENCODE_DIR}/{self.config.version}"
        await ensure_installed(
            runtime,
            directory=directory,
            ready=f"[ -x {directory}/node_modules/.bin/opencode ]",
            install=INSTALL,
            env={
                **self.config.resolved_env,
                "VF_OPENCODE_DIR": directory,
                "VF_OPENCODE_VERSION": self.config.version,
            },
            label="OpenCode",
        )
        await super().setup(runtime)

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
        root = f"{STATE_ROOT}/{trace.id}"
        config_dir = f"{root}/config/opencode"
        await self.install_skills(runtime, f"{config_dir}/skills")
        system_prompt, prompt = self.resolve_prompt(data)
        if system_prompt:
            await runtime.write(f"{config_dir}/AGENTS.md", system_prompt.encode())

        package = {
            "chat_completions": "openai-compatible",
            "responses": "openai-compatible/responses",
            "anthropic_messages": "anthropic-compatible",
        }[self.config.dialect]
        model: dict = {"modelID": ctx.model}
        if self.config.input_modalities is not None:
            # OpenCode 2's config schema has `capabilities`; a models.dev-style
            # `modalities` key is silently dropped and leaves images on.
            model["capabilities"] = {
                "input": list(self.config.input_modalities),
                "output": ["text"],
            }
        compaction = self.config.compaction
        if compaction is not None:
            # OpenCode caps each request's output at `limit.output`, 32,000 tokens when
            # unset, and at the room left in the window: with the window as the output
            # limit, only the window bounds it; a provider's own output cap bounds it too.
            model["limit"] = {
                "context": compaction.context_window,
                "output": compaction.max_output_tokens or compaction.context_window,
            }
        settings = {
            "model": "intercept/model",
            "providers": {
                "intercept": {
                    "env": [KEY_VAR],
                    "package": f"@opencode/ai/providers/{package}",
                    "settings": {"baseURL": endpoint},
                    "models": {"model": model},
                }
            },
            "default_agent": "build",
            # Auxiliary title/summary model calls must not become rollout turns.
            "agents": {"title": {"disabled": True}, "summary": {"disabled": True}},
            "permissions": [
                {"action": "*", "resource": "*", "effect": "allow"},
                *(
                    {"action": tool, "resource": "*", "effect": "deny"}
                    for tool in self.config.disabled_tools or []
                ),
            ],
            "update": "disable",
            "share": "disabled",
            "snapshots": False,
        }
        if self.config.location_idle_timeout is not None:
            settings["experimental"] = {
                "location_idle_timeout": self.config.location_idle_timeout
            }
        if compaction is not None and compaction.summarize_at_tokens is not None:
            # OpenCode compacts at `limit.context - buffer` tokens.
            settings["compaction"] = {
                "buffer": compaction.context_window - compaction.summarize_at_tokens
            }
        guard_dir = f"{root}/shell-guard"
        await runtime.write(f"{guard_dir}/index.js", SHELL_GUARD)
        await runtime.write(f"{guard_dir}/cap.js", SHELL_CAP)
        plugins = [
            {
                "package": guard_dir,
                "options": {
                    "maxOutputBytes": self.config.max_shell_output_bytes,
                    "node": f"{NODE_BIN_DIR}/node",
                    "cap": f"{guard_dir}/cap.js",
                },
            }
        ]
        if mcp_urls:
            plugin_dir = f"{root}/mcp-ready"
            await runtime.write(f"{plugin_dir}/index.js", MCP_READY)
            plugins.append(
                {
                    "package": plugin_dir,
                    "options": {
                        "servers": list(mcp_urls),
                        "timeout": int(self.config.tool_timeout * 1000),
                    },
                }
            )
        settings["plugins"] = plugins
        return ACPConfig(
            command=[
                f"{OPENCODE_DIR}/{self.config.version}/node_modules/.bin/opencode",
                "acp",
            ],
            env={
                **self.config.resolved_env,
                KEY_VAR: secret,
                "XDG_CONFIG_HOME": f"{root}/config",
                "XDG_DATA_HOME": f"{root}/data",
                "XDG_CACHE_HOME": f"{root}/cache",
                "XDG_STATE_HOME": f"{root}/state",
                "OPENCODE_CONFIG_DIR": config_dir,
                "OPENCODE_CONFIG_CONTENT": json.dumps(settings),
                "OPENCODE_DISABLE_AUTOUPDATE": "1",
                "OPENCODE_DISABLE_MODELS_FETCH": "1",
            },
            prompt=prompt,
        )

    async def cleanup(self, trace: Trace, runtime: Runtime) -> None:
        await remove_dir(runtime, f"{STATE_ROOT}/{trace.id}", "OpenCode state")
