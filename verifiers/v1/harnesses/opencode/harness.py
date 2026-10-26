"""Run OpenCode 2's native ACP server against interception."""

import json
from pathlib import Path

from verifiers.v1.acp import ACPConfig, ACPHarness
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import DialectName, HarnessConfig, PinnedVersion
from verifiers.v1.harnesses.node import ensure_node
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

INSTALL = r"""
set -e
export PATH="/var/tmp/vf-node/bin:$PATH"
npm install --prefix "$VF_OPENCODE_DIR" --no-audit --no-fund --omit=dev \
    "@opencode/cli@$VF_OPENCODE_VERSION" >/dev/null
"""


class OpenCodeHarnessConfig(HarnessConfig):
    version: PinnedVersion = "2.0.26"
    """OpenCode 2 release to install, pinned for reproducibility."""
    dialect: DialectName = "chat_completions"
    disabled_tools: list[str] | None = None
    """OpenCode permission actions to deny, such as shell, read, or edit."""


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
        settings = {
            "model": "intercept/model",
            "providers": {
                "intercept": {
                    "env": [KEY_VAR],
                    "package": f"@opencode/ai/providers/{package}",
                    "settings": {"baseURL": endpoint},
                    "models": {"model": {"modelID": ctx.model}},
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
        if mcp_urls:
            plugin_dir = f"{root}/mcp-ready"
            await runtime.write(f"{plugin_dir}/index.js", MCP_READY)
            settings["plugins"] = [
                {
                    "package": plugin_dir,
                    "options": {
                        "servers": list(mcp_urls),
                        "timeout": int(self.config.tool_timeout * 1000),
                    },
                }
            ]
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
