"""Pi harness using the upstream pi-acp adapter."""

import json
import shlex
from pathlib import Path

from verifiers.v1.acp import ACPConfig, ACPHarness
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import (
    DialectName,
    HarnessConfig,
    PinnedVersion,
    skill_destination,
)
from verifiers.v1.harnesses.utils.install import ensure_installed, remove_dir
from verifiers.v1.harnesses.utils.remote import native_remote_tools
from verifiers.v1.harnesses.utils.replay import drop_blank_text
from verifiers.v1.harnesses.utils.sandbox_tools import RemoteToolsSpec
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace
from verifiers.v1.types import Message
from verifiers.v1.utils.logging import get_logger

logger = get_logger(__name__)

KEY_VAR = "PI_INTERCEPT_KEY"

PI_DIR = "/var/tmp/vf-pi"
STATE_ROOT = "/tmp/vf-pi-runs"
PACKAGES_DIR = f"{PI_DIR}/mcp"
PI_BIN = f"{PACKAGES_DIR}/node_modules/.bin/pi"
ACP_VERSION = "0.0.34"
ACP_BIN = f"{PACKAGES_DIR}/node_modules/.bin/pi-acp"
ACP_COMMAND = ["node", ACP_BIN]

GATE_EXTENSION = (Path(__file__).resolve().parent / "gate.mjs").read_text()

INSTALL = r"""
set -e
packages=/var/tmp/vf-pi/mcp

versions="$AGENT_PI_VERSION:$AGENT_PI_ACP_VERSION"
if [ "$(cat "$packages/.versions" 2>/dev/null)" != "$versions" ]; then
    npm install --prefix "$packages" --ignore-scripts --no-audit --no-fund --omit=dev \
        "@earendil-works/pi-coding-agent@$AGENT_PI_VERSION" \
        "pi-acp@$AGENT_PI_ACP_VERSION" >/dev/null
    printf %s "$versions" > "$packages/.versions"
fi
"""


class PiHarnessConfig(HarnessConfig):
    version: PinnedVersion = "1.5.2"
    """Pi release to install, pinned for reproducibility."""
    dialect: DialectName = "chat_completions"
    supports_developer_role: bool | None = None
    """Override Pi's chat-completions role detection for custom model endpoints."""


class PiHarness(ACPHarness[PiHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_TOOL_INTERCEPTION = True
    # Pi's project skill discovery is trust-gated (a prompt print mode can't answer),
    # so the installed skills are passed explicitly via `--skill` at launch.
    SUPPORTS_SKILLS = True

    # pi-ai drops whitespace-only assistant text on replay over chat completions and
    # Anthropic messages, not over Responses. Recheck on version bumps:
    # https://github.com/earendil-works/pi/blob/v1.5.2/packages/ai/src/api/openai-completions.ts#L1295
    # https://github.com/earendil-works/pi/blob/v1.5.2/packages/ai/src/api/anthropic-messages.ts#L1704
    def replay_view(self, message: Message, dialect: DialectName | None) -> Message:
        if dialect in ("chat_completions", "anthropic_messages"):
            return drop_blank_text(message)
        return message

    async def remote_tools(self, runtime: Runtime, data: TaskData) -> RemoteToolsSpec:
        return native_remote_tools("pi", self.config.disabled_tools)

    async def setup(self, runtime: Runtime) -> None:
        logger.debug(
            "Ensuring Pi %s and pi-acp %s are installed",
            self.config.version,
            ACP_VERSION,
        )
        await ensure_installed(
            runtime,
            directory=PI_DIR,
            ready=(
                f'[ -x "{ACP_BIN}" ] && '
                f'[ "$(cat "{PACKAGES_DIR}/.versions" 2>/dev/null)" = '
                f'"{self.config.version}:{ACP_VERSION}" ]'
            ),
            install=INSTALL,
            env={
                "AGENT_PI_VERSION": self.config.version,
                "AGENT_PI_ACP_VERSION": ACP_VERSION,
            },
            label="pi",
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
        system_prompt, prompt = self.resolve_prompt(data)
        if runtime._execution_bridge is None:
            raise RuntimeError("Pi requires an execution bridge")
        agent_dir = f"{STATE_ROOT}/{trace.id}"
        skills_dir = f"{agent_dir}/skills"
        await self.install_skills(runtime, skills_dir)
        if self.config.skills:
            # Pi puts skill file paths in its system prompt for the model to `read`.
            await self.install_skills(runtime._execution_bridge.runtime, skills_dir)
        reasoning = ctx.sampling.reasoning_effort not in (
            None,
            "none",
        ) or ctx.model.rsplit("/", 1)[-1].startswith(("gpt-5", "o1", "o3", "o4"))
        provider, separator, model = ctx.model.partition("/")
        if not separator:
            provider, model = "openai", ctx.model
        api = {
            "chat_completions": "openai-completions",
            "responses": "openai-responses",
            "anthropic_messages": "anthropic-messages",
        }[self.config.dialect]
        base_url = (
            endpoint.removesuffix("/v1")
            if self.config.dialect == "anthropic_messages"
            else endpoint
        )
        model_config = {
            "id": model,
            "reasoning": reasoning,
            "input": ["text", "image"],
            **(
                {"compat": {"sessionAffinityFormat": "openai-nosession"}}
                if self.config.dialect == "responses"
                else {}
            ),
        }
        if (
            self.config.dialect == "chat_completions"
            and self.config.supports_developer_role is not None
        ):
            model_config["compat"] = {
                "supportsDeveloperRole": self.config.supports_developer_role
            }
        models = {
            "providers": {
                provider: {
                    "baseUrl": base_url,
                    "api": api,
                    "apiKey": f"${KEY_VAR}",
                    "models": [model_config],
                }
            }
        }
        await runtime.write(f"{agent_dir}/models.json", json.dumps(models).encode())

        if mcp_urls:
            mcp_headers = self.config.resolve_mcp_headers(mcp_urls)
            mcp = {
                "mcpServers": {
                    name: {
                        "url": url,
                        "headers": mcp_headers.get(name, {}),
                        "exposure": "direct",
                    }
                    for name, url in mcp_urls.items()
                }
            }
            await runtime.write(f"{agent_dir}/mcp.json", json.dumps(mcp).encode())

        env = {
            **self.config.resolved_env,
            KEY_VAR: secret,
            "PI_CODING_AGENT_DIR": agent_dir,
            "PI_OFFLINE": "1",
            "PI_TELEMETRY": "0",
        }
        skill_args = [
            arg
            for skill in self.config.skills
            for arg in ("--skill", skill_destination(skill, skills_dir))
        ]
        pi_args = [
            PI_BIN,
            "--no-approve",
            "--provider",
            provider,
            "--model",
            model,
            *skill_args,
        ]
        if self.config.disabled_tools:
            pi_args += ["--exclude-tools", ",".join(self.config.disabled_tools)]
        if system_prompt:
            # Pi reads a path argument as the file's contents, and rereads it on
            # resource reloads, so the file stays until cleanup.
            path = f"{agent_dir}/system-prompt.md"
            await runtime.write(path, system_prompt.encode())
            pi_args += ["--append-system-prompt", path]
        pi_wrapper = f"{agent_dir}/pi"
        await runtime.write(
            pi_wrapper,
            f'#!/bin/sh\nexec node {shlex.join(pi_args)} "$@"\n'.encode(),
        )
        await runtime.run(["chmod", "+x", pi_wrapper], {})
        env["PI_ACP_PI_COMMAND"] = pi_wrapper
        return ACPConfig(
            env=env,
            command=ACP_COMMAND,
            prompt=prompt,
            # Pi reads the task-scoped MCP configuration from its agent directory.
            mcp_urls={},
        )

    async def cleanup(self, trace: Trace, runtime: Runtime) -> None:
        await remove_dir(runtime, f"{STATE_ROOT}/{trace.id}", "Pi state")

    async def gate_tools(
        self, config: ACPConfig, runtime: Runtime, url: str, secret: str
    ) -> None:
        agent_dir = config.env["PI_CODING_AGENT_DIR"]
        await runtime.write(f"{agent_dir}/gate.mjs", GATE_EXTENSION.encode())
        gated = f"{agent_dir}/pi-gated"
        await runtime.write(
            gated,
            f"#!/bin/sh\nexec {config.env['PI_ACP_PI_COMMAND']} "
            f'--extension {agent_dir}/gate.mjs "$@"\n'.encode(),
        )
        await runtime.run(["chmod", "+x", gated], {})
        config.env["PI_ACP_PI_COMMAND"] = gated
