"""Run Claude Code through the Claude Agent SDK ACP adapter."""

import json
import shlex
from functools import partial
from typing import Literal

from verifiers.v1.acp import ACPConfig, ACPHarness
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import DialectName, HarnessConfig, PinnedVersion
from verifiers.v1.harnesses.node import NODE_BIN_DIR, ensure_node
from verifiers.v1.harnesses.utils.install import ensure_installed, remove_dir
from verifiers.v1.harnesses.utils.replay import drop_blank_text_blocks
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace
from verifiers.v1.types import AssistantMessage, Message, ToolCall

CLAUDE_ACP_DIR = "/var/tmp/vf-claude-agent-acp-{version}-{acp_version}"
PACKAGES_DIR = f"{CLAUDE_ACP_DIR}/packages"
ACP_VERSION = "0.108.1"
CLAUDE_BIN = f"{PACKAGES_DIR}/node_modules/.bin/claude"
ACP_BIN = f"{PACKAGES_DIR}/node_modules/.bin/claude-agent-acp"
CLAUDE_CONFIG_ROOT = ".vf-claude"
# Manual mode routes tool approvals to ACP. Auto needs a separate classifier.
CLAUDE_PERMISSIONS = {"defaultMode": "default"}
ACP_INSTALL = r"""
set -e
export PATH="/var/tmp/vf-node/bin:$PATH"
rm -f {ready}
npm install --prefix {packages} --no-audit --no-fund \
    --omit=dev \
    "@anthropic-ai/claude-code@$AGENT_CLAUDE_CODE_VERSION" \
    "@agentclientprotocol/claude-agent-acp@$AGENT_CLAUDE_ACP_VERSION" >/dev/null
touch {ready}
"""


class ClaudeCodeHarnessConfig(HarnessConfig):
    dialect: Literal["anthropic_messages"] = "anthropic_messages"
    version: PinnedVersion = "2.1.327"
    """Claude Code release to install, pinned for reproducibility."""


class ClaudeCodeHarness(ACPHarness[ClaudeCodeHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_SKILLS = True
    SUPPORTS_TOOL_INTERCEPTION = True

    def replay_view(
        self,
        message: Message,
        dialect: DialectName | None,
        *,
        workdir: str | None = None,
    ) -> Message:
        # Claude Code filters whitespace-only blocks and strips the Bash tool's
        # redundant `cd <cwd> && ` before storing its input for replay.
        message = drop_blank_text_blocks(message)
        if (
            not workdir
            or not isinstance(message, AssistantMessage)
            or not message.tool_calls
        ):
            return message
        prefix = f"cd {workdir} && "
        calls = [self._replay_bash_call(call, prefix) for call in message.tool_calls]
        return (
            message.model_copy(update={"tool_calls": calls})
            if calls != message.tool_calls
            else message
        )

    @staticmethod
    def _replay_bash_call(call: ToolCall, prefix: str) -> ToolCall:
        if call.name != "Bash" or call.namespace is not None or call.type != "function":
            return call
        try:
            arguments = json.loads(call.arguments)
        except json.JSONDecodeError:
            return call
        command = arguments.get("command") if isinstance(arguments, dict) else None
        if isinstance(command, str) and command.startswith(prefix):
            while command.startswith(prefix):
                command = command.removeprefix(prefix)
            arguments["command"] = command
            return call.model_copy(update={"arguments": json.dumps(arguments)})
        return call

    async def setup(self, runtime: Runtime) -> None:
        await ensure_node(runtime)
        versions = {"version": self.config.version, "acp_version": ACP_VERSION}
        directory = CLAUDE_ACP_DIR.format(**versions)
        packages = PACKAGES_DIR.format(**versions)
        claude_bin = CLAUDE_BIN.format(**versions)
        acp_bin = ACP_BIN.format(**versions)
        ready = f"{directory}/.ready"
        script = ACP_INSTALL.replace("{packages}", packages).replace("{ready}", ready)
        await ensure_installed(
            runtime,
            directory=directory,
            ready=f"[ -f {ready} ] && [ -x {claude_bin} ] && [ -x {acp_bin} ]",
            install=script,
            env={
                **self.config.resolved_env,
                "AGENT_CLAUDE_CODE_VERSION": self.config.version,
                "AGENT_CLAUDE_ACP_VERSION": ACP_VERSION,
            },
            label="Claude Agent ACP",
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
        workdir = getattr(runtime.config, "workdir", None)
        if workdir and workdir.startswith("/") and shlex.quote(workdir) == workdir:
            trace._replay_view = partial(self.replay_view, workdir=workdir)
        system_prompt, prompt = self.resolve_prompt(data)
        config_dir = self.config_dir(trace)
        await runtime.write(
            f"{config_dir}/settings.json",
            json.dumps({"permissions": CLAUDE_PERMISSIONS}).encode(),
        )
        await self.install_skills(runtime, f"{config_dir}/skills")
        versions = {"version": self.config.version, "acp_version": ACP_VERSION}
        session_meta = {
            "claudeCode": {
                "options": {
                    "strictMcpConfig": True,
                    "disallowedTools": self.config.disabled_tools or [],
                    # Evaluations do not need a model-generated next-user prompt.
                    "promptSuggestions": False,
                    # A supplied title prevents the adapter's background title-model call.
                    "extraArgs": {"name": trace.id},
                }
            },
            **({"systemPrompt": {"append": system_prompt}} if system_prompt else {}),
        }
        env = {
            **self.config.resolved_env,
            "ANTHROPIC_BASE_URL": endpoint.removesuffix("/v1"),
            "ANTHROPIC_API_KEY": secret,
            "ANTHROPIC_MODEL": ctx.model,
            "CLAUDE_CODE_EXECUTABLE": CLAUDE_BIN.format(**versions),
            "CLAUDE_CONFIG_DIR": config_dir,
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            # Discovering deferred tools changes the tool schema mid-trace.
            "ENABLE_TOOL_SEARCH": "false",
            "DISABLE_AUTOUPDATER": "1",
            "IS_SANDBOX": "1",
        }
        return ACPConfig(
            env=env,
            command=[f"{NODE_BIN_DIR}/node", ACP_BIN.format(**versions)],
            prompt=prompt or "",
            session_meta=session_meta,
        )

    async def gate_tools(
        self, config: ACPConfig, runtime: Runtime, url: str, secret: str
    ) -> None:
        # An `ask` rule on every tool makes Claude Code raise a permission request for
        # each call, read-only tools included; the ACP adapter hands it to the runner.
        # `CLAUDE_CONFIG_DIR` holds the user settings, which the adapter loads by default.
        await runtime.write(
            f"{config.env['CLAUDE_CONFIG_DIR']}/settings.json",
            json.dumps({"permissions": {**CLAUDE_PERMISSIONS, "ask": ["*"]}}).encode(),
        )

    async def cleanup(self, trace: Trace, runtime: Runtime) -> None:
        await remove_dir(runtime, self.config_dir(trace), "Claude config")

    @staticmethod
    def config_dir(trace: Trace) -> str:
        return f"{CLAUDE_CONFIG_ROOT}/{trace.id}"
