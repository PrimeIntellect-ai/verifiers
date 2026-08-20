"""Run Claude Code through the Claude Agent SDK ACP adapter."""

import shlex
from pathlib import Path

from pydantic import Field

from verifiers.v1.acp import ACPConfig, ACPHarness
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.errors import HarnessError
from verifiers.v1.harnesses.node import NODE_BIN_DIR, ensure_node
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

CLAUDE_ACP_DIR = "/var/tmp/vf-claude-agent-acp-{version}-{acp_version}"
PACKAGES_DIR = f"{CLAUDE_ACP_DIR}/packages"
ACP_VERSION = "0.67.0"
claudeVersion = "2.1.232"
CLAUDE_BIN = f"{PACKAGES_DIR}/node_modules/.bin/claude"
ACP_BIN = f"{PACKAGES_DIR}/node_modules/.bin/claude-agent-acp"
acpLib = (
    f"{PACKAGES_DIR}/node_modules/@agentclientprotocol/claude-agent-acp/dist/lib.js"
)
acpIndex = (
    f"{PACKAGES_DIR}/node_modules/@agentclientprotocol/claude-agent-acp/dist/index.js"
)
CLAUDE_CONFIG_ROOT = ".vf-claude"
SKILLS_DIR = ".claude/skills"
wrapperSource = Path(__file__).with_name("wrapper.mjs").read_text()
ACP_INSTALL = r"""
set -e
export PATH="/var/tmp/vf-node/bin:$PATH"
rm -f {ready}
npm install --prefix {packages} --no-audit --no-fund \
    --omit=dev \
    "@anthropic-ai/claude-code@$VF_CLAUDE_CODE_VERSION" \
    "@agentclientprotocol/claude-agent-acp@$VF_CLAUDE_ACP_VERSION" >/dev/null
touch {ready}
"""


class ClaudeCodeHarnessConfig(HarnessConfig):
    version: str = Field(default=claudeVersion, pattern=r"^[A-Za-z0-9._+-]+$")
    """Claude Code release to install, pinned for reproducibility."""


class ClaudeCodeHarness(ACPHarness[ClaudeCodeHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_SKILLS = True
    SUPPORTS_PRE_TOOL_INTERCEPTION = True
    SUPPORTS_POST_TOOL_INTERCEPTION = True

    async def setup(self, runtime: Runtime) -> None:
        await self.install_skills(runtime, SKILLS_DIR)
        await ensure_node(runtime)
        versions = {"version": self.config.version, "acp_version": ACP_VERSION}
        directory = CLAUDE_ACP_DIR.format(**versions)
        packages = PACKAGES_DIR.format(**versions)
        claude_bin = CLAUDE_BIN.format(**versions)
        acp_bin = ACP_BIN.format(**versions)
        ready = f"{directory}/.ready"
        script = ACP_INSTALL.replace("{packages}", packages).replace("{ready}", ready)
        ensure = shlex.quote(
            f"[ -f {ready} ] && [ -x {claude_bin} ] && [ -x {acp_bin} ] || ({script})"
        )
        acp_guarded = (
            f"mkdir -p {directory} && "
            f'"$(command -v flock || command -v lockf)" {directory}/install.lock '
            f"sh -c {ensure}"
        )
        acp_result = await runtime.run(
            ["sh", "-c", acp_guarded],
            {
                **self.config.resolved_env,
                "VF_CLAUDE_CODE_VERSION": self.config.version,
                "VF_CLAUDE_ACP_VERSION": ACP_VERSION,
            },
        )
        if acp_result.exit_code != 0:
            detail = (acp_result.stderr or acp_result.stdout).strip()[-500:]
            raise RuntimeError(f"Claude Agent ACP install failed: {detail}")
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
        config_dir = self.config_dir(trace)
        versions = {"version": self.config.version, "acp_version": ACP_VERSION}
        session_meta = {
            "claudeCode": {
                "options": {
                    "strictMcpConfig": True,
                    "disallowedTools": self.config.disabled_tools or [],
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
            "DISABLE_AUTOUPDATER": "1",
            "IS_SANDBOX": "1",
        }
        return ACPConfig(
            env=env,
            command=[f"{NODE_BIN_DIR}/node", ACP_BIN.format(**versions)],
            prompt=prompt or "",
            session_meta=session_meta,
        )

    async def configure_tool_interception(
        self,
        config: ACPConfig,
        runtime: Runtime,
        url: str,
        secret: str,
    ) -> None:
        if self.config.version != claudeVersion:
            raise HarnessError(
                "Claude Code tool interception is verified only for version "
                f"{claudeVersion}"
            )
        versions = {"version": self.config.version, "acp_version": ACP_VERSION}
        config.command = [
            f"{NODE_BIN_DIR}/node",
            "--input-type=module",
            "--eval",
            wrapperSource,
            acpLib.format(**versions),
            acpIndex.format(**versions),
        ]
        assert config.session_meta is not None
        claudeCode = config.session_meta["claudeCode"]
        assert isinstance(claudeCode, dict)
        options = claudeCode["options"]
        assert isinstance(options, dict)
        options["settingSources"] = ["user"]
        await self.install_skills(runtime, f"{config.env['CLAUDE_CONFIG_DIR']}/skills")

    async def cleanup(self, trace: Trace, runtime: Runtime) -> None:
        result = await runtime.run(["rm", "-rf", self.config_dir(trace)], {})
        if result.exit_code != 0:
            raise RuntimeError(
                f"failed to clean up Claude config: {result.stderr.strip()[-500:]}"
            )

    @staticmethod
    def config_dir(trace: Trace) -> str:
        return f"{CLAUDE_CONFIG_ROOT}/{trace.id}"
