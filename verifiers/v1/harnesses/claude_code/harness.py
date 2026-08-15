"""Run Claude Code through the Claude Agent SDK ACP adapter."""

import shlex
from pathlib import Path

from pydantic import Field

from verifiers.v1.acp import ACPConfig, ACPHarness
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.errors import HarnessError
from verifiers.v1.harnesses.node import NODE_BIN_DIR, ensure_node
from verifiers.v1.interception.tool import TOOL_HOOK_SCRIPT
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

CLAUDE_ACP_DIR = "/var/tmp/vf-claude-agent-acp-{version}-{acp_version}"
PACKAGES_DIR = f"{CLAUDE_ACP_DIR}/packages"
ACP_VERSION = "0.67.0"
CLAUDE_CODE_VERSION = "2.1.232"
CLAUDE_BIN = f"{PACKAGES_DIR}/node_modules/.bin/claude"
ACP_BIN = f"{PACKAGES_DIR}/node_modules/.bin/claude-agent-acp"
ACP_LIB = (
    f"{PACKAGES_DIR}/node_modules/@agentclientprotocol/claude-agent-acp/dist/lib.js"
)
ACP_INDEX = (
    f"{PACKAGES_DIR}/node_modules/@agentclientprotocol/claude-agent-acp/dist/index.js"
)
CLAUDE_CONFIG_ROOT = ".vf-claude"
SKILLS_DIR = ".claude/skills"
CLAUDE_ACP_WRAPPER_SCRIPT = (
    TOOL_HOOK_SCRIPT + "\n" + Path(__file__).with_name("wrapper.mjs").read_text()
)
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
    version: str = Field(default=CLAUDE_CODE_VERSION, pattern=r"^[A-Za-z0-9._+-]+$")
    """Claude Code release to install, pinned for reproducibility."""


class ClaudeCodeHarness(ACPHarness[ClaudeCodeHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_SKILLS = True
    SUPPORTS_TOOL_INTERCEPTION = True

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
        trace: Trace,
        runtime: Runtime,
        url: str,
        secret: str,
    ) -> None:
        if self.config.version != CLAUDE_CODE_VERSION:
            raise HarnessError(
                "Claude Code tool interception is verified only for version "
                f"{CLAUDE_CODE_VERSION}"
            )
        config_dir = self.config_dir(trace)
        await self.install_skills(runtime, f"{config_dir}/skills")
        assert config.session_meta is not None
        versions = {"version": self.config.version, "acp_version": ACP_VERSION}
        config.command = [
            f"{NODE_BIN_DIR}/node",
            "--input-type=module",
            "--eval",
            CLAUDE_ACP_WRAPPER_SCRIPT,
            ACP_LIB.format(**versions),
            ACP_INDEX.format(**versions),
        ]
        # The wrapper consumes this private ACP metadata before the adapter constructs
        # the Claude query, so neither value reaches the model-controlled subprocess.
        config.session_meta["vfToolInterception"] = {"url": url, "secret": secret}
        claude = config.session_meta["claudeCode"]
        assert isinstance(claude, dict)
        options = claude["options"]
        assert isinstance(options, dict)
        # claude-agent-acp otherwise includes project/local sources, which lets the task
        # merge its own native hooks into the Claude process.
        options["settingSources"] = ["user"]

    async def cleanup(self, trace: Trace, runtime: Runtime) -> None:
        result = await runtime.run(["rm", "-rf", self.config_dir(trace)], {})
        if result.exit_code != 0:
            raise RuntimeError(
                f"failed to clean up Claude config: {result.stderr.strip()[-500:]}"
            )

    @staticmethod
    def config_dir(trace: Trace) -> str:
        return f"{CLAUDE_CONFIG_ROOT}/{trace.id}"
