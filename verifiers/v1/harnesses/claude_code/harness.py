"""Run Claude Code through the Claude Agent SDK ACP adapter."""

import shlex

from pydantic import Field

from verifiers.v1.acp import ACP
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness, HarnessSession
from verifiers.v1.harnesses.node import NODE_BIN_DIR, ensure_node
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

CLAUDE_ACP_DIR = "/var/tmp/vf-claude-agent-acp-{version}-{acp_version}"
PACKAGES_DIR = f"{CLAUDE_ACP_DIR}/packages"
ACP_VERSION = "0.65.0"
CLAUDE_BIN = f"{PACKAGES_DIR}/node_modules/.bin/claude"
ACP_BIN = f"{PACKAGES_DIR}/node_modules/.bin/claude-agent-acp"
CLAUDE_CONFIG_ROOT = ".vf-claude"
SKILLS_DIR = ".claude/skills"
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

CLAUDE_ACP = ACP()


class ClaudeCodeHarnessConfig(HarnessConfig):
    version: str = Field(default="2.1.223", pattern=r"^[A-Za-z0-9._+-]+$")
    """Claude Code release to install, pinned for reproducibility."""


class ClaudeCodeHarness(Harness[ClaudeCodeHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_RESUME = True
    SUPPORTS_SKILLS = True

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
        await CLAUDE_ACP.setup(self, runtime)

    async def session(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
    ) -> HarnessSession:
        if not runtime.supports_live_processes:
            return await super().session(
                ctx, trace, runtime, endpoint, secret, mcp_urls, data
            )
        system_prompt, prompt = self.resolve_prompt(data)
        config_dir = self.config_dir(trace)
        versions = {"version": self.config.version, "acp_version": ACP_VERSION}
        options: dict[str, object] = {
            "strictMcpConfig": True,
            "disallowedTools": self.config.disabled_tools or [],
        }
        session_meta: dict[str, object] = {"claudeCode": {"options": options}}
        if system_prompt:
            session_meta["systemPrompt"] = {"append": system_prompt}
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
        return CLAUDE_ACP.session(
            self,
            ctx,
            trace,
            runtime,
            endpoint,
            secret,
            mcp_urls,
            data,
            env=env,
            command=[f"{NODE_BIN_DIR}/node", ACP_BIN.format(**versions)],
            prompt=prompt or "",
            session_meta=session_meta,
        )

    async def launch(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
    ) -> ProgramResult:
        system_prompt, prompt = self.resolve_prompt(data)
        config_dir = self.config_dir(trace)
        versions = {"version": self.config.version, "acp_version": ACP_VERSION}

        options: dict[str, object] = {
            "strictMcpConfig": True,
            "disallowedTools": self.config.disabled_tools or [],
        }
        session_meta: dict[str, object] = {"claudeCode": {"options": options}}
        if system_prompt:
            session_meta["systemPrompt"] = {"append": system_prompt}
        env = {
            **self.config.resolved_env,
            # Claude appends /v1/messages; give it the interception root, not the model endpoint.
            "ANTHROPIC_BASE_URL": endpoint.removesuffix("/v1"),
            "ANTHROPIC_API_KEY": secret,
            "ANTHROPIC_MODEL": ctx.model,
            "CLAUDE_CODE_EXECUTABLE": CLAUDE_BIN.format(**versions),
            "CLAUDE_CONFIG_DIR": config_dir,
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "DISABLE_AUTOUPDATER": "1",
            "IS_SANDBOX": "1",
        }
        return await CLAUDE_ACP.run(
            runtime,
            env,
            [f"{NODE_BIN_DIR}/node", ACP_BIN.format(**versions)],
            prompt or "",
            mcp_urls=mcp_urls,
            session_path=f"{config_dir}/acp-session",
            session_meta=session_meta,
            trace=trace,
        )

    async def cleanup(self, trace: Trace, runtime: Runtime) -> None:
        result = await runtime.run(["rm", "-rf", self.config_dir(trace)], {})
        if result.exit_code != 0:
            raise RuntimeError(
                f"failed to clean up Claude config: {result.stderr.strip()[-500:]}"
            )

    @staticmethod
    def config_dir(trace: Trace) -> str:
        return f"{CLAUDE_CONFIG_ROOT}/{trace.id}"
