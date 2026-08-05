"""Run Claude Code through the Claude Agent SDK ACP adapter."""

import shlex

from verifiers.v1.acp import ACP
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness
from verifiers.v1.harnesses._node import NODE_BIN_DIR, ensure_node
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

CLAUDE_ACP_DIR = "/tmp/vf-claude-agent-acp"
PACKAGES_DIR = f"{CLAUDE_ACP_DIR}/packages"
ACP_VERSION = "0.63.0"
ACP_BIN = f"{PACKAGES_DIR}/node_modules/.bin/claude-agent-acp"
ACP_COMMAND = [f"{NODE_BIN_DIR}/node", ACP_BIN]
CLAUDE_CONFIG_ROOT = ".vf-claude"
SKILLS_DIR = ".claude/skills"
ACP_INSTALL = r"""
set -e
export PATH="/tmp/vf-node/bin:$PATH"
if [ "$(cat /tmp/vf-claude-agent-acp/.version 2>/dev/null)" = "$VF_CLAUDE_ACP_VERSION" ] \
    && [ -x /tmp/vf-claude-agent-acp/packages/node_modules/.bin/claude-agent-acp ]; then
    exit 0
fi
npm install --prefix /tmp/vf-claude-agent-acp/packages --ignore-scripts --no-audit --no-fund \
    --omit=dev \
    "@agentclientprotocol/claude-agent-acp@$VF_CLAUDE_ACP_VERSION" >/dev/null
printf %s "$VF_CLAUDE_ACP_VERSION" > /tmp/vf-claude-agent-acp/.version
"""

CLAUDE_ACP = ACP()


class ClaudeCodeHarnessConfig(HarnessConfig):
    pass


class ClaudeCodeHarness(Harness[ClaudeCodeHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_RESUME = True
    SUPPORTS_SKILLS = True

    async def setup(self, runtime: Runtime) -> None:
        await self.install_skills(runtime, SKILLS_DIR)
        await ensure_node(runtime)
        acp_guarded = (
            f"mkdir -p {CLAUDE_ACP_DIR} && "
            f'"$(command -v flock || command -v lockf)" {CLAUDE_ACP_DIR}/install.lock '
            f"sh -c {shlex.quote(ACP_INSTALL)}"
        )
        acp_result = await runtime.run(
            ["sh", "-c", acp_guarded], {"VF_CLAUDE_ACP_VERSION": ACP_VERSION}
        )
        if acp_result.exit_code != 0:
            detail = (acp_result.stderr or acp_result.stdout).strip()[-500:]
            raise RuntimeError(f"Claude Agent ACP install failed: {detail}")
        await CLAUDE_ACP.setup(self, runtime)

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
        config_dir = self._config_dir(trace)

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
            "CLAUDE_CONFIG_DIR": config_dir,
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "DISABLE_AUTOUPDATER": "1",
            "IS_SANDBOX": "1",
        }
        return await CLAUDE_ACP.run(
            runtime,
            env,
            ACP_COMMAND,
            prompt,
            mcp_urls=mcp_urls,
            session_path=f"{config_dir}/acp-session",
            session_meta=session_meta,
        )

    async def cleanup(self, trace: Trace, runtime: Runtime) -> None:
        result = await runtime.run(["rm", "-rf", self._config_dir(trace)], {})
        if result.exit_code != 0:
            raise RuntimeError(
                f"failed to clean up Claude config: {result.stderr.strip()[-500:]}"
            )

    @staticmethod
    def _config_dir(trace: Trace) -> str:
        return f"{CLAUDE_CONFIG_ROOT}/{trace.id}"
