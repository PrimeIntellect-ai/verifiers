"""The Claude Code harness: installs the CLI into the runtime and runs it headlessly.

Claude Code speaks the Anthropic Messages API, so `ANTHROPIC_BASE_URL` points it at the
rollout interception endpoint. Task-owned MCP servers are passed as explicit HTTP configs.
"""

import json
import logging

from verifiers.v1.clients import RolloutContext
from verifiers.v1.errors import ProgramError
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

INSTALL = r"""
set -e
bin="/tmp/vf-claude-code/.local/bin/claude"
if [ -x "$bin" ] && [ "$("$bin" --version 2>/dev/null | cut -d ' ' -f1)" = "{version}" ]; then
    exit 0
fi
command -v curl >/dev/null || { apt-get update -qq && apt-get install -y -qq curl >/dev/null; }
curl -fsSL https://claude.ai/install.sh | HOME=/tmp/vf-claude-code bash -s {version}
"""


class ClaudeCodeHarnessConfig(HarnessConfig):
    """The Claude Code CLI harness."""

    id: str = "claude-code"
    version: str = "2.1.177"
    """Claude Code release to install, pinned for reproducibility."""


class ClaudeCodeHarness(Harness[ClaudeCodeHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_TASK_TOOLS = True

    async def launch(
        self,
        ctx: RolloutContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> ProgramResult:
        system_prompt, instruction = self.resolve_prompt(trace.task)
        env = {
            **self.config.env,
            # Harness endpoints end in /v1 for OpenAI clients; Anthropic appends /v1/messages.
            "ANTHROPIC_BASE_URL": endpoint.removesuffix("/v1"),
            "ANTHROPIC_API_KEY": secret,
            "ANTHROPIC_CUSTOM_MODEL_OPTION": ctx.model,
            "CLAUDE_CONFIG_DIR": ".vf-claude",
            "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "DISABLE_UPDATES": "1",
        }
        logger.info(
            "claude-code: ensuring Claude Code %s is installed", self.config.version
        )
        install = await runtime.run(
            ["sh", "-c", INSTALL.replace("{version}", self.config.version)], {}
        )
        if install.exit_code != 0:
            raise ProgramError(
                f"Claude Code install failed: {install.stderr.strip()[-500:]}"
            )

        binary = "/tmp/vf-claude-code/.local/bin/claude"
        argv = [
            binary,
            "--print",
            "--dangerously-skip-permissions",
            "--no-session-persistence",
            "--setting-sources",
            "project",
            "--model",
            ctx.model,
        ]
        if system_prompt:
            argv += ["--append-system-prompt", system_prompt]
        if mcp_urls:
            config = {
                "mcpServers": {
                    name: {"type": "http", "url": url} for name, url in mcp_urls.items()
                }
            }
            argv += ["--strict-mcp-config", "--mcp-config", json.dumps(config)]
        argv.append(instruction)
        result = await runtime.run(argv, env)
        if result.exit_code != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise ProgramError(
                f"Claude Code exited {result.exit_code}: {detail[-2000:]}"
            )
        return result


def load_harness(config: ClaudeCodeHarnessConfig) -> ClaudeCodeHarness:
    return ClaudeCodeHarness(config)


__all__ = ["ClaudeCodeHarness", "ClaudeCodeHarnessConfig", "load_harness"]
