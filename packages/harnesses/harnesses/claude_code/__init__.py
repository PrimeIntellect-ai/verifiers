"""Claude Code harness: install the CLI and run it against the interception endpoint."""

import json
import logging
import shlex

from verifiers.v1.clients import RolloutContext
from verifiers.v1.errors import ProgramError
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

CLAUDE_DIR = "/tmp/vf-claude-code"
CLAUDE_BIN = f"{CLAUDE_DIR}/.local/bin/claude"
INSTALL = r"""
set -e
command -v curl >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq curl >/dev/null; }
curl -fsSL https://claude.ai/install.sh | HOME={dir} bash -s {version}
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
            "ANTHROPIC_BASE_URL": endpoint.removesuffix("/v1"),
            "ANTHROPIC_API_KEY": secret,
            "ANTHROPIC_CUSTOM_MODEL_OPTION": ctx.model,
            "API_FORCE_IDLE_TIMEOUT": "0",
            "CLAUDE_CONFIG_DIR": ".vf-claude",
            "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "DISABLE_UPDATES": "1",
            "IS_SANDBOX": "1",
        }
        logger.info(
            "claude-code: ensuring Claude Code %s is installed", self.config.version
        )
        script = INSTALL.replace("{version}", self.config.version).replace(
            "{dir}", CLAUDE_DIR
        )
        version = f'$({CLAUDE_BIN} --version 2>/dev/null | cut -d " " -f1)'
        ensure = shlex.quote(
            f'[ -x {CLAUDE_BIN} ] && [ "{version}" = "{self.config.version}" ]'
            f" || ({script})"
        )
        guarded = (
            f"mkdir -p {CLAUDE_DIR} && flock {CLAUDE_DIR}/install.lock sh -c {ensure}"
        )
        install = await runtime.run(["sh", "-c", guarded], {})
        if install.exit_code != 0:
            raise ProgramError(
                f"Claude Code install failed: {install.stderr.strip()[-500:]}"
            )

        argv = [
            CLAUDE_BIN,
            "--print",
            "--bare",
            "--dangerously-skip-permissions",
            "--no-session-persistence",
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
            argv += ["--mcp-config", json.dumps(config)]
        argv += ["--strict-mcp-config", instruction]
        return await runtime.run(argv, env)


def load_harness(config: ClaudeCodeHarnessConfig) -> ClaudeCodeHarness:
    return ClaudeCodeHarness(config)


__all__ = ["ClaudeCodeHarness", "ClaudeCodeHarnessConfig", "load_harness"]
