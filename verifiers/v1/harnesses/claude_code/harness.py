"""Install Claude Code and run it headlessly."""

import json
import shlex

from verifiers.v1.clients import RolloutContext
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace

CLAUDE_VERSION = "2.1.207"
CLAUDE_HOME = f"/tmp/vf-claude-code-{CLAUDE_VERSION}"
CLAUDE_BIN = f"{CLAUDE_HOME}/.local/bin/claude"
INSTALL = f"""
set -e
command -v curl >/dev/null || (apt-get update -qq && apt-get install -y -qq curl ca-certificates >/dev/null)
curl -fsSL https://claude.ai/install.sh | HOME={CLAUDE_HOME} bash -s {CLAUDE_VERSION}
"""


class ClaudeCodeHarness(Harness[HarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    # images would require streaming inputs
    SUPPORTS_MESSAGE_PROMPT = False

    async def setup(self, runtime: Runtime) -> None:
        # Cache the pinned binary across local rollouts; Linux has flock, macOS has lockf.
        install = shlex.quote(f"[ -x {CLAUDE_BIN} ] || ({INSTALL})")
        guarded = (
            f"mkdir -p {CLAUDE_HOME} && "
            f'"$(command -v flock || command -v lockf)" {CLAUDE_HOME}/install.lock '
            f"bash -o pipefail -c {install}"
        )
        result = await runtime.run(["sh", "-c", guarded], {})
        if result.exit_code != 0:
            detail = (result.stderr or result.stdout).strip()[-500:]
            raise RuntimeError(f"Claude Code install failed: {detail}")

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
            **self.config.resolved_env,
            "ANTHROPIC_BASE_URL": endpoint.removesuffix("/v1"),
            "ANTHROPIC_API_KEY": secret,
            "CLAUDE_CONFIG_DIR": ".vf-claude",
            "DISABLE_AUTOUPDATER": "1",
            "IS_SANDBOX": "1",
        }
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
        if self.config.disabled_tools:
            argv += ["--disallowedTools", ",".join(self.config.disabled_tools)]
        mcp = {
            "mcpServers": {
                name: {"type": "http", "url": url} for name, url in mcp_urls.items()
            }
        }
        argv += ["--mcp-config", json.dumps(mcp), "--strict-mcp-config", instruction]
        return await runtime.run_program(argv, env)
