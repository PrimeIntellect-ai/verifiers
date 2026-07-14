"""Run Claude Code against interception as an Anthropic Messages client."""

import json
import shlex

from verifiers.v1.clients import ModelContext
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace

CLAUDE_HOME = "/tmp/vf-claude-code-{version}"
CLAUDE_BIN = f"{CLAUDE_HOME}/.local/bin/claude"
INSTALL = """
set -e
command -v curl >/dev/null || (apt-get update -qq && apt-get install -y -qq curl ca-certificates >/dev/null)
curl -fsSL https://claude.ai/install.sh | HOME={home} bash -s {version}
"""


class ClaudeCodeHarnessConfig(HarnessConfig):
    version: str = "2.1.207"
    """Claude Code release to install; pinned for reproducibility."""


class ClaudeCodeHarness(Harness[ClaudeCodeHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    # images would require streaming inputs
    SUPPORTS_MESSAGE_PROMPT = False

    async def setup(self, runtime: Runtime) -> None:
        home = CLAUDE_HOME.format(version=self.config.version)
        binary = CLAUDE_BIN.format(version=self.config.version)
        script = INSTALL.format(version=self.config.version, home=home)
        # Cache the pinned binary across local rollouts; Linux has flock, macOS has lockf.
        install = shlex.quote(f"[ -x {binary} ] || ({script})")
        guarded = (
            f"mkdir -p {home} && "
            f'"$(command -v flock || command -v lockf)" {home}/install.lock '
            f"bash -o pipefail -c {install}"
        )
        result = await runtime.run(["sh", "-c", guarded], {})
        if result.exit_code != 0:
            detail = (result.stderr or result.stdout).strip()[-500:]
            raise RuntimeError(f"Claude Code install failed: {detail}")

    async def launch(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> ProgramResult:
        system_prompt, instruction = self.resolve_prompt(trace.task.data)
        if ctx.client.base_url.startswith("https://api.pinference.ai/"):
            # remove the /v1 from pinference
            ctx.client.base_url = ctx.client.base_url.removesuffix("/v1")
        env = {
            **self.config.resolved_env,
            # Claude appends /v1/messages; give it the interception root, not the model endpoint.
            "ANTHROPIC_BASE_URL": endpoint.removesuffix("/v1"),
            "ANTHROPIC_API_KEY": secret,
            "CLAUDE_CONFIG_DIR": ".vf-claude",
            "DISABLE_AUTOUPDATER": "1",
            "IS_SANDBOX": "1",
        }
        argv = [
            CLAUDE_BIN.format(version=self.config.version),
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
