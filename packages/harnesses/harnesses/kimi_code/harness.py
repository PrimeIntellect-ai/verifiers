"""The Kimi Code harness: installs the CLI into the runtime and runs it headlessly.

Kimi Code speaks the OpenAI Chat Completions API through an environment-defined model, so
the rollout interception endpoint and session secret are passed through `KIMI_MODEL_*`.
Task-owned MCP servers live in an isolated Kimi home.
"""

import json
import logging
import shlex

from verifiers.v1.clients import RolloutContext
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

BINARY = "/tmp/vf-kimi-code/bin/kimi"
KIMI_HOME = ".vf-kimi-code"

INSTALL = r"""
set -e
bin="/tmp/vf-kimi-code/bin/kimi"
if [ -x "$bin" ] && [ "$("$bin" --version 2>/dev/null)" = "{version}" ]; then
    exit 0
fi
command -v curl >/dev/null || { apt-get update -qq && apt-get install -y -qq curl ca-certificates >/dev/null; }
installer=/tmp/vf-kimi-code-install.sh
curl -fsSL https://code.kimi.com/kimi-code/install.sh -o "$installer"
env \
    KIMI_VERSION="{version}" \
    KIMI_INSTALL_DIR=/tmp/vf-kimi-code \
    KIMI_NO_MODIFY_PATH=1 \
    bash "$installer"
"""


class KimiCodeHarnessConfig(HarnessConfig):
    """The Kimi Code CLI harness."""

    id: str = "kimi-code"
    version: str = "0.14.3"
    """Kimi Code release to install, pinned for reproducibility."""


class KimiCodeHarness(Harness[KimiCodeHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = False
    SUPPORTS_TASK_TOOLS = True

    async def setup(self, runtime: Runtime) -> None:
        logger.info(
            "kimi-code: ensuring Kimi Code %s is installed", self.config.version
        )
        script = INSTALL.replace("{version}", self.config.version)
        guarded = (
            "mkdir -p /tmp/vf-kimi-code && "
            f"flock /tmp/vf-kimi-code/install.lock sh -c {shlex.quote(script)}"
        )
        install = await runtime.run(["sh", "-c", guarded], {})
        if install.exit_code != 0:
            raise RuntimeError(
                f"Kimi Code install failed: {install.stderr.strip()[-500:]}"
            )

    async def launch(
        self,
        ctx: RolloutContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> ProgramResult:
        _, prompt = self.resolve_prompt(trace.task)
        env = {
            **self.config.env,
            "KIMI_CODE_HOME": KIMI_HOME,
            "KIMI_MODEL_NAME": ctx.model,
            "KIMI_MODEL_API_KEY": secret,
            "KIMI_MODEL_PROVIDER_TYPE": "openai",
            "KIMI_MODEL_BASE_URL": endpoint,
            "KIMI_MODEL_CAPABILITIES": "tool_use",
            "KIMI_DISABLE_TELEMETRY": "1",
            "KIMI_CODE_NO_AUTO_UPDATE": "1",
        }

        mcp = {"mcpServers": {name: {"url": url} for name, url in mcp_urls.items()}}
        # Values are Kimi permission patterns such as `Bash` or `Bash(rm -rf*)`.
        # https://moonshotai.github.io/kimi-code/en/configuration/config-files#permission
        permission_rules = "\n".join(
            "\n".join(
                (
                    "[[permission.rules]]",
                    'decision = "deny"',
                    'scope = "user"',
                    f"pattern = {json.dumps(tool)}",
                    'reason = "Disabled by Verifiers harness configuration."',
                )
            )
            for tool in self.config.disabled_tools or []
        )
        if permission_rules:
            await runtime.write(f"{KIMI_HOME}/config.toml", permission_rules.encode())
        await runtime.write(f"{KIMI_HOME}/mcp.json", json.dumps(mcp).encode())
        # `--prompt` is Kimi Code's non-interactive print mode.
        return await runtime.run_program([BINARY, "--prompt", prompt], env)
