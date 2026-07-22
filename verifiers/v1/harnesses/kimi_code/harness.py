"""Kimi receives interception through `KIMI_MODEL_*` and runs through native ACP."""

import json
import logging
import shlex

from verifiers.v1.acp import ACP
from verifiers.v1.clients import ModelContext
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

BINARY = "/tmp/vf-kimi-code/bin/kimi"
KIMI_HOME = ".vf-kimi-code"
ACP_COMMAND = [
    "sh",
    "-c",
    f'KIMI_CODE_HOME="$PWD/$KIMI_CODE_HOME" exec {BINARY} acp',
]

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

KIMI_ACP = ACP()


class KimiCodeHarnessConfig(HarnessConfig):
    version: str = "0.29.0"
    """Kimi Code release to install, pinned for reproducibility."""


class KimiCodeHarness(Harness[KimiCodeHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_RESUME = True

    async def setup(self, runtime: Runtime) -> None:
        logger.info(
            "kimi-code: ensuring Kimi Code %s is installed", self.config.version
        )
        script = INSTALL.replace("{version}", self.config.version)
        guarded = (
            "mkdir -p /tmp/vf-kimi-code && "
            '"$(command -v flock || command -v lockf)" '
            f"/tmp/vf-kimi-code/install.lock sh -c {shlex.quote(script)}"
        )
        install = await runtime.run(["sh", "-c", guarded], {})
        if install.exit_code != 0:
            raise RuntimeError(
                f"Kimi Code install failed: {install.stderr.strip()[-500:]}"
            )
        await KIMI_ACP.setup(self, runtime)

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
        kimi_home = f"{KIMI_HOME}/{trace.id}"
        env = {
            **self.config.resolved_env,
            "KIMI_CODE_HOME": kimi_home,
            "KIMI_MODEL_NAME": ctx.model,
            "KIMI_MODEL_API_KEY": secret,
            "KIMI_MODEL_PROVIDER_TYPE": "openai",
            "KIMI_MODEL_BASE_URL": endpoint,
            "KIMI_MODEL_CAPABILITIES": "tool_use",
            "KIMI_DISABLE_TELEMETRY": "1",
            "KIMI_CODE_NO_AUTO_UPDATE": "1",
        }
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
            await runtime.write(f"{kimi_home}/config.toml", permission_rules.encode())
        return await KIMI_ACP.run(
            runtime,
            env,
            ACP_COMMAND,
            prompt,
            mcp_urls=mcp_urls,
            system_prompt=system_prompt,
            session_path=f"{kimi_home}/acp-session",
        )
