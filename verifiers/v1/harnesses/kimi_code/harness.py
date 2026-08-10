"""Kimi receives interception through an isolated provider and runs through native ACP."""

import logging
import shlex

import tomli_w
from pydantic import Field

from verifiers.v1.acp import ACPConfig, ACPHarness
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

BINARY = "/tmp/vf-kimi-code/bin/kimi"
KIMI_HOME = ".vf-kimi-code"
ACP_COMMAND = [BINARY, "acp"]
SKILLS_DIR = f"{KIMI_HOME}/skills"

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
    version: str = Field(default="0.34.0", pattern=r"^[A-Za-z0-9._+-]+$")
    """Kimi Code release to install, pinned for reproducibility."""


class KimiCodeHarness(ACPHarness[KimiCodeHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_RESUME = True
    SUPPORTS_SKILLS = True

    async def setup(self, runtime: Runtime) -> None:
        await self.install_skills(runtime, SKILLS_DIR)
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
        await self.acp.setup(self, runtime)

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
        kimi_home = f"{KIMI_HOME}/{trace.id}"
        # Anthropic preserves signed reasoning and tool identity across history replay.
        config: dict[str, object] = {
            "default_model": "intercept",
            "extra_skill_dirs": [SKILLS_DIR] if self.config.skills else [],
            "providers": {
                "intercept": {
                    "type": "anthropic",
                    "api_key": secret,
                    "base_url": endpoint.removesuffix("/v1"),
                }
            },
            "models": {
                "intercept": {
                    "provider": "intercept",
                    "model": ctx.model,
                    "max_context_size": 262144,
                    "capabilities": ["tool_use"],
                }
            },
        }
        if self.config.disabled_tools:
            config["permission"] = {
                "rules": [
                    {
                        "decision": "deny",
                        "scope": "user",
                        "pattern": tool,
                        "reason": "Disabled by Verifiers harness configuration.",
                    }
                    for tool in self.config.disabled_tools
                ]
            }
        await runtime.write(f"{kimi_home}/config.toml", tomli_w.dumps(config).encode())

        system_prompt, prompt = self.resolve_prompt(data)
        return ACPConfig(
            env={
                **self.config.resolved_env,
                "KIMI_CODE_HOME": kimi_home,
                # Kimi's env-model overlay cannot select the configured provider.
                "KIMI_MODEL_NAME": "",
                "KIMI_DISABLE_TELEMETRY": "1",
                "KIMI_CODE_NO_AUTO_UPDATE": "1",
            },
            command=ACP_COMMAND,
            prompt=prompt,
            system_prompt=system_prompt,
            session_path=f"{KIMI_HOME}/{trace.id}/acp-session",
        )
