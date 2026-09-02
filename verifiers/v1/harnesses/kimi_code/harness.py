"""Run Kimi Code's native ACP server against interception."""

import logging
import shlex
import tomllib
from typing import Literal

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
    version: str = Field(default="0.36.0", pattern=r"^[A-Za-z0-9._+-]+$")
    """Kimi Code release to install, pinned for reproducibility."""
    transport: Literal["chat_completions", "responses", "anthropic_messages"] = (
        "chat_completions"
    )
    """Model API transport."""


class KimiCodeHarness(ACPHarness[KimiCodeHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_SKILLS = True
    SUPPORTS_TOOL_INTERCEPTION = True

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
        kimi_home = f"{KIMI_HOME}/{trace.id}"
        provider_type = {
            "chat_completions": "openai",
            "responses": "openai_responses",
            "anthropic_messages": "anthropic",
        }[self.config.transport]
        base_url = (
            endpoint.removesuffix("/v1")
            if self.config.transport == "anthropic_messages"
            else endpoint
        )
        config = {
            "extra_skill_dirs": [SKILLS_DIR] if self.config.skills else [],
            **(
                {
                    "permission": {
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
                }
                if self.config.disabled_tools
                else {}
            ),
        }
        await runtime.write(f"{kimi_home}/config.toml", tomli_w.dumps(config).encode())

        system_prompt, prompt = self.resolve_prompt(data)
        return ACPConfig(
            env={
                **self.config.resolved_env,
                "KIMI_CODE_HOME": kimi_home,
                "KIMI_MODEL_NAME": ctx.model,
                "KIMI_MODEL_API_KEY": secret,
                "KIMI_MODEL_BASE_URL": base_url,
                "KIMI_MODEL_PROVIDER_TYPE": provider_type,
                "KIMI_DISABLE_TELEMETRY": "1",
                "KIMI_CODE_NO_AUTO_UPDATE": "1",
            },
            command=ACP_COMMAND,
            prompt=prompt,
            system_prompt=system_prompt,
        )

    async def gate_tools(
        self, config: ACPConfig, runtime: Runtime, url: str, secret: str
    ) -> None:
        # An `ask` rule for every tool routes each call through Kimi's approval bridge,
        # which its ACP server turns into a permission request for the runner.
        path = f"{config.env['KIMI_CODE_HOME']}/config.toml"
        settings = tomllib.loads((await runtime.read(path)).decode())
        settings.setdefault("permission", {}).setdefault("rules", []).append(
            {"decision": "ask", "scope": "user", "pattern": "*"}
        )
        await runtime.write(path, tomli_w.dumps(settings).encode())
