"""Run Kimi Code's native ACP server against interception."""

import logging
import tomllib

import tomli_w

from verifiers.v1.acp import ACPConfig, ACPHarness
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import DialectName, HarnessConfig, PinnedVersion
from verifiers.v1.harnesses.node import NODE_BIN_DIR, ensure_node
from verifiers.v1.harnesses.utils.install import ensure_installed, remove_dir
from verifiers.v1.harnesses.utils.replay import drop_blank_text
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace
from verifiers.v1.types import Message

logger = logging.getLogger(__name__)

KIMI_DIR = "/var/tmp/vf-kimi-code"
KIMI_BIN = f"{KIMI_DIR}/node_modules/.bin/kimi"
KIMI_HOME = ".vf-kimi-code"
ACP_COMMAND = [f"{NODE_BIN_DIR}/node", KIMI_BIN, "acp"]

# From npm rather than Kimi's installer: code.kimi.com sits behind a WAF that drops
# traffic from some sandbox providers' datacenter ranges (prime's among them).
INSTALL = r"""
set -e
packages=/var/tmp/vf-kimi-code
export PATH="/var/tmp/vf-node/bin:$PATH"
if [ "$(cat "$packages/.version" 2>/dev/null)" != "$VF_KIMI_VERSION" ]; then
    npm install --prefix "$packages" --ignore-scripts --no-audit --no-fund --omit=dev \
        "@moonshot-ai/kimi-code@$VF_KIMI_VERSION" >/dev/null
    printf %s "$VF_KIMI_VERSION" > "$packages/.version"
fi
"""


class KimiCodeHarnessConfig(HarnessConfig):
    version: PinnedVersion = "2.0.2"
    """Kimi Code release to install, pinned for reproducibility."""
    dialect: DialectName = "chat_completions"


class KimiCodeHarness(ACPHarness[KimiCodeHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_SKILLS = True
    SUPPORTS_TOOL_INTERCEPTION = True

    # Kimi Code drops whitespace-only assistant text when it stores a step, so every
    # transport replays it without. Recheck on version bumps:
    # https://github.com/MoonshotAI/kimi-code/blob/@moonshot-ai/kimi-code@2.0.2/packages/agent-core-v2/src/agent/loop/loopService.ts#L1723
    def replay_view(self, message: Message, dialect: DialectName | None) -> Message:
        return drop_blank_text(message)

    async def setup(self, runtime: Runtime) -> None:
        await ensure_node(runtime)
        logger.info(
            "kimi-code: ensuring Kimi Code %s is installed", self.config.version
        )
        await ensure_installed(
            runtime,
            directory=KIMI_DIR,
            install=INSTALL,
            env={"VF_KIMI_VERSION": self.config.version},
            label="Kimi Code",
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
        skills_dir = f"{kimi_home}/skills"
        await self.install_skills(runtime, skills_dir)
        provider_type = {
            "chat_completions": "openai",
            "responses": "openai_responses",
            "anthropic_messages": "anthropic",
        }[self.config.dialect]
        base_url = (
            endpoint.removesuffix("/v1")
            if self.config.dialect == "anthropic_messages"
            else endpoint
        )
        config = {
            "extra_skill_dirs": [skills_dir] if self.config.skills else [],
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

    async def cleanup(self, trace: Trace, runtime: Runtime) -> None:
        await remove_dir(runtime, f"{KIMI_HOME}/{trace.id}", "Kimi Code state")

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
