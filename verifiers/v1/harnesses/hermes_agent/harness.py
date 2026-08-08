"""Run Hermes Agent against interception through its native ACP server."""

import json
import logging
from pathlib import Path

from pydantic import Field

from verifiers.v1.acp import ACP
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

PROGRAM_SOURCE = (Path(__file__).resolve().parent / "program.py").read_text()
SKILLS_DIR = ".vf-hermes-skills"
HERMES_ACP = ACP()
PROVIDER = "intercept"
logger = logging.getLogger(__name__)


class HermesAgentHarnessConfig(HarnessConfig):
    version: str = Field(default="0.19.0", pattern=r"^[A-Za-z0-9._+-]+$")
    """Hermes Agent release to install, pinned for reproducibility."""
    use_bundled_skill: bool = False
    """Enable Hermes Agent's bundled skill catalog in addition to uploaded skills."""


class HermesAgentHarness(Harness[HermesAgentHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_RESUME = True
    SUPPORTS_SKILLS = True

    async def setup(self, runtime: Runtime) -> None:
        await self.install_skills(runtime, SKILLS_DIR)
        logger.info(
            "hermes-agent: ensuring Hermes Agent %s is installed", self.config.version
        )
        source = PROGRAM_SOURCE.replace("{version}", self.config.version)
        await runtime.prepare_uv_script(source, self.config.resolved_env)
        await HERMES_ACP.setup(self, runtime)

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
        if self.config.disabled_tools:
            raise ValueError("Hermes Agent ACP does not support disabling tools")

        system_prompt, prompt = self.resolve_prompt(data)
        home = self._home(trace)
        model: dict[str, object] = {
            "provider": PROVIDER,
            "default": ctx.model,
            "base_url": endpoint,
            "api_mode": "chat_completions",
        }
        if ctx.sampling.max_tokens is not None:
            model["max_tokens"] = ctx.sampling.max_tokens
        await runtime.write(
            f"{home}/config.yaml",
            json.dumps(
                {
                    "model": model,
                    # The ACP client already approves tool requests. Avoid routing Hermes'
                    # redundant smart-approval model calls through interception as turns.
                    "approvals": {"mode": "off"},
                    "providers": {
                        PROVIDER: {
                            "api": endpoint,
                            "api_key": secret,
                            "transport": "chat_completions",
                            "discover_models": False,
                            "models": [ctx.model],
                        }
                    },
                }
            ).encode(),
        )
        if not self.config.use_bundled_skill:
            await runtime.write(
                f"{home}/.no-bundled-skills",
                b"Verifiers supplies evaluation skills explicitly.\n",
            )

        if self.config.skills:
            skill_home = f"{home}/skills"
            for command in (
                ["rm", "-rf", skill_home],
                ["cp", "-R", SKILLS_DIR, skill_home],
            ):
                copied = await runtime.run(command, {})
                if copied.exit_code != 0:
                    raise RuntimeError(
                        f"failed to stage Hermes skills: {copied.stderr.strip()[-500:]}"
                    )

        env = {
            **self.config.resolved_env,
            "HERMES_HOME": home,
            "HERMES_ACP_SKIP_CONFIGURED_MCP": "1",
        }
        source = PROGRAM_SOURCE.replace("{version}", self.config.version)
        command = await runtime.prepare_uv_script(source, env)
        calls_before = len(trace.calls)
        result = await HERMES_ACP.run(
            runtime,
            env,
            command,
            prompt,
            mcp_urls=mcp_urls,
            system_prompt=system_prompt,
        )
        if not any(call.node is not None for call in trace.calls[calls_before:]):
            detail = (result.stderr or result.stdout).strip()[-500:] or "<no output>"
            raise RuntimeError(
                "Hermes Agent completed without committing a model turn: " + detail
            )
        return result

    async def cleanup(self, trace: Trace, runtime: Runtime) -> None:
        result = await runtime.run(["rm", "-rf", self._home(trace)], {})
        if result.exit_code != 0:
            raise RuntimeError(
                f"failed to clean up Hermes home: {result.stderr.strip()[-500:]}"
            )

    @staticmethod
    def _home(trace: Trace) -> str:
        return f"/tmp/vf-hermes/{trace.id}"
