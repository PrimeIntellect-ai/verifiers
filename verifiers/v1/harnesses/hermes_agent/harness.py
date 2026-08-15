"""Run Hermes Agent against interception through its native ACP server."""

import json
from pathlib import Path

from pydantic import Field

from verifiers.v1.acp import ACPConfig, ACPHarness
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.errors import HarnessError
from verifiers.v1.interception.tool import (
    HERMES_TOOL_HOOK_SOURCE,
    install_tool_hook,
)
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

PROGRAM_SOURCE = (Path(__file__).resolve().parent / "program.py").read_text()


class HermesAgentHarnessConfig(HarnessConfig):
    version: str = Field(default="0.19.0", pattern=r"^[A-Za-z0-9._+-]+$")
    """Hermes Agent release to install, pinned for reproducibility."""
    use_bundled_skill: bool = False
    """Enable Hermes Agent's bundled skill catalog in addition to uploaded skills."""


class HermesAgentHarness(ACPHarness[HermesAgentHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_SKILLS = True
    SUPPORTS_TOOL_INTERCEPTION = True

    async def setup(self, runtime: Runtime) -> None:
        await runtime.prepare_uv_script(
            PROGRAM_SOURCE.replace("{version}", self.config.version),
            self.config.resolved_env,
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
        if self.config.disabled_tools:
            raise ValueError("Hermes Agent ACP does not support disabling tools")

        home = f"/tmp/vf-hermes/{trace.id}"
        # Keep interception routing separate from vendor names that Hermes may resolve
        # to built-in cloud providers instead of the configured endpoint.
        model = {
            "provider": "openai",
            "default": ctx.model,
            **(
                {"max_tokens": ctx.sampling.max_tokens}
                if ctx.sampling.max_tokens is not None
                else {}
            ),
        }
        provider = {
            "api": endpoint,
            "api_key": secret,
            "discover_models": False,
        }
        if ctx.client.type == "eval":
            provider["transport"] = "${HERMES_INTERCEPT_TRANSPORT}"
        config = {
            "model": model,
            # The ACP client already approves tool requests. Avoid routing Hermes'
            # redundant smart-approval model calls through interception as turns.
            "approvals": {"mode": "off"},
            # Session titles are UI metadata, not part of the agent conversation.
            "auxiliary": {"title_generation": {"enabled": False}},
            "providers": {"openai": provider},
        }
        await runtime.write(f"{home}/config.yaml", json.dumps(config).encode())
        if not self.config.use_bundled_skill:
            await runtime.write(f"{home}/.no-bundled-skills", b"")
        await self.install_skills(runtime, f"{home}/skills")

        env = {
            **self.config.resolved_env,
            "HERMES_HOME": home,
            "HERMES_INFERENCE_MODEL": ctx.model,
        }
        system_prompt, prompt = self.resolve_prompt(data)
        return ACPConfig(
            env=env,
            command=await runtime.prepare_uv_script(
                PROGRAM_SOURCE.replace("{version}", self.config.version), env
            ),
            prompt=prompt,
            system_prompt=system_prompt,
        )

    async def configure_tool_interception(
        self,
        config: ACPConfig,
        trace: Trace,
        runtime: Runtime,
        url: str,
        secret: str,
    ) -> None:
        if self.config.version != "0.19.0":
            raise HarnessError(
                "Hermes Agent tool interception is verified only for version 0.19.0"
            )
        home = config.env["HERMES_HOME"]
        config.env["HERMES_ENABLE_PROJECT_PLUGINS"] = "0"
        config.env["HERMES_SAFE_MODE"] = "0"
        plugin = f"{home}/plugins/verifiers-interception"
        config.env.update(
            await install_tool_hook(
                runtime,
                f"{plugin}/__init__.py",
                url,
                secret,
                HERMES_TOOL_HOOK_SOURCE,
            )
        )
        manifest = (
            "name: verifiers-interception\n"
            'version: "1"\n'
            "hooks:\n"
            "  - transform_tool_result\n"
        )
        await runtime.write(f"{plugin}/plugin.yaml", manifest.encode())
        config_path = f"{home}/config.yaml"
        settings = json.loads((await runtime.read(config_path)).decode())
        settings["plugins"] = {"enabled": ["verifiers-interception"]}
        await runtime.write(config_path, json.dumps(settings).encode())

    async def cleanup(self, trace: Trace, runtime: Runtime) -> None:
        result = await runtime.run(["rm", "-rf", f"/tmp/vf-hermes/{trace.id}"], {})
        if result.exit_code:
            raise RuntimeError(
                f"failed to clean up Hermes home: {result.stderr.strip()[-500:]}"
            )
