"""Run Hermes Agent against interception through its native ACP server."""

import json
from pathlib import Path

from verifiers.v1.acp import ACPConfig, ACPHarness
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig, PinnedVersion
from verifiers.v1.harnesses.utils.install import ensure_installed, remove_dir
from verifiers.v1.runtimes import Runtime
from verifiers.v1.runtimes.base import _ENSURE_UV
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

PROGRAM_SOURCE = (Path(__file__).resolve().parent / "program.py").read_text()
HERMES_DIR = "/var/tmp/vf-hermes-agent-{version}"
INSTALL = f"""
set -e
{_ENSURE_UV}
command -v curl >/dev/null || (apt-get update -qq && apt-get install -y -qq curl ca-certificates >/dev/null)
curl -fsSL "https://github.com/NousResearch/hermes-agent/archive/refs/tags/$VF_HERMES_VERSION.tar.gz" \\
    | tar -xz --strip-components=1 -C "$VF_HERMES_DIR"
uv sync --project "$VF_HERMES_DIR" --locked --no-dev --extra acp --extra mcp
touch "$VF_HERMES_DIR/.ready"
"""


class HermesAgentHarnessConfig(HarnessConfig):
    version: PinnedVersion = "v2026.9.14"
    """Hermes Agent Git release tag to install, pinned for reproducibility."""
    use_bundled_skill: bool = False
    """Enable Hermes Agent's bundled skill catalog in addition to uploaded skills."""


class HermesAgentHarness(ACPHarness[HermesAgentHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_SKILLS = True

    async def setup(self, runtime: Runtime) -> None:
        # Hermes needs its source-tree assets and supports editable installs only.
        directory = HERMES_DIR.format(version=self.config.version)
        await ensure_installed(
            runtime,
            directory=directory,
            ready=f"test -f {directory}/.ready",
            install=INSTALL,
            env={
                **self.config.resolved_env,
                "VF_HERMES_DIR": directory,
                "VF_HERMES_VERSION": self.config.version,
            },
            label="Hermes Agent",
            shell=("bash", "-o", "pipefail", "-c"),
        )
        await super().setup(runtime)

    async def prepare_acp(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_servers: dict[str, dict],
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
            command=[
                f"{HERMES_DIR.format(version=self.config.version)}/.venv/bin/python",
                "-P",  # Keep task files from shadowing installed Hermes modules.
                "-c",
                PROGRAM_SOURCE,
            ],
            prompt=prompt,
            system_prompt=system_prompt,
        )

    async def cleanup(self, trace: Trace, runtime: Runtime) -> None:
        await remove_dir(runtime, f"/tmp/vf-hermes/{trace.id}", "Hermes home")
