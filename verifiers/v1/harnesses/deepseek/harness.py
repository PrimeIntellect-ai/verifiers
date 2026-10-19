"""DeepSeek's minimal coding agent, driven through its native ACP bridge."""

import json
from pathlib import Path
from typing import Literal

from verifiers.v1.acp import ACPConfig, ACPHarness
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig, PinnedVersion
from verifiers.v1.harnesses.node import NODE_BIN_DIR, ensure_node
from verifiers.v1.harnesses.utils.install import ensure_installed, remove_dir
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

DEEPSEEK_DIR = "/var/tmp/vf-deepseek-{version}"
USAGE_COMPAT = (Path(__file__).parent / "usage.mjs").read_bytes()
GATE = (Path(__file__).parent / "gate.mjs").read_bytes()
INSTALL = r"""
set -e
export PATH="/var/tmp/vf-node/bin:$PATH"
npm install --prefix "$VF_DEEPSEEK_DIR" --no-audit --no-fund \
    "@deepseek-ai/dsh@$VF_DEEPSEEK_VERSION" >/dev/null
touch "$VF_DEEPSEEK_DIR/.ready"
"""


class DeepSeekHarnessConfig(HarnessConfig):
    # The `deepseek-official` route speaks Anthropic messages:
    # https://github.com/deepseek-ai/deepseek-harness/blob/dsh-v0.2.1-alpha.3/packages/llm/llm-deepseek/src/messages-api.ts#L14-L17
    dialect: Literal["anthropic_messages"] = "anthropic_messages"
    version: PinnedVersion = "0.2.1-alpha.3"
    """DeepSeek CLI release, pinned to its shipped sdk-minimal composition."""


class DeepSeekHarness(ACPHarness[DeepSeekHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_TOOL_INTERCEPTION = True

    async def setup(self, runtime: Runtime) -> None:
        await ensure_node(runtime)
        directory = DEEPSEEK_DIR.format(version=self.config.version)
        await ensure_installed(
            runtime,
            directory=directory,
            ready=f"test -f {directory}/.ready",
            install=INSTALL,
            env={
                "VF_DEEPSEEK_DIR": directory,
                "VF_DEEPSEEK_VERSION": self.config.version,
            },
            label="DeepSeek Harness",
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
            raise ValueError("DeepSeek minimal ACP does not support disabling tools")

        home = f"/tmp/vf-deepseek/{trace.id}"
        system_prompt, prompt = self.resolve_prompt(data)
        # Keep the upstream minimal tree; replace only its SDK stdio front door.
        # https://github.com/deepseek-ai/deepseek-harness/tree/master/packages/bundle/sdk-minimal
        patches = [
            {"id": "sdk-app-startup", "disabled": True},
            {"id": "sdk-jsonrpc-server", "disabled": True},
            # Provider metadata is captured by verifiers, not uploaded by DSH.
            {"id": "session-log-deepseek", "disabled": True},
            {"id": "plugin-package-inventory-deepseek", "disabled": True},
            {
                "insert": [
                    {
                        "id": "acp-app-startup",
                        "name": "@deepseek-ai/dsh-acp-app",
                    },
                    {
                        "id": "acp",
                        "name": "@deepseek-ai/dsh-acp",
                        "inject": ["acpAppStartup"],
                        "config": {"provider": "deepseek-official", "model": ctx.model},
                    },
                ]
            },
        ]
        if system_prompt:
            patches.append(
                {
                    "id": "system-prompt",
                    "config": {
                        "includeHarnessIdentity": False,
                        "includeRuntimeContext": False,
                        "personaPrefix": "You are a helpful software engineer assistant."
                        + "\n\n"
                        + system_prompt,
                    },
                }
            )
        patch_path = f"{home}/acp.patch.json"
        # JSON is valid YAML and preserves arbitrary task prompt text verbatim.
        await runtime.write(patch_path, json.dumps(patches).encode())
        usage_path = f"{home}/usage.mjs"
        await runtime.write(usage_path, USAGE_COMPAT)
        return ACPConfig(
            command=[
                f"{NODE_BIN_DIR}/node",
                "--import",
                usage_path,
                f"{DEEPSEEK_DIR.format(version=self.config.version)}/node_modules/@deepseek-ai/dsh/lib/bin.js",
                "--profile",
                "sdk-minimal",
                "--patch",
                patch_path,
            ],
            env={
                **self.config.resolved_env,
                "DSH_HOME": home,
                "DEEPSEEK_BASE_URL": endpoint,
                "DEEPSEEK_API_KEY": secret,
            },
            prompt=prompt,
        )

    async def gate_tools(
        self, config: ACPConfig, runtime: Runtime, url: str, secret: str
    ) -> None:
        home = config.env["DSH_HOME"]
        gate_path = f"{home}/gate.mjs"
        await runtime.write(gate_path, GATE)
        patch_path = f"{home}/acp.patch.json"
        patches = json.loads(await runtime.read(patch_path))
        patches.append(
            {
                "insert": [
                    {"id": "vf-approval", "name": "@deepseek-ai/dsh-user-approval"},
                    {"id": "vf-tool-gate", "name": gate_path},
                ]
            }
        )
        await runtime.write(patch_path, json.dumps(patches).encode())

    async def cleanup(self, trace: Trace, runtime: Runtime) -> None:
        await remove_dir(runtime, f"/tmp/vf-deepseek/{trace.id}", "DeepSeek state")
