"""Run DeepSeek Harness against interception through its ACP server."""

import hashlib
import json
import logging
import re
import shlex
from pathlib import Path
from typing import Literal

from pydantic import Field

from verifiers.v1.acp import ACPConfig, ACPHarness
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harnesses.node import NODE_BIN_DIR, ensure_node
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

KEY_VAR = "DSH_INTERCEPT_KEY"
DSH_DIR = "/var/tmp/vf-deepseek-harness"
PACKAGES_DIR = f"{DSH_DIR}/packages"
DSH_BIN = f"{PACKAGES_DIR}/node_modules/.bin/dsh-acp-demo"
NODE_ADDON_VERSION = "0.1.4"
DEFAULT_CONTEXT_WINDOW = 262_144
ADAPTER_SOURCE = (Path(__file__).resolve().parent / "adapter.mjs").read_bytes()

INSTALL = r"""
set -e
packages=/var/tmp/vf-deepseek-harness/packages
export PATH="/var/tmp/vf-node/bin:$PATH"

if ! command -v python3 >/dev/null || ! command -v make >/dev/null || ! command -v c++ >/dev/null; then
    if command -v apt-get >/dev/null; then
        apt-get update -qq
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends python3 make g++ >/dev/null
        rm -rf /var/lib/apt/lists/*
    elif command -v apk >/dev/null; then
        apk add --no-cache python3 make g++ >/dev/null
    else
        echo "DeepSeek Harness requires Python, make, and a C++ compiler to build node-pty" >&2
        exit 1
    fi
fi

versions="$VF_DEEPSEEK_HARNESS_VERSION:$VF_NODE_ADDON_VERSION"
if [ "$(cat "$packages/.versions" 2>/dev/null)" != "$versions" ]; then
    npm install --prefix "$packages" --no-audit --no-fund --omit=dev \
        "node-addon-require-builtin@$VF_NODE_ADDON_VERSION" \
        "@deepseek-ai/dsh-acp-demo@$VF_DEEPSEEK_HARNESS_VERSION" \
        "@deepseek-ai/dsh-bash-local@$VF_DEEPSEEK_HARNESS_VERSION" \
        "@deepseek-ai/dsh-llm@$VF_DEEPSEEK_HARNESS_VERSION" \
        "@deepseek-ai/dsh-mcp-client@$VF_DEEPSEEK_HARNESS_VERSION" \
        "@deepseek-ai/dsh-subprocess-local@$VF_DEEPSEEK_HARNESS_VERSION" >/dev/null
    printf %s "$versions" > "$packages/.versions"
fi
"""


class DeepSeekHarnessConfig(HarnessConfig):
    version: str = Field(default="0.1.0-rc.7", pattern=r"^[A-Za-z0-9._+-]+$")
    """DeepSeek Harness release to install, pinned for reproducibility."""
    transport: Literal["chat_completions", "responses", "anthropic_messages"] = (
        "chat_completions"
    )
    """Model API transport."""


class DeepSeekHarness(ACPHarness[DeepSeekHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_SKILLS = True

    async def setup(self, runtime: Runtime) -> None:
        await ensure_node(runtime)
        logger.info(
            "deepseek-harness: ensuring DeepSeek Harness %s is installed",
            self.config.version,
        )
        lock = f"{DSH_DIR}.install.lock"
        guarded = (
            f'until ln -s "$$" {lock} 2>/dev/null; do '
            f"owner=$(readlink {lock}); "
            f'if ! kill -0 "$owner" 2>/dev/null; then '
            f'[ "$(readlink {lock})" != "$owner" ] || rm -f {lock}; fi; '
            f"sleep 0.1; done; "
            f'trap \'[ "$(readlink {lock})" != "$$" ] || rm -f {lock}\' EXIT; '
            f"sh -c {shlex.quote(INSTALL)}"
        )
        install = await runtime.run(
            ["sh", "-c", guarded],
            {
                "VF_DEEPSEEK_HARNESS_VERSION": self.config.version,
                "VF_NODE_ADDON_VERSION": NODE_ADDON_VERSION,
            },
        )
        if install.exit_code != 0:
            detail = (install.stderr or install.stdout).strip()[-500:]
            raise RuntimeError(f"DeepSeek Harness install failed: {detail}")
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
            raise ValueError("DeepSeek Harness ACP does not support disabling tools")

        system_prompt, prompt = self.resolve_prompt(data)
        run_dir = self.run_dir(trace)
        dsh_home = f"{run_dir}/dsh"
        await self.install_skills(runtime, f"{dsh_home}/skills")

        adapter_path = f"{run_dir}/adapter.mjs"
        await runtime.write(adapter_path, ADAPTER_SOURCE)
        bare_tool_prefixes = []
        composition = [
            {
                "id": "llm-verifiers",
                "name": "./adapter.mjs",
                "config": {
                    "endpoint": endpoint,
                    "transport": self.config.transport,
                    "model": ctx.model,
                    "contextWindow": DEFAULT_CONTEXT_WINDOW,
                    "bareToolPrefixes": bare_tool_prefixes,
                    **(
                        {"maxTokens": ctx.sampling.max_tokens}
                        if ctx.sampling.max_tokens is not None
                        else {}
                    ),
                },
            },
            {
                "id": "subprocess",
                "name": "@deepseek-ai/dsh-subprocess-local",
            },
            {"id": "bash", "name": "@deepseek-ai/dsh-bash-local"},
            {
                "id": "acp-agent",
                "name": "@deepseek-ai/dsh-acp-demo",
                "config": {
                    "provider": "verifiers",
                    "model": ctx.model,
                    "dshHome": dsh_home,
                    "persistenceRoot": f"{run_dir}/sessions",
                    "persistenceCompression": "none",
                    "persona": system_prompt or "",
                    "workspaceContext": {"maxBytes": 65536},
                    "skills": {"filesystem": {"watch": False}},
                    "toolBash": {"enableRunInBackground": False},
                    "toolJobs": False,
                    "goals": False,
                },
            },
        ]

        for index, (name, url) in enumerate(mcp_urls.items()):
            server_name = re.sub(r"[^A-Za-z0-9_-]", "_", name) or "server"
            if server_name != name or len(server_name) > 32:
                digest = hashlib.sha1(name.encode()).hexdigest()[:8]
                server_name = f"{server_name[:23]}_{digest}"
            if not name:
                bare_tool_prefixes.append(f"mcp__{server_name}__")
            composition.append(
                {
                    "id": f"mcp-{index}",
                    "name": "@deepseek-ai/dsh-mcp-client",
                    "config": {
                        "serverName": server_name,
                        "transport": "streamable-http",
                        "url": url,
                        "toolCallTimeoutMs": int(self.config.tool_timeout * 1000),
                        "failOnStartupError": True,
                    },
                }
            )

        config_path = f"{run_dir}/cordis.yml"
        await runtime.write(
            config_path, json.dumps(composition, ensure_ascii=False).encode()
        )
        return ACPConfig(
            env={
                **self.config.resolved_env,
                KEY_VAR: secret,
                "DSH_HOME": dsh_home,
                "DSH_AGENTS_HOME": f"{run_dir}/agents",
                "NO_COLOR": "1",
            },
            command=[f"{NODE_BIN_DIR}/node", DSH_BIN, "--config", config_path],
            prompt=prompt,
            # DeepSeek's ACP server rejects per-session MCP declarations. Its
            # Cordis MCP plugins above own the equivalent rollout-scoped tools.
            mcp_urls={},
        )

    async def cleanup(self, trace: Trace, runtime: Runtime) -> None:
        result = await runtime.run(["rm", "-rf", self.run_dir(trace)], {})
        if result.exit_code != 0:
            raise RuntimeError(
                f"failed to clean up DeepSeek Harness state: "
                f"{result.stderr.strip()[-500:]}"
            )

    @staticmethod
    def run_dir(trace: Trace) -> str:
        # Keeping the config below the npm prefix lets Cordis resolve its bare
        # plugin specifiers while the trace id isolates concurrent rollouts.
        return f"{PACKAGES_DIR}/runs/{trace.id}"
