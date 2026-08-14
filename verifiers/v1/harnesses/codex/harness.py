"""Run Codex through its App Server-backed ACP adapter."""

import hashlib
import json
import logging
import re
import shlex
from collections import Counter

from pydantic import Field

from verifiers.v1.acp import ACPConfig, ACPHarness
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harnesses.node import NODE_BIN_DIR, ensure_node
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

CODEX_DIR = "/var/tmp/vf-codex-{version}-{acp_version}"
PACKAGES_DIR = f"{CODEX_DIR}/acp"
ACP_VERSION = "1.2.0"
CODEX_BIN = f"{PACKAGES_DIR}/node_modules/.bin/codex"
ACP_BIN = f"{PACKAGES_DIR}/node_modules/.bin/codex-acp"
SKILLS_DIR = ".agents/skills"
INSTALL = r"""
set -e
export PATH="/var/tmp/vf-node/bin:$PATH"
rm -f {ready}
npm install --prefix {packages} --ignore-scripts --no-audit --no-fund \
    --omit=dev \
    "@agentclientprotocol/codex-acp@$VF_CODEX_ACP_VERSION" \
    "@openai/codex@$VF_CODEX_VERSION" >/dev/null
touch {ready}
"""


class CodexHarnessConfig(HarnessConfig):
    version: str = Field(default="0.147.0", pattern=r"^[A-Za-z0-9._+-]+$")
    """Codex release to install, pinned for reproducibility."""
    multi_agent: bool = False
    """Enable Codex's native multi-agent v2 tools."""


class CodexHarness(ACPHarness[CodexHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = False  # TODO
    SUPPORTS_MCP = True
    SUPPORTS_SKILLS = True

    async def setup(self, runtime: Runtime) -> None:
        await self.install_skills(runtime, SKILLS_DIR)
        await ensure_node(runtime)
        logger.info(
            "codex: ensuring Codex %s and codex-acp %s are installed",
            self.config.version,
            ACP_VERSION,
        )
        versions = {"version": self.config.version, "acp_version": ACP_VERSION}
        directory = CODEX_DIR.format(**versions)
        packages = PACKAGES_DIR.format(**versions)
        codex_bin = CODEX_BIN.format(**versions)
        acp_bin = ACP_BIN.format(**versions)
        ready = f"{directory}/.ready"
        script = INSTALL.replace("{packages}", packages).replace("{ready}", ready)
        ensure = shlex.quote(
            f"[ -f {ready} ] && [ -x {codex_bin} ] && [ -x {acp_bin} ] || ({script})"
        )
        guarded = (
            f"mkdir -p {directory} && "
            f'"$(command -v flock || command -v lockf)" {directory}/install.lock '
            f"sh -c {ensure}"
        )
        install = await runtime.run(
            ["sh", "-c", guarded],
            {
                **self.config.resolved_env,
                "VF_CODEX_VERSION": self.config.version,
                "VF_CODEX_ACP_VERSION": ACP_VERSION,
            },
        )
        if install.exit_code != 0:
            raise RuntimeError(f"codex install failed: {install.stderr.strip()[-500:]}")
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
        if data.system_prompt is not None and not isinstance(data.prompt, str):
            system_prompt, prompt = data.system_prompt, data.prompt
        else:
            system_prompt, prompt = self.resolve_prompt(data)
        env = await self.build_env(ctx, trace, runtime, endpoint, secret, mcp_urls)
        return ACPConfig(
            env=env,
            command=[
                f"{NODE_BIN_DIR}/node",
                ACP_BIN.format(version=self.config.version, acp_version=ACP_VERSION),
            ],
            prompt=prompt,
            # Codex reads MCP servers from the config written by build_env().
            mcp_urls={},
            system_prompt=system_prompt,
        )

    async def cleanup(self, trace: Trace, runtime: Runtime) -> None:
        result = await runtime.run(["rm", "-rf", self.trace_home(trace)], {})
        if result.exit_code != 0:
            raise RuntimeError(
                f"failed to clean up Codex home: {result.stderr.strip()[-500:]}"
            )

    @staticmethod
    def trace_home(trace: Trace) -> str:
        return f"/tmp/vf-codex-home-{trace.id}"

    async def build_env(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> dict[str, str]:
        home = self.trace_home(trace)
        created = await runtime.run(["mkdir", "-p", home], {})
        if created.exit_code != 0:
            raise RuntimeError(
                f"failed to create Codex home: {created.stderr.strip()[-500:]}"
            )

        mcp_config = (
            "mcp_servers={"
            + ",".join(
                f"{json.dumps(name, ensure_ascii=False)}="
                f"{{url={json.dumps(url, ensure_ascii=False)},required=true,"
                f"startup_timeout_sec=60.0,tool_timeout_sec={self.config.tool_timeout}}}"
                for name, url in mcp_urls.items()
            )
            + "}"
            if mcp_urls
            else ""
        )
        await runtime.write(f"{home}/config.toml", mcp_config.encode())

        namespace_bases = {
            name: (namespace if namespace.startswith("mcp__") else f"mcp__{namespace}")
            for name in mcp_urls
            for namespace in (re.sub(r"[^a-zA-Z0-9_]", "_", name) or "_",)
        }
        namespace_counts = Counter(namespace_bases.values())
        direct_mcp_namespaces: list[str] = []
        for name, namespace in namespace_bases.items():
            if namespace_counts[namespace] > 1:
                suffix = hashlib.sha1(f"{name}\0{name}\0".encode()).hexdigest()[:12]
                namespace = (
                    f"{namespace[:-2]}_{suffix}__"
                    if namespace.endswith("__")
                    else f"{namespace}_{suffix}"
                )
            direct_mcp_namespaces.append(namespace)
            if len(namespace) > 49:
                direct_mcp_namespaces.append(namespace[:49])

        features: dict[str, object] = {
            "apps": False,
            "plugins": False,
            "multi_agent": False,
            "multi_agent_v2": {"enabled": self.config.multi_agent},
            **{tool: False for tool in self.config.disabled_tools or []},
        }
        if direct_mcp_namespaces:
            features["code_mode"] = {
                "direct_only_tool_namespaces": list(
                    dict.fromkeys(direct_mcp_namespaces)
                )
            }
        config = {
            "model": ctx.model,
            "features": features,
        }
        return {
            **self.config.resolved_env,
            "CODEX_CONFIG": json.dumps(config),
            "CODEX_HOME": home,
            "DEFAULT_AUTH_REQUEST": json.dumps(
                {
                    "methodId": "gateway",
                    "_meta": {
                        "gateway": {
                            "baseUrl": endpoint,
                            "headers": {"Authorization": f"Bearer {secret}"},
                            "providerName": "Verifiers",
                        }
                    },
                }
            ),
            "INITIAL_AGENT_MODE": "agent-full-access",
            "NO_BROWSER": "1",
        }
