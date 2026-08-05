"""Run Codex through its App Server-backed ACP adapter."""

import hashlib
import json
import logging
import re
import shlex
from collections import Counter

from verifiers.v1.acp import ACP
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness
from verifiers.v1.harnesses._node import NODE_BIN_DIR, ensure_node
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

PROVIDER = "intercept"
KEY_VAR = "CODEX_INTERCEPT_KEY"

CODEX_DIR = "/tmp/vf-codex"
CODEX_BIN = f"{CODEX_DIR}/bin/codex"
PACKAGES_DIR = f"{CODEX_DIR}/acp"
ACP_VERSION = "1.1.7"
ACP_BIN = f"{PACKAGES_DIR}/node_modules/.bin/codex-acp"
ACP_COMMAND = [
    "sh",
    "-c",
    f'export PATH="{NODE_BIN_DIR}:$PATH"; exec {ACP_BIN}',
]
SKILLS_DIR = ".agents/skills"
INSTALL = r"""
set -e
export PATH="/tmp/vf-node/bin:$PATH"
versions="$VF_CODEX_VERSION:$VF_CODEX_ACP_VERSION"
if [ "$(cat /tmp/vf-codex/.versions 2>/dev/null)" = "$versions" ] \
    && [ -x /tmp/vf-codex/bin/codex ] \
    && [ -x /tmp/vf-codex/acp/node_modules/.bin/codex-acp ]; then
    exit 0
fi
command -v curl >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq curl >/dev/null; }
case "$(uname -m)" in aarch64|arm64) arch=aarch64 ;; *) arch=x86_64 ;; esac
triple="${arch}-unknown-linux-musl"
mkdir -p /tmp/vf-codex/bin
curl -fsSL "https://github.com/openai/codex/releases/download/rust-v$VF_CODEX_VERSION/codex-${triple}.tar.gz" \
    | tar -xz -C /tmp/vf-codex/bin
mv "/tmp/vf-codex/bin/codex-${triple}" /tmp/vf-codex/bin/codex
chmod +x /tmp/vf-codex/bin/codex
npm install --prefix /tmp/vf-codex/acp --ignore-scripts --no-audit --no-fund \
    --omit=dev --omit=optional \
    "@agentclientprotocol/codex-acp@$VF_CODEX_ACP_VERSION" >/dev/null
printf %s "$versions" > /tmp/vf-codex/.versions
"""

CODEX_ACP = ACP()


class CodexHarnessConfig(HarnessConfig):
    version: str = "0.145.0"
    """Codex release to install (the `rust-v<version>` GitHub release); pinned for reproducibility."""
    multi_agent: bool = False
    """Enable Codex's native multi-agent v2 tools."""


class CodexHarness(Harness[CodexHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = False  # TODO
    SUPPORTS_MCP = True
    SUPPORTS_RESUME = True
    SUPPORTS_SKILLS = True

    async def setup(self, runtime: Runtime) -> None:
        await self.install_skills(runtime, SKILLS_DIR)
        await ensure_node(runtime)
        logger.info(
            "codex: ensuring Codex %s and codex-acp %s are installed",
            self.config.version,
            ACP_VERSION,
        )
        guarded = (
            f"mkdir -p {CODEX_DIR} && "
            f'"$(command -v flock || command -v lockf)" {CODEX_DIR}/install.lock '
            f"sh -c {shlex.quote(INSTALL)}"
        )
        install = await runtime.run(
            ["sh", "-c", guarded],
            {
                "VF_CODEX_VERSION": self.config.version,
                "VF_CODEX_ACP_VERSION": ACP_VERSION,
            },
        )
        if install.exit_code != 0:
            raise RuntimeError(f"codex install failed: {install.stderr.strip()[-500:]}")
        await CODEX_ACP.setup(self, runtime)

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
        if (
            data.system_prompt is not None
            and data.prompt is not None
            and not isinstance(data.prompt, str)
        ):
            system_prompt, prompt = data.system_prompt, data.prompt
        else:
            system_prompt, prompt = self.resolve_prompt(data)
        env = await self._env(ctx, trace, runtime, endpoint, secret, mcp_urls)
        return await CODEX_ACP.run(
            runtime,
            env,
            ACP_COMMAND,
            prompt,
            mcp_urls=mcp_urls,
            system_prompt=system_prompt,
            session_path=f"{self._home(trace)}/acp-session",
        )

    async def cleanup(self, trace: Trace, runtime: Runtime) -> None:
        result = await runtime.run(["rm", "-rf", self._home(trace)], {})
        if result.exit_code != 0:
            raise RuntimeError(
                f"failed to clean up Codex home: {result.stderr.strip()[-500:]}"
            )

    @staticmethod
    def _home(trace: Trace) -> str:
        return f"/tmp/vf-codex-home-{trace.id}"

    async def _env(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> dict[str, str]:
        home = self._home(trace)
        created = await runtime.run(["mkdir", "-p", home], {})
        if created.exit_code != 0:
            raise RuntimeError(
                f"failed to create Codex home: {created.stderr.strip()[-500:]}"
            )

        mcp_config = (
            "mcp_servers={"
            + ",".join(
                (
                    f"{json.dumps(name, ensure_ascii=False)}="
                    f"{{url={json.dumps(url, ensure_ascii=False)},required=true,"
                    f"startup_timeout_sec=60.0,tool_timeout_sec={self.config.tool_timeout}}}"
                )
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
            "model_provider": PROVIDER,
            "model_providers": {
                PROVIDER: {
                    "name": PROVIDER,
                    "base_url": endpoint,
                    "env_key": KEY_VAR,
                    "wire_api": "responses",
                    "requires_openai_auth": False,
                }
            },
            "features": features,
        }
        return {
            **self.config.resolved_env,
            "CODEX_API_KEY": secret,
            KEY_VAR: secret,
            "APP_SERVER_LOGS": f"{home}/logs",
            "CODEX_CONFIG": json.dumps(config),
            "CODEX_HOME": home,
            "CODEX_PATH": CODEX_BIN,
            "DEFAULT_AUTH_REQUEST": json.dumps({"methodId": "api-key"}),
            "INITIAL_AGENT_MODE": "agent-full-access",
            "MODEL_PROVIDER": PROVIDER,
            "NO_BROWSER": "1",
        }
