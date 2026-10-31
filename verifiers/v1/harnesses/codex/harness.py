"""Run Codex through its App Server-backed ACP adapter."""

import hashlib
import json
import logging
import re
import tomllib
from pathlib import Path
from typing import Literal

import tomli_w

from verifiers.v1.acp import ACPConfig, ACPHarness, ACPTurn
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig, PinnedVersion
from verifiers.v1.harnesses.node import INSTALL as NODE_INSTALL
from verifiers.v1.harnesses.node import (
    NODE_BIN_DIR,
    NODE_DIR,
    NODE_VERSION,
    ensure_node,
)
from verifiers.v1.harnesses.utils.install import ensure_installed, remove_dir
from verifiers.v1.harnesses.utils.prebuilt import Prebuilt, prebuilt_name
from verifiers.v1.harnesses.utils.prebuilt import link as link_prebuilt
from verifiers.v1.runtimes import Runtime
from verifiers.v1.runtimes.deployment import ExecutionConnection
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

CODEX_DIR = "/var/tmp/vf-codex-{version}-{acp_version}"
PACKAGES_DIR = f"{CODEX_DIR}/acp"
ACP_VERSION = "2.5.0"
CODEX_BIN = f"{PACKAGES_DIR}/node_modules/.bin/codex"
ACP_BIN = f"{PACKAGES_DIR}/node_modules/.bin/codex-acp"
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


GATE_HOOK = (Path(__file__).resolve().parent / "gate.mjs").read_text()
GATE_HANDLER = {
    "type": "command",
    "command": f'{NODE_BIN_DIR}/node "$CODEX_HOME/vf-gate.mjs"',
    "timeout": 120,
    "async": False,
}


class CodexHarnessConfig(HarnessConfig):
    # Codex accepts no other wire API:
    # https://github.com/openai/codex/blob/rust-v0.166.0/codex-rs/model-provider-info/src/lib.rs#L115
    dialect: Literal["responses"] = "responses"
    version: PinnedVersion = "0.166.0"
    """Codex release to install, pinned for reproducibility."""
    code_mode: bool = False
    """Enable Code Mode when the model's metadata does not select its tool mode."""
    multi_agent: bool = False
    """Enable Codex's native multi-agent v2 tools."""


class CodexHarness(ACPHarness[CodexHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = False  # TODO
    SUPPORTS_MCP = True
    SUPPORTS_SKILLS = True
    SUPPORTS_TOOL_INTERCEPTION = True
    SUPPORTS_SPLIT_EXECUTION = True
    _executor: str | None = None
    """A split deployment's exec-server URL in the workspace."""
    _executor_cwd: str | None = None

    def acp_turn_result(self, trace: Trace, result: ACPTurn) -> None:
        # codex-acp returns terminal failures in metadata with stop_reason=end_turn; a
        # `warning` is a retry it already recovered from.
        failure = (
            result.response_metadata.get("jetbrains", {})
            .get("air", {})
            .get("sessionFailure")
        )
        if failure and failure["severity"] == "error":
            raise RuntimeError(f"Codex {failure['category']}: {failure['title']}")

    async def setup(self, runtime: Runtime) -> None:
        await self._install(runtime)
        await super().setup(runtime)

    def prebuilt(self) -> Prebuilt | None:
        if self.config.skills:
            return None  # skills install per run
        versions = {"version": self.config.version, "acp_version": ACP_VERSION}
        codex_bin = CODEX_BIN.format(**versions)
        return Prebuilt(
            name=prebuilt_name(
                "codex",
                NODE_INSTALL,
                NODE_VERSION,
                INSTALL,
                self.config.version,
                ACP_VERSION,
            ),
            dirs=(NODE_DIR, CODEX_DIR.format(**versions)),
            probe=f"{NODE_BIN_DIR}/node {codex_bin} --version",
            install=self._install,
        )

    async def _install(self, runtime: Runtime) -> None:
        if await link_prebuilt(runtime, self.prebuilt()):
            return
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
        await ensure_installed(
            runtime,
            directory=directory,
            ready=f"[ -f {ready} ] && [ -x {codex_bin} ] && [ -x {acp_bin} ]",
            install=script,
            env={
                **self.config.resolved_env,
                "VF_CODEX_VERSION": self.config.version,
                "VF_CODEX_ACP_VERSION": ACP_VERSION,
            },
            label="codex",
        )

    def validate_split_execution(self) -> None:
        super().validate_split_execution()
        if self.config.skills:
            raise ValueError("Codex split deployment does not support custom skills")

    async def setup_execution(
        self, runtime: Runtime, workspace: Runtime, connection: ExecutionConnection
    ) -> None:
        # Codex's own executor runs the native shell and file tools in the workspace.
        # Its client sends credentials only over TLS or loopback, so the listener takes
        # none; the LAN admits the harness's VM and the workspace's own services.
        await self._install(workspace)
        codex = CODEX_BIN.format(version=self.config.version, acp_version=ACP_VERSION)
        port = await connection.serve(
            workspace,
            [
                f"{NODE_BIN_DIR}/node",
                codex,
                "exec-server",
                "--listen",
                "ws://0.0.0.0:0",
            ],
            {},
            re.compile(r"^ws://\S+:(\d+)$", re.MULTILINE),
        )
        cwd = await workspace.run(["pwd", "-P"], {})
        if cwd.exit_code:
            raise RuntimeError(
                f"could not resolve the workspace directory: {cwd.stderr}"
            )
        self._executor = f"ws://{connection.host}:{port}"
        self._executor_cwd = cwd.stdout.strip()

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
            cwd=self._executor_cwd,
            # Codex reads MCP servers from the config written by build_env().
            mcp_urls={},
            system_prompt=system_prompt,
            client_capabilities={
                "_meta": {
                    "jetbrains": {
                        "air": {"version": 1, "capabilities": ["sessionFailure"]}
                    }
                }
            },
        )

    async def gate_tools(
        self, config: ACPConfig, runtime: Runtime, url: str, secret: str
    ) -> None:
        # Codex asks its ACP client only when a command escapes the sandbox, and never in
        # full access, so the gate is a PreToolUse hook instead.
        await runtime.write(
            f"{config.env['CODEX_HOME']}/vf-gate.mjs",
            GATE_HOOK.replace("__URL__", json.dumps(url))
            .replace("__SECRET__", json.dumps(secret))
            .encode(),
        )
        # Codex keys hook trust by the config's canonical path, resolving symlinks.
        home = await runtime.run(
            ["sh", "-c", 'cd "$1" && pwd -P', "vf-gate", config.env["CODEX_HOME"]],
            {},
        )
        if home.exit_code:
            raise RuntimeError(f"could not resolve Codex home: {home.stderr}")
        path = f"{home.stdout.strip()}/config.toml"
        settings = tomllib.loads((await runtime.read(path)).decode())
        # User hooks require a hash of Codex's normalized definition. Keeping the
        # definition and its trust entry in this rollout's home avoids shared state.
        identity = {"event_name": "pre_tool_use", "hooks": [GATE_HANDLER]}
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        settings["hooks"] = {
            "PreToolUse": [{"hooks": [GATE_HANDLER]}],
            "state": {f"{path}:pre_tool_use:0:0": {"trusted_hash": f"sha256:{digest}"}},
        }
        await runtime.write(path, tomli_w.dumps(settings).encode())

    async def cleanup(self, trace: Trace, runtime: Runtime) -> None:
        await remove_dir(runtime, self.trace_home(trace), "Codex home")

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
        await self.install_skills(runtime, f"{home}/skills")
        mcp_config = "features={mcp_2026_07_28=true}\n" + (
            "mcp_servers={"
            + ",".join(
                f"{json.dumps(name, ensure_ascii=False)}="
                f"{{url={json.dumps(url, ensure_ascii=False)},required=true,"
                'omit_tools_from=["deferred","code_mode"],'
                f"startup_timeout_sec=60.0,tool_timeout_sec={self.config.tool_timeout}}}"
                for name, url in mcp_urls.items()
            )
            + "}"
            if mcp_urls
            else ""
        )
        await runtime.write(f"{home}/config.toml", mcp_config.encode())
        if self._executor is not None:
            # The workspace's executor is the only one: the harness's VM runs no tools.
            await runtime.write(
                f"{home}/environments.toml",
                tomli_w.dumps(
                    {
                        "default": "workspace",
                        "include_local": False,
                        "environments": [{"id": "workspace", "url": self._executor}],
                    }
                ).encode(),
            )

        features: dict[str, object] = {
            "apps": False,
            "plugins": False,
            "code_mode": self.config.code_mode,
            "multi_agent": False,
            "multi_agent_v2": {"enabled": self.config.multi_agent},
            **{tool: False for tool in self.config.disabled_tools or []},
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
