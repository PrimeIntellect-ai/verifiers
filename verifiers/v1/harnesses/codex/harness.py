"""Run Codex through its App Server-backed ACP adapter."""

import asyncio
import hashlib
import json
import shlex
import tomllib
from pathlib import Path
from typing import Literal

import tomli_w

from verifiers.v1.acp import ACPConfig, ACPHarness, ACPTurn
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig, PinnedVersion
from verifiers.v1.errors import HarnessError
from verifiers.v1.harnesses.node import NODE_BIN_DIR, ensure_node
from verifiers.v1.harnesses.utils.install import ensure_installed, remove_dir
from verifiers.v1.harnesses.utils.remote_runtime import prepare_remote_bridge
from verifiers.v1.runtimes import Runtime
from verifiers.v1.runtimes.deployment import ExecutionConnection
from verifiers.v1.runtimes.prepared import load_prepared, record_arguments
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace
from verifiers.v1.types import ToolCall
from verifiers.v1.utils.logging import get_logger

logger = get_logger(__name__)

CODEX_DIR = "/var/tmp/vf-codex-{version}-{acp_version}"
PACKAGES_DIR = f"{CODEX_DIR}/acp"
ACP_VERSION = "2.5.0"
CODEX_BIN = f"{PACKAGES_DIR}/node_modules/.bin/codex"
ACP_BIN = f"{PACKAGES_DIR}/node_modules/.bin/codex-acp"
EXECUTOR_DIR = "/var/tmp/agent-tools/codex-executor-{version}"
EXECUTOR_BIN = f"{EXECUTOR_DIR}/node_modules/.bin/codex"
NATIVE_EXECUTOR_BIN = f"{EXECUTOR_DIR}/bin/codex"
_LINUX_VENDOR = {
    "x86_64": ("@openai/codex-linux-x64", "x86_64-unknown-linux-musl"),
    "amd64": ("@openai/codex-linux-x64", "x86_64-unknown-linux-musl"),
    "aarch64": ("@openai/codex-linux-arm64", "aarch64-unknown-linux-musl"),
    "arm64": ("@openai/codex-linux-arm64", "aarch64-unknown-linux-musl"),
}
INSTALL = r"""
set -e
export PATH="/var/tmp/vf-node/bin:$PATH"
rm -f {ready}
npm install --prefix {packages} --ignore-scripts --no-audit --no-fund \
    --omit=dev \
    "@agentclientprotocol/codex-acp@$AGENT_CODEX_ACP_VERSION" \
    "@openai/codex@$AGENT_CODEX_VERSION" >/dev/null
touch {ready}
"""
EXECUTOR_INSTALL = r"""
set -e
export PATH="{node_bin}:$PATH"
npm install --prefix {directory} --ignore-scripts --no-audit --no-fund \
    --omit=dev "@openai/codex@$AGENT_CODEX_VERSION" >/dev/null
"""


_DOWNLOAD_NATIVE_CODEX = r"""
import os, shutil, sys, tarfile, urllib.request, uuid

url, triple, target = sys.argv[1], sys.argv[2], sys.argv[3]
parent = os.path.dirname(target.rstrip("/")) or "/"
os.makedirs(parent, exist_ok=True)
staging = f"{target.rstrip('/')}.tmp.{uuid.uuid4().hex}"
os.makedirs(staging, exist_ok=True)
os.chmod(staging, 0o755)
prefix = f"package/vendor/{triple}/"
wanted = ("bin/codex", "codex-resources/bwrap", "codex-path/rg")
found = set()
try:
    req = urllib.request.Request(url, headers={"User-Agent": "verifiers"})
    with urllib.request.urlopen(req, timeout=60) as resp, tarfile.open(
        fileobj=resp, mode="r|gz"
    ) as archive:
        for member in archive:
            if not member.name.startswith(prefix):
                continue
            rel = member.name[len(prefix):]
            if rel in wanted:
                dest = os.path.join(staging, rel)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                stream = archive.extractfile(member)
                if stream is None:
                    raise RuntimeError(f"missing content for {rel}")
                with open(dest, "wb") as out:
                    shutil.copyfileobj(stream, out, length=262144)
                os.chmod(dest, 0o755)
                found.add(rel)
    if "bin/codex" not in found:
        raise RuntimeError("codex binary missing from archive")
    if os.path.exists(target):
        shutil.rmtree(target)
    os.replace(staging, target)
finally:
    shutil.rmtree(staging, ignore_errors=True)
"""


GATE_HOOK = (Path(__file__).resolve().parent / "gate.mjs").read_text()
GATE_HANDLER = {
    "type": "command",
    "command": f'{NODE_BIN_DIR}/node "$CODEX_HOME/tool-gate.mjs"',
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
    SUPPORTS_SPLIT_MCP = True
    _executor_cwd: str | None = None
    _install_event: asyncio.Event | None = None
    _install_failed: bool = False
    _staged_configs: dict[str, dict] | None = None

    def _ready_event(self) -> asyncio.Event:
        if self._install_event is None:
            self._install_event = asyncio.Event()
        return self._install_event

    async def remote_tools(self, runtime: Runtime, data: TaskData) -> None:
        # Codex's native execution environment transports files and persistent
        # processes, including nested Code Mode calls, without emulating tools.
        return None

    def can_gate_tool(self, call: ToolCall) -> bool:
        # 0.166.0 deliberately omits PreToolUse for write_stdin. Only these
        # native tools have a synchronous veto hook that can carry a replacement.
        return call.name in {"exec_command", "shell", "shell_command", "apply_patch"}

    def check_tool_call(self, call: ToolCall, *, intercepted: bool) -> None:
        if intercepted and call.name in {"exec", "wait"}:
            raise HarnessError(
                "Codex Code Mode cannot enforce request interception on nested "
                "write_stdin calls; use a model with direct tools for this task"
            )

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
        try:
            await self._install(runtime)
        except BaseException:
            self._install_failed = True
            raise
        finally:
            self._ready_event().set()
        await super().setup(runtime)

    async def _install(self, runtime: Runtime) -> None:
        await ensure_node(runtime)
        logger.debug(
            "Ensuring Codex %s and codex-acp %s are installed",
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
                "AGENT_CODEX_VERSION": self.config.version,
                "AGENT_CODEX_ACP_VERSION": ACP_VERSION,
            },
            label="codex",
        )

    def validate_split_execution(self) -> None:
        super().validate_split_execution()
        if self.config.skills:
            raise ValueError("Codex split deployment does not support custom skills")

    async def _sync_native_executor(
        self, runtime: Runtime, setup_runtime: Runtime, connection: ExecutionConnection
    ) -> list[str] | None:
        """Install the static musl Codex binary in B without Node or npm."""
        target_root = setup_runtime.with_user(None)
        python = await target_root.resolve_python()
        b_vendor = _LINUX_VENDOR.get(target_root.backend.cache.arch or "")
        if b_vendor is None:
            return None
        pkg_name, triple = b_vendor
        directory = EXECUTOR_DIR.format(version=self.config.version)
        native_bin = NATIVE_EXECUTOR_BIN.format(version=self.config.version)
        ready_check = f"[ -x {shlex.quote(native_bin)} ]"
        await load_prepared(target_root)
        install_key = f"install:{directory}:native-{triple}-{self.config.version}"
        if install_key not in target_root.backend.cache.installed:
            tarball_suffix = pkg_name.removeprefix("@openai/codex-")
            url = (
                "https://registry.npmjs.org/@openai/codex/-/"
                f"codex-{self.config.version}-{tarball_suffix}.tgz"
            )
            dl = await target_root.run(
                [
                    python,
                    "-I",
                    "-c",
                    _DOWNLOAD_NATIVE_CODEX,
                    url,
                    triple,
                    directory,
                ],
                {},
            )
            if dl.exit_code != 0:
                if connection.transport is None:
                    return None
                await self._ready_event().wait()
                if self._install_failed:
                    return None
                await runtime.resolve_python()
                if _LINUX_VENDOR.get(runtime.backend.cache.arch or "") != b_vendor:
                    return None
                packages = PACKAGES_DIR.format(
                    version=self.config.version, acp_version=ACP_VERSION
                )
                source_vendor = f"{packages}/node_modules/{pkg_name}/vendor/{triple}"
                await connection.transport.sync_dir(
                    runtime.with_user(None),
                    source_vendor,
                    target_root,
                    directory,
                    exclude=("codex-resources/voice", "bin/codex-code-mode-host"),
                )
            await target_root.run(
                [
                    python,
                    *record_arguments(
                        target_root,
                        install_key,
                        {"install": directory, "ready": ready_check},
                    ),
                ],
                {},
            )
            target_root.backend.cache.installed.add(install_key)
        return [native_bin, "exec-server", "--listen", "stdio"]

    async def setup_execution(
        self, runtime: Runtime, workspace: Runtime, connection: ExecutionConnection
    ) -> None:
        # Codex's own executor runs native tools through the same connection
        # on every runtime. The harness's VM has no local execution environment.
        native_cmd = await self._sync_native_executor(
            runtime, connection.setup_runtime, connection
        )
        if native_cmd is not None:
            directory = EXECUTOR_DIR.format(version=self.config.version)
            connection.register(
                "codex",
                native_cmd,
                {
                    "CODEX_MANAGED_PACKAGE_ROOT": directory,
                    "CODEX_MANAGED_BY_NPM": "1",
                },
            )
        else:
            await self.setup_executor(connection.setup_runtime)
            codex = EXECUTOR_BIN.format(version=self.config.version)
            connection.register(
                "codex",
                [f"{NODE_BIN_DIR}/node", codex, "exec-server", "--listen", "stdio"],
                {},
            )
        self._executor_cwd = workspace.info.workdir

    async def setup_executor(self, runtime: Runtime) -> None:
        """Install only the credential-free executor in B during trusted setup."""
        await ensure_node(runtime)
        directory = EXECUTOR_DIR.format(version=self.config.version)
        await ensure_installed(
            runtime,
            directory=directory,
            ready=f"[ -x {EXECUTOR_BIN.format(version=self.config.version)} ]",
            install=EXECUTOR_INSTALL.replace("{directory}", directory).replace(
                "{node_bin}", NODE_BIN_DIR
            ),
            # Harness environment overrides belong to A, never the task worker.
            env={"AGENT_CODEX_VERSION": self.config.version},
            label="Codex executor",
        )

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
        home = config.env["CODEX_HOME"]
        path = f"{home}/config.toml"
        staged = (self._staged_configs or {}).pop(home, None)
        if staged is not None:
            settings = dict(staged)
        else:
            settings = tomllib.loads((await runtime.read(path)).decode())
        # User hooks require a hash of Codex's normalized definition. Keeping the
        # definition and its trust entry in this rollout's home avoids shared state.
        # Include `/private/tmp` alongside `/tmp` for macOS symlinked temp roots without a shell probe.
        identity = {"event_name": "pre_tool_use", "hooks": [GATE_HANDLER]}
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        candidate_paths = [path]
        if path.startswith("/tmp/"):
            candidate_paths.append(f"/private{path}")
        settings["hooks"] = {
            "PreToolUse": [{"hooks": [GATE_HANDLER]}],
            "state": {
                f"{candidate}:pre_tool_use:0:0": {"trusted_hash": f"sha256:{digest}"}
                for candidate in candidate_paths
            },
        }
        await asyncio.gather(
            runtime.write(
                f"{home}/tool-gate.mjs",
                GATE_HOOK.replace("__URL__", json.dumps(url))
                .replace("__SECRET__", json.dumps(secret))
                .encode(),
            ),
            runtime.write(path, tomli_w.dumps(settings).encode()),
        )

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
        writes = []
        if runtime._execution_bridge is not None:
            command = await prepare_remote_bridge(
                runtime,
                f"{endpoint.removesuffix('/v1')}/execution",
                secret,
                "codex",
                protocol="codex",
            )
            # The pinned native environment provider routes every filesystem and
            # process operation through this stdio worker. Removing the local
            # environment also prevents explicit environment_id="local" fallback.
            environments = {
                "default": "task",
                "include_local": False,
                "environments": [
                    {"id": "task", "program": command[0], "args": command[1:]}
                ],
            }
            writes.append(
                runtime.write(
                    f"{home}/environments.toml", tomli_w.dumps(environments).encode()
                )
            )
        mcp_headers = self.config.resolve_mcp_headers(mcp_urls)
        mcp_config = "features={mcp_2026_07_28=true}\n" + (
            "mcp_servers={"
            + ",".join(
                f"{json.dumps(name, ensure_ascii=False)}="
                f"{{url={json.dumps(url, ensure_ascii=False)},required=true,"
                + "http_headers={"
                + ",".join(
                    f"{json.dumps(key)}={json.dumps(value)}"
                    for key, value in mcp_headers.get(name, {}).items()
                )
                + "},"
                + 'omit_tools_from=["deferred","code_mode"],'
                + f"startup_timeout_sec=60.0,tool_timeout_sec={self.config.tool_timeout}}}"
                for name, url in mcp_urls.items()
            )
            + "}"
            if mcp_urls
            else ""
        )
        if self._staged_configs is None:
            self._staged_configs = {}
        self._staged_configs[home] = tomllib.loads(mcp_config)
        writes.append(runtime.write(f"{home}/config.toml", mcp_config.encode()))
        await asyncio.gather(*writes)
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
