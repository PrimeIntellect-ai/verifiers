"""Run Codex through its App Server-backed ACP adapter."""

import asyncio
import hashlib
import io
import json
import shlex
import tarfile
import tomllib
from pathlib import Path
from typing import Literal

import httpx
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
ACP_VERSION = "2.7.0"
CODEX_BIN = f"{PACKAGES_DIR}/node_modules/.bin/codex"
ACP_BIN = f"{PACKAGES_DIR}/node_modules/.bin/codex-acp"
EXECUTOR_DIR = "/var/tmp/agent-tools/codex-executor-{version}"
NATIVE_EXECUTOR_BIN = f"{EXECUTOR_DIR}/bin/codex"
_LINUX_VENDOR = {
    "x86_64": ("linux-x64", "x86_64-unknown-linux-musl"),
    "amd64": ("linux-x64", "x86_64-unknown-linux-musl"),
    "aarch64": ("linux-arm64", "aarch64-unknown-linux-musl"),
    "arm64": ("linux-arm64", "aarch64-unknown-linux-musl"),
}
_EXECUTOR_BUNDLE_CACHE: dict[tuple[str, str], bytes] = {}
_EXECUTOR_BUNDLE_LOCKS: dict[tuple[str, str], asyncio.Lock] = {}

INSTALL = r"""
set -e
export PATH="/var/tmp/vf-node/bin:$PATH"
rm -f {ready}
# Resolve the adapter's bundled CLI to the harness pin, including across minor releases.
mkdir -p {packages}
cd {packages}
printf '{"overrides":{"@agentclientprotocol/codex-acp":{"@openai/codex":"%s"}}}\n' \
    "$AGENT_CODEX_VERSION" > package.json
npm install --ignore-scripts --no-audit --no-fund \
    --omit=dev \
    "@agentclientprotocol/codex-acp@$AGENT_CODEX_ACP_VERSION" \
    "@openai/codex@$AGENT_CODEX_VERSION" >/dev/null
touch {ready}
"""


async def _cached_executor_bundle(
    version: str, tarball_suffix: str, triple: str
) -> bytes:
    """Download and repack Codex's static musl executor once per host process."""
    key = (version, triple)
    if cached := _EXECUTOR_BUNDLE_CACHE.get(key):
        return cached
    lock = _EXECUTOR_BUNDLE_LOCKS.setdefault(key, asyncio.Lock())
    async with lock:
        if cached := _EXECUTOR_BUNDLE_CACHE.get(key):
            return cached
        url = f"https://registry.npmjs.org/@openai/codex/-/codex-{version}-{tarball_suffix}.tgz"
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "verifiers"})
            resp.raise_for_status()
            raw = resp.content

        def repack() -> bytes:
            prefix = f"package/vendor/{triple}/"
            wanted = {"bin/codex", "codex-resources/bwrap", "codex-path/rg"}
            found: set[str] = set()
            buf = io.BytesIO()
            with (
                tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as src,
                tarfile.open(fileobj=buf, mode="w:gz", compresslevel=1) as dst,
            ):
                for member in src:
                    if not member.name.startswith(prefix):
                        continue
                    rel = member.name[len(prefix) :]
                    if rel in wanted:
                        stream = src.extractfile(member)
                        if stream is None:
                            raise RuntimeError(f"missing content for {rel}")
                        data = stream.read()
                        info = tarfile.TarInfo(name=rel)
                        info.size = len(data)
                        info.mode = 0o755
                        dst.addfile(info, io.BytesIO(data))
                        found.add(rel)
            if "bin/codex" not in found:
                raise RuntimeError("codex binary missing from archive")
            return buf.getvalue()

        bundle = await asyncio.to_thread(repack)
        _EXECUTOR_BUNDLE_CACHE[key] = bundle
        return bundle


GATE_HOOK = (Path(__file__).resolve().parent / "gate.mjs").read_text()
GATE_HANDLER = {
    "type": "command",
    "command": f'{NODE_BIN_DIR}/node "$CODEX_HOME/tool-gate.mjs"',
    "timeout": 120,
    "async": False,
}


class CodexHarnessConfig(HarnessConfig):
    # Codex accepts no other wire API:
    # https://github.com/openai/codex/blob/rust-v0.170.0/codex-rs/model-provider-info/src/lib.rs#L119
    dialect: Literal["responses"] = "responses"
    version: PinnedVersion = "0.170.0"
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
    _executor_cwd: str | None = None
    _staged_configs: dict[str, dict] | None = None

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
        await super().setup(runtime)

    async def setup_execution(
        self, runtime: Runtime, workspace: Runtime, connection: ExecutionConnection
    ) -> None:
        """Install the static musl Codex binary in B without Node or npm."""
        target_root = connection.setup_runtime.with_user(None)
        python = await target_root.resolve_python()
        b_vendor = _LINUX_VENDOR.get(target_root.backend.cache.arch or "")
        if b_vendor is None:
            raise HarnessError(
                f"unsupported workspace architecture for Codex executor: {target_root.backend.cache.arch!r}"
            )
        tarball_suffix, triple = b_vendor
        directory = EXECUTOR_DIR.format(version=self.config.version)
        native_bin = NATIVE_EXECUTOR_BIN.format(version=self.config.version)
        ready_check = f"[ -x {shlex.quote(native_bin)} ]"
        await load_prepared(target_root)
        install_key = f"install:{directory}:native-{triple}-{self.config.version}"
        if install_key not in target_root.backend.cache.installed:
            bundle = await _cached_executor_bundle(
                self.config.version, tarball_suffix, triple
            )
            archive = f"{directory}.tgz"
            await target_root.write(archive, bundle)
            rec = shlex.join(
                [
                    python,
                    *record_arguments(
                        target_root,
                        install_key,
                        {"install": directory, "ready": ready_check},
                    ),
                ]
            )
            dir_q, arc_q = shlex.quote(directory), shlex.quote(archive)
            res = await target_root.run(
                [
                    "sh",
                    "-c",
                    (
                        f'set -e; stage={dir_q}.tmp.$$; mkdir -p "$stage"; '
                        f'tar -xzf {arc_q} -C "$stage"; rm -f {arc_q}; '
                        f'rm -rf {dir_q}; mv "$stage" {dir_q}; chmod -R a+rX {dir_q}; {rec}'
                    ),
                ],
                {},
            )
            if res.exit_code != 0:
                raise HarnessError(
                    f"Codex executor install failed: {(res.stderr or res.stdout).strip()[-500:]}"
                )
            target_root.backend.cache.installed.add(install_key)
        connection.register(
            "codex",
            [native_bin, "exec-server", "--listen", "stdio"],
            {
                "CODEX_MANAGED_PACKAGE_ROOT": directory,
                "CODEX_MANAGED_BY_NPM": "1",
            },
        )
        self._executor_cwd = workspace.info.workdir

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
            # An untitled session starts a separate model turn to generate a UI title.
            session_title=f"Verifiers rollout {trace.id}",
            client_capabilities={
                "_meta": {
                    "jetbrains": {
                        "air": {
                            "version": 1,
                            "capabilities": ["sessionFailure", "sessionIndex"],
                        }
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
        if self.config.skills and runtime._execution_bridge is not None:
            await self.install_skills(
                runtime._execution_bridge.setup_runtime, f"{home}/skills"
            )
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
        # The app server and its ACP sessions must start with the same provider.
        settings = {
            "model_provider": "custom-gateway",
            "model_providers": {
                "custom-gateway": {
                    "name": "Verifiers",
                    "base_url": endpoint,
                    "wire_api": "responses",
                    "env_key": "CODEX_INTERCEPT_KEY",
                    "http_headers": {"X-Client-Feature-ID": "codex"},
                }
            },
            "features": {"mcp_2026_07_28": True},
            "mcp_servers": {
                name: {
                    "url": str(url),
                    "required": True,
                    "http_headers": mcp_headers.get(name, {}),
                    "omit_tools_from": ["deferred", "code_mode"],
                    "startup_timeout_sec": 60.0,
                    "tool_timeout_sec": self.config.tool_timeout,
                }
                for name, url in mcp_urls.items()
            },
        }
        if self._staged_configs is None:
            self._staged_configs = {}
        self._staged_configs[home] = dict(settings)
        writes.append(
            runtime.write(f"{home}/config.toml", tomli_w.dumps(settings).encode())
        )
        await asyncio.gather(*writes)
        disabled = set(self.config.disabled_tools or ())
        features: dict[str, object] = {
            "apps": False,
            "plugins": False,
            "code_mode": self.config.code_mode,
            "multi_agent": False,
            "multi_agent_v2": {"enabled": self.config.multi_agent},
            **{tool: False for tool in disabled - {"web_search"}},
        }
        config: dict[str, object] = {
            "model": ctx.model,
            "features": features,
            **({"web_search": "disabled"} if "web_search" in disabled else {}),
        }
        return {
            **self.config.resolved_env,
            # A rollout has no interactive composer that needs suggested follow-ups.
            "CODEX_ACP_PROMPT_SUGGESTIONS": "false",
            "CODEX_CONFIG": json.dumps(config),
            "CODEX_HOME": home,
            "CODEX_INTERCEPT_KEY": secret,
            "INITIAL_AGENT_MODE": "agent-full-access",
            "NO_BROWSER": "1",
        }
