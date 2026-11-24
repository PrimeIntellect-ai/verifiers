"""Run Codex through its App Server-backed ACP adapter."""

import asyncio
import contextlib
import gzip
import hashlib
import json
import uuid
from pathlib import Path, PurePosixPath
from typing import Literal
from weakref import WeakValueDictionary

import tomli_w

from verifiers.v1.acp import ACPConfig, ACPHarness, ACPTurn
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig, PinnedVersion
from verifiers.v1.errors import HarnessError
from verifiers.v1.harnesses.utils.install import (
    cached_npm_tarball_members,
    ensure_installed,
)
from verifiers.v1.harnesses.utils.remote_runtime import prepare_remote_bridge
from verifiers.v1.runtimes import Runtime
from verifiers.v1.runtimes.deployment import ExecutionConnection
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace
from verifiers.v1.types import ToolCall
from verifiers.v1.utils.aio import run_shielded
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
_RAW_WRITE_LOCKS: WeakValueDictionary[
    tuple[asyncio.AbstractEventLoop, str, str], asyncio.Lock
] = WeakValueDictionary()

INSTALL = r"""
set -e
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

_PUBLISH_EXECUTOR = """
stage=$1; parent=$2; suffix=$3; shift 3
for file do
    if [ -n "$suffix" ]; then gzip -d "$stage/$file$suffix"; fi
    chmod 755 "$stage/$file"
done
chmod -R a+rX "$stage"
mv "$stage" "$parent/" || test -x "$parent/${stage##*/}/bin/codex"
"""


async def _executor_files(
    version: str, tarball_suffix: str, triple: str
) -> dict[str, bytes]:
    url = f"https://registry.npmjs.org/@openai/codex/-/codex-{version}-{tarball_suffix}.tgz"
    prefix = f"package/vendor/{triple}/"
    wanted = {"bin/codex", "codex-resources/bwrap", "codex-path/rg"}

    def select(name: str) -> str | None:
        rel = name.removeprefix(prefix)
        return rel if rel != name and rel in wanted else None

    files = await cached_npm_tarball_members(
        url,
        select,
        transform=lambda raw: gzip.compress(raw, compresslevel=1),
    )
    if "bin/codex" not in files:
        raise HarnessError("codex binary missing from archive")
    return files


async def _install_executor(
    runtime: Runtime, version: str, tarball_suffix: str, triple: str
) -> None:
    directory = PurePosixPath(EXECUTOR_DIR.format(version=version))
    cache = runtime.backend.cache
    key = f"codex-executor:{directory}:{triple}"
    async with cache.setup_lock:
        if key in cache.installed:
            return
        probe = await runtime.run(
            [
                "sh",
                "-c",
                'if [ -x "$1" ]; then echo ready; elif command -v gzip >/dev/null; then echo gzip; else echo raw; fi',
                "codex",
                str(directory / "bin/codex"),
            ],
            {},
        )
        if probe.exit_code:
            raise HarnessError(f"Codex executor check failed: {probe.stderr.strip()}")
        if probe.stdout.strip() != "ready":
            files = await _executor_files(version, tarball_suffix, triple)
            scratch = directory.parent / f".codex-{uuid.uuid4().hex}"
            staged = scratch / directory.name
            suffix = ".gz" if probe.stdout.strip() == "gzip" else ""
            raw_lock = _RAW_WRITE_LOCKS.setdefault(
                (asyncio.get_running_loop(), version, triple), asyncio.Lock()
            )
            try:
                for path, data in files.items():
                    target = str(staged / (path + suffix))
                    if suffix:
                        await runtime.write(target, data)
                    else:
                        async with raw_lock:
                            await runtime.write(
                                target, await asyncio.to_thread(gzip.decompress, data)
                            )
                result = await runtime.run(
                    [
                        "sh",
                        "-ec",
                        _PUBLISH_EXECUTOR,
                        "codex",
                        str(staged),
                        str(directory.parent),
                        suffix,
                        *files,
                    ],
                    {},
                )
                if result.exit_code:
                    raise HarnessError(
                        f"Codex executor install failed: {(result.stderr or result.stdout).strip()[-500:]}"
                    )
            finally:
                with contextlib.suppress(Exception):
                    await run_shielded(runtime.run(["rm", "-rf", str(scratch)], {}))
        cache.installed.add(key)


GATE_HOOK = (Path(__file__).resolve().parent / "gate.mjs").read_text()
GATE_HANDLER = {
    "type": "command",
    "command": 'node "$CODEX_HOME/tool-gate.mjs"',
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
    HARNESS_TARGET = "codex"
    APPENDS_SYSTEM_PROMPT = False  # TODO
    SUPPORTS_MCP = True
    SUPPORTS_SKILLS = True
    SUPPORTS_TOOL_INTERCEPTION = True
    _executor_cwd: str | None = None

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
        self, runtime: Runtime, connection: ExecutionConnection
    ) -> None:
        """Install the static musl Codex binary in the worker sandbox without Node or npm."""
        target_root = connection.runtime.with_user(None)
        arch = target_root.backend.cache.arch
        if not arch:
            arch = (await target_root.run(["uname", "-m"], {})).stdout.strip()
        b_vendor = _LINUX_VENDOR.get(arch)
        if b_vendor is None:
            raise HarnessError(
                f"unsupported worker architecture for Codex executor: {arch!r}"
            )
        tarball_suffix, triple = b_vendor
        directory = EXECUTOR_DIR.format(version=self.config.version)
        native_bin = NATIVE_EXECUTOR_BIN.format(version=self.config.version)
        await _install_executor(
            target_root, self.config.version, tarball_suffix, triple
        )
        connection.register(
            "codex",
            [native_bin, "exec-server", "--listen", "stdio"],
            {
                "CODEX_MANAGED_PACKAGE_ROOT": directory,
                "CODEX_MANAGED_BY_NPM": "1",
            },
        )
        self._executor_cwd = connection.runtime.info.workdir

    async def prepare_acp(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
        tool_interception_url: str | None = None,
    ) -> ACPConfig:
        if data.system_prompt is not None and not isinstance(data.prompt, str):
            system_prompt, prompt = data.system_prompt, data.prompt
        else:
            system_prompt, prompt = self.resolve_prompt(data)
        env = await self.build_env(
            ctx,
            trace,
            runtime,
            endpoint,
            secret,
            mcp_urls,
            tool_interception_url=tool_interception_url,
        )
        return ACPConfig(
            env=env,
            command=[
                "node",
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

    async def build_env(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        *,
        tool_interception_url: str | None = None,
    ) -> dict[str, str]:
        home = f"/tmp/vf-codex-home-{trace.id}"
        if runtime._execution_bridge is None:
            raise HarnessError("Codex requires an execution bridge")
        await self.install_skills(
            runtime, f"{home}/skills", mirror_to=runtime._execution_bridge.runtime
        )
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
        writes = [
            runtime.write(
                f"{home}/environments.toml", tomli_w.dumps(environments).encode()
            )
        ]
        mcp_headers = self.config.resolve_mcp_headers(mcp_urls)
        config_path = f"{home}/config.toml"
        # The app server and its ACP sessions must start with the same provider.
        settings: dict[str, object] = {
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
        if tool_interception_url is not None:
            # Codex asks its ACP client only when a command escapes the sandbox, and never in
            # full access, so the gate is a PreToolUse hook instead.
            identity = {"event_name": "pre_tool_use", "hooks": [GATE_HANDLER]}
            digest = hashlib.sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            candidate_paths = [config_path]
            if config_path.startswith("/tmp/"):
                candidate_paths.append(f"/private{config_path}")
            settings["hooks"] = {
                "PreToolUse": [{"hooks": [GATE_HANDLER]}],
                "state": {
                    f"{candidate}:pre_tool_use:0:0": {
                        "trusted_hash": f"sha256:{digest}"
                    }
                    for candidate in candidate_paths
                },
            }
            writes.append(
                runtime.write(
                    f"{home}/tool-gate.mjs",
                    GATE_HOOK.replace("__URL__", json.dumps(tool_interception_url))
                    .replace("__SECRET__", json.dumps(secret))
                    .encode(),
                )
            )
        writes.append(runtime.write(config_path, tomli_w.dumps(settings).encode()))
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
