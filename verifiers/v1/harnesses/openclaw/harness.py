"""Run OpenClaw's Gateway-backed ACP agent against interception."""

import asyncio
import fcntl
import json
import logging
import secrets
import shlex
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from pydantic import Field

from verifiers.v1.acp import ACP
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness
from verifiers.v1.runtimes import DockerRuntime, ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

# OpenClaw and its bundled Node runtime exceed the small /tmp tmpfs in some VMs.
OPENCLAW_DIR = "/var/tmp/vf-openclaw-{version}"
OPENCLAW_BIN = f"{OPENCLAW_DIR}/bin/openclaw"
INSTALL = r"""
set -e
command -v curl >/dev/null || (apt-get update -qq && apt-get install -y -qq curl ca-certificates >/dev/null)
curl -fsSL --proto '=https' --tlsv1.2 https://openclaw.ai/install-cli.sh | bash -s -- --prefix {dir} --version {version} --no-onboard
"""

OPENCLAW_ACP = ACP()


@asynccontextmanager
async def _gateway_start_lock(runtime: Runtime) -> AsyncIterator[None]:
    if not isinstance(runtime, DockerRuntime) or runtime.network_restricted:
        yield
        return
    # Unrestricted Docker containers share the host network across server workers.
    with Path("/tmp/vf-openclaw-gateway.lock").open("a") as lock:  # noqa: ASYNC230
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                await asyncio.sleep(0.1)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


class OpenClawHarnessConfig(HarnessConfig):
    version: str = Field(default="2026.7.1-2", pattern=r"^[A-Za-z0-9._+-]+$")
    """OpenClaw release to install, pinned for reproducibility."""
    use_bundled_skill: bool = True
    """Enable OpenClaw's bundled skill catalog in addition to uploaded harness skills."""


class OpenClawHarness(Harness[OpenClawHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_RESUME = True
    SUPPORTS_SKILLS = True

    async def setup(self, runtime: Runtime) -> None:
        if not hasattr(self, "_staged_skills_dir"):
            self._staged_skills_dir = (
                f".vf-openclaw/staged-skills-{secrets.token_hex(8)}"
            )
            self._skills_setup_lock = asyncio.Lock()
        if self.config.skills:
            async with self._skills_setup_lock:
                # A complete tree is immutable, so concurrent setups can safely reuse it.
                ready_path = f"{self._staged_skills_dir}/.ready"
                ready = await runtime.run(["test", "-f", ready_path], {})
                if ready.exit_code != 0:
                    cleared = await runtime.run(
                        ["rm", "-rf", self._staged_skills_dir], {}
                    )
                    if cleared.exit_code != 0:
                        raise RuntimeError(
                            "failed to clear OpenClaw skills: "
                            f"{cleared.stderr.strip()[-500:]}"
                        )
                    await self.install_skills(runtime, self._staged_skills_dir)
                    await runtime.write(ready_path, b"")
        directory = OPENCLAW_DIR.format(version=self.config.version)
        binary = OPENCLAW_BIN.format(version=self.config.version)
        script = INSTALL.replace("{version}", self.config.version).replace(
            "{dir}", directory
        )
        ensure = shlex.quote(f"[ -x {binary} ] || ({script})")
        guarded = (
            f"mkdir -p {directory} && "
            f'"$(command -v flock || command -v lockf)" {directory}/install.lock '
            f"bash -o pipefail -c {ensure}"
        )
        logger.info("openclaw: ensuring OpenClaw %s is installed", self.config.version)
        result = await runtime.run(["sh", "-c", guarded], self.config.resolved_env)
        if result.exit_code != 0:
            detail = (result.stderr or result.stdout).strip()[-500:]
            raise RuntimeError(f"OpenClaw install failed: {detail}")
        await OPENCLAW_ACP.setup(self, runtime)

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
        system_prompt, prompt = self.resolve_prompt(data)
        reasoning = ctx.sampling.reasoning_effort not in (
            None,
            "none",
        ) or ctx.model.rsplit("/", 1)[-1].startswith(("gpt-5", "o1", "o3", "o4"))
        directory = OPENCLAW_DIR.format(version=self.config.version)
        state_dir = f".vf-openclaw/{trace.id}"
        config_path = f"{state_dir}/openclaw.json"
        skills_dir = f"{state_dir}/skills"
        config = {
            "gateway": {"mode": "local"},
            "agents": {
                "defaults": {
                    "workspace": ".",
                    "skipBootstrap": True,
                    "sandbox": {"mode": "off"},
                    "model": {"primary": f"intercept/{ctx.model}"},
                }
            },
            "tools": {
                "profile": "coding",
                "fs": {"workspaceOnly": True},
                "exec": {"host": "gateway", "mode": "full"},
                "deny": self.config.disabled_tools or [],
            },
            "models": {
                "mode": "replace",
                "providers": {
                    "intercept": {
                        "baseUrl": endpoint,
                        "apiKey": "${OPENCLAW_INTERCEPT_KEY}",
                        "api": "openai-responses",
                        "authHeader": True,
                        "models": [
                            {
                                "id": ctx.model,
                                "name": ctx.model,
                                "reasoning": reasoning,
                                "input": ["text", "image"],
                            }
                        ],
                    }
                },
            },
            "mcp": {
                "servers": {
                    name: {
                        "url": url,
                        "transport": "streamable-http",
                        "connectionTimeoutMs": 60_000,
                        "requestTimeoutMs": int(self.config.tool_timeout * 1000),
                    }
                    for name, url in mcp_urls.items()
                }
            },
        }
        created = await runtime.run(["mkdir", "-p", state_dir], {})
        if created.exit_code != 0:
            raise RuntimeError(
                f"OpenClaw state directory failed: {created.stderr.strip()[-500:]}"
            )
        if self.config.skills:
            exists = await runtime.run(["test", "-d", skills_dir], {})
            if exists.exit_code != 0:
                copied = await runtime.run(
                    ["cp", "-R", self._staged_skills_dir, skills_dir], {}
                )
                if copied.exit_code != 0:
                    raise RuntimeError(
                        f"failed to stage OpenClaw skills: {copied.stderr.strip()[-500:]}"
                    )
            config["skills"] = {"load": {"extraDirs": [skills_dir]}}
        if not self.config.use_bundled_skill:
            # OpenClaw treats an empty allowlist as all; a no-match key disables the catalog.
            config.setdefault("skills", {})["allowBundled"] = ["__none__"]
        await runtime.write(config_path, json.dumps(config).encode())

        binary = OPENCLAW_BIN.format(version=self.config.version)
        node = f"{directory}/tools/node/bin/node"
        allocate_port = shlex.join(
            [
                node,
                "-e",
                (
                    'const net=require("node:net");const server=net.createServer();'
                    'server.listen(0,"127.0.0.1",()=>{'
                    "process.stdout.write(String(server.address().port));server.close();});"
                ),
            ]
        )
        log_path = f"{state_dir}/gateway.log"
        pid_path = f"{state_dir}/gateway.pid"
        env = {
            **self.config.resolved_env,
            "OPENCLAW_CONFIG_PATH": config_path,
            "OPENCLAW_STATE_DIR": state_dir,
            "OPENCLAW_INTERCEPT_KEY": secret,
            "OPENCLAW_HIDE_BANNER": "1",
            "OPENCLAW_SUPPRESS_NOTES": "1",
            "NO_COLOR": "1",
        }
        async with _gateway_start_lock(runtime):
            allocated = await runtime.run(["sh", "-c", allocate_port], {})
            if allocated.exit_code != 0:
                raise RuntimeError(
                    f"OpenClaw port allocation failed: {allocated.stderr.strip()[-500:]}"
                )
            gateway_port = allocated.stdout.strip()
            port_probe = shlex.join(
                [
                    node,
                    "-e",
                    (
                        'const net=require("node:net");'
                        f'const socket=net.connect({{host:"127.0.0.1",port:{gateway_port}}});'
                        'socket.on("connect",()=>socket.end());'
                        'socket.on("error",()=>process.exit(1));'
                    ),
                ]
            )
            await runtime.run_background(
                [
                    "sh",
                    "-c",
                    (
                        f"echo $$ >{shlex.quote(pid_path)}; "
                        f"exec {shlex.quote(binary)} gateway run "
                        f"--port {shlex.quote(gateway_port)} --bind loopback "
                        f"--auth token --token {shlex.quote(trace.id)} "
                        "--allow-unconfigured"
                    ),
                ],
                env,
                log_path,
            )
            bound = await runtime.run(
                [
                    "sh",
                    "-c",
                    (
                        "attempt=0; "
                        f"until [ -s {shlex.quote(pid_path)} ] && "
                        f'kill -0 "$(cat {shlex.quote(pid_path)})" 2>/dev/null && '
                        f"{port_probe} >/dev/null 2>&1; do "
                        f"if [ -s {shlex.quote(pid_path)} ] && "
                        f'! kill -0 "$(cat {shlex.quote(pid_path)})" 2>/dev/null; then '
                        f"tail -100 {shlex.quote(log_path)} >&2; exit 1; fi; "
                        "attempt=$((attempt + 1)); "
                        f'[ "$attempt" -lt 1200 ] || {{ tail -100 {shlex.quote(log_path)} >&2; exit 1; }}; '
                        "sleep 0.1; done"
                    ),
                ],
                {},
            )
            if bound.exit_code != 0:
                detail = (bound.stderr or bound.stdout).strip()[-2000:]
                raise RuntimeError(f"OpenClaw gateway failed to bind: {detail}")
        readiness = await runtime.run(
            [
                "sh",
                "-c",
                (
                    "attempt=0; "
                    f"until [ -s {shlex.quote(pid_path)} ] && "
                    f'kill -0 "$(cat {shlex.quote(pid_path)})" 2>/dev/null && '
                    f'curl -fsS "http://127.0.0.1:{gateway_port}/healthz" '
                    ">/dev/null 2>&1; do "
                    f"if [ -s {shlex.quote(pid_path)} ] && "
                    f'! kill -0 "$(cat {shlex.quote(pid_path)})" 2>/dev/null; then '
                    f"tail -100 {shlex.quote(log_path)} >&2; exit 1; fi; "
                    "attempt=$((attempt + 1)); "
                    f'[ "$attempt" -lt 120 ] || {{ tail -100 {shlex.quote(log_path)} >&2; exit 1; }}; '
                    "sleep 1; done"
                ),
            ],
            {},
        )
        if readiness.exit_code != 0:
            detail = (readiness.stderr or readiness.stdout).strip()[-2000:]
            raise RuntimeError(f"OpenClaw gateway failed to start: {detail}")
        command = [
            "sh",
            "-c",
            (
                "set -eu; "
                f"gateway_pid=$(cat {shlex.quote(pid_path)}); "
                f'trap \'kill "$gateway_pid" 2>/dev/null || true; '
                'attempt=0; while kill -0 "$gateway_pid" 2>/dev/null && '
                ' [ "$attempt" -lt 50 ]; do attempt=$((attempt + 1)); sleep 0.1; done; '
                'kill -9 "$gateway_pid" 2>/dev/null || true; '
                'while kill -0 "$gateway_pid" 2>/dev/null; do sleep 0.1; done; '
                f"rm -f {shlex.quote(pid_path)}' EXIT; "
                f'{shlex.quote(binary)} acp --url "ws://127.0.0.1:{gateway_port}" '
                f"--token {shlex.quote(trace.id)} "
                f"--session {shlex.quote(f'agent:main:acp-bridge:{trace.id}')} "
                "--no-prefix-cwd"
            ),
        ]
        # OpenClaw rejects ACP per-session MCP declarations; the isolated Gateway
        # config above owns the equivalent task-scoped server definitions.
        return await OPENCLAW_ACP.run(
            runtime,
            env,
            command,
            prompt,
            mcp_urls={},
            system_prompt=system_prompt,
            session_path=f"{state_dir}/acp-session",
            # OpenClaw can end after its final tool completes without a text message.
            allow_empty_tool_reply=True,
        )

    async def cleanup(self, trace: Trace, runtime: Runtime) -> None:
        state_dir = f".vf-openclaw/{trace.id}"
        pid_path = f"{state_dir}/gateway.pid"
        result = await runtime.run(
            [
                "sh",
                "-c",
                (
                    f"if [ -s {shlex.quote(pid_path)} ]; then "
                    f"gateway_pid=$(cat {shlex.quote(pid_path)}); "
                    'kill "$gateway_pid" 2>/dev/null || true; '
                    'attempt=0; while kill -0 "$gateway_pid" 2>/dev/null && '
                    ' [ "$attempt" -lt 50 ]; do attempt=$((attempt + 1)); sleep 0.1; done; '
                    'kill -9 "$gateway_pid" 2>/dev/null || true; '
                    'while kill -0 "$gateway_pid" 2>/dev/null; do sleep 0.1; done; fi; '
                    f"rm -rf {shlex.quote(state_dir)}"
                ),
            ],
            {},
        )
        if result.exit_code != 0:
            raise RuntimeError(
                f"failed to clean up OpenClaw state: {result.stderr.strip()[-500:]}"
            )
