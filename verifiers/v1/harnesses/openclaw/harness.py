"""Run OpenClaw's Gateway-backed ACP agent against interception."""

import asyncio
import json
import logging
import secrets

from verifiers.v1.acp import ACPConfig, ACPHarness
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig, PinnedVersion
from verifiers.v1.harnesses.utils.install import ensure_installed, remove_dir
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

# OpenClaw and its bundled Node runtime exceed the small /tmp tmpfs in some VMs.
OPENCLAW_DIR = "/var/tmp/vf-openclaw-{version}"
OPENCLAW_BIN = f"{OPENCLAW_DIR}/bin/openclaw"
SETUP = r"""
set -e
if [ ! -x "$VF_OPENCLAW_BIN" ]; then
    command -v curl >/dev/null || (apt-get update -qq && apt-get install -y -qq curl ca-certificates >/dev/null)
    curl -fsSL --proto '=https' --tlsv1.2 https://openclaw.ai/install-cli.sh \
        | bash -s -- --prefix "$VF_OPENCLAW_DIR" --version "$VF_OPENCLAW_VERSION" --no-onboard
fi
set -- $(find "$VF_OPENCLAW_DIR/tools" -type f -path '*/openclaw/dist/session-accessor.sqlite-transcript-store-*.js')
[ "$#" -eq 1 ] || { echo "could not locate OpenClaw's transcript sanitizer for replay patch" >&2; exit 1; }
"$VF_OPENCLAW_DIR/tools/node/bin/node" - "$1" <<'NODE'
const fs = require("node:fs");
const file = process.argv[2];
const stat = fs.statSync(file);
if (stat.size > 1_000_000) {
    throw new Error("OpenClaw's transcript sanitizer exceeds the replay patch size limit");
}
const source = fs.readFileSync(file, "utf8");
const patches = [
    [
        "const OPAQUE_REPLAY_TOKEN_RE = /^[A-Za-z0-9+/_-]+={0,2}$/;",
        "const OPAQUE_REPLAY_TOKEN_RE = /^[A-Za-z0-9+/_-]+={0,2}(?:[.][A-Za-z0-9+/_-]+={0,2})*$/;",
    ],
    ["\t\tsummary: [],", "\t\tsummary: parsed.summary ?? [],"],
];
let patched = source;
for (const [before, after] of patches) {
    patched = patched.replaceAll(before, after);
}
if (!patches.every(([, after]) => patched.includes(after))) {
    throw new Error("OpenClaw's transcript sanitizer does not match the replay patch");
}
if (patched !== source) {
    const temporary = `${file}.${process.pid}.tmp`;
    try {
        // Publish complete bytes at once so live and concurrent setups never see a partial bundle.
        fs.writeFileSync(temporary, patched, { mode: stat.mode });
        fs.renameSync(temporary, file);
    } finally {
        fs.rmSync(temporary, { force: true });
    }
}
NODE
"""

OPENCLAW_COMMAND = [
    "sh",
    "-c",
    r"""
set -eu
exec 3<&0
gateway_pid=
acp_pid=
cleanup() {
    trap - EXIT HUP INT TERM
    [ -z "$acp_pid" ] || kill -TERM -"$acp_pid" 2>/dev/null || true
    [ -z "$gateway_pid" ] || kill -TERM -"$gateway_pid" 2>/dev/null || true
    [ -z "$acp_pid$gateway_pid" ] || sleep 1
    [ -z "$acp_pid" ] || kill -KILL -"$acp_pid" 2>/dev/null || true
    [ -z "$gateway_pid" ] || kill -KILL -"$gateway_pid" 2>/dev/null || true
    [ -z "$acp_pid" ] || wait "$acp_pid" 2>/dev/null || true
    [ -z "$gateway_pid" ] || wait "$gateway_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 143' HUP INT TERM
root=${VF_OPENCLAW_BIN%/bin/openclaw}
gateway_attempt=0
while [ "$gateway_attempt" -lt 5 ]; do
    port=$("$root/tools/node/bin/node" -e 'const net=require("node:net");const server=net.createServer();server.listen(0,"127.0.0.1",()=>{process.stdout.write(String(server.address().port));server.close();});')
    export OPENCLAW_GATEWAY_PORT="$port"
    # OpenClaw respawns Node; separate groups let the trap reap both process trees.
    setsid "$VF_OPENCLAW_BIN" gateway run </dev/null >"$OPENCLAW_STATE_DIR/gateway.log" &
    gateway_pid=$!
    attempt=0
    while ! curl -fsS "http://127.0.0.1:$port/readyz" >/dev/null 2>&1; do
        if ! kill -0 "$gateway_pid" 2>/dev/null; then
            wait "$gateway_pid" 2>/dev/null || true
            gateway_pid=
            break
        fi
        attempt=$((attempt + 1))
        [ "$attempt" -lt 120 ] || { tail -100 "$OPENCLAW_STATE_DIR/gateway.log" >&2; exit 1; }
        sleep 1
    done
    [ -n "$gateway_pid" ] && break
    # The probe releases its socket before Gateway binds, so retry early exits
    # when concurrent host-network rollouts select the same port.
    gateway_attempt=$((gateway_attempt + 1))
done
if [ -z "$gateway_pid" ]; then
    tail -100 "$OPENCLAW_STATE_DIR/gateway.log" >&2
    exit 1
fi
setsid "$VF_OPENCLAW_BIN" acp --verbose --session "agent:main:acp-bridge:$OPENCLAW_GATEWAY_TOKEN" --no-prefix-cwd <&3 >&1 &
acp_pid=$!
wait "$acp_pid"
""",
]


class OpenClawHarnessConfig(HarnessConfig):
    version: PinnedVersion = "2026.8.1"
    """OpenClaw release to install, pinned for reproducibility."""
    use_bundled_skill: bool = True
    """Enable OpenClaw's bundled skill catalog in addition to uploaded harness skills."""


class OpenClawHarness(ACPHarness[OpenClawHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
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
                    await remove_dir(
                        runtime, self._staged_skills_dir, "OpenClaw skills"
                    )
                    await self.install_skills(runtime, self._staged_skills_dir)
                    await runtime.write(ready_path, b"")
        directory = OPENCLAW_DIR.format(version=self.config.version)
        binary = OPENCLAW_BIN.format(version=self.config.version)
        logger.info("openclaw: ensuring OpenClaw %s is installed", self.config.version)
        # Borrowed runtimes can share this cache, so one filesystem lock owns both
        # installation and the transcript adjustment.
        await ensure_installed(
            runtime,
            directory=directory,
            install=SETUP,
            env={
                **self.config.resolved_env,
                "VF_OPENCLAW_BIN": binary,
                "VF_OPENCLAW_DIR": directory,
                "VF_OPENCLAW_VERSION": self.config.version,
            },
            label="OpenClaw",
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
        mcp_urls: dict[str, str],
        data: TaskData,
    ) -> ACPConfig:
        system_prompt, prompt = self.resolve_prompt(data)
        provider, _, _ = ctx.model.partition("/")
        state_dir = f".vf-openclaw/{trace.id}"
        config_path = f"{state_dir}/openclaw.json"
        skills_dir = f"{state_dir}/skills"
        config = {
            "gateway": {
                "mode": "local",
                "bind": "loopback",
                "auth": {
                    "mode": "token",
                    "token": "${OPENCLAW_GATEWAY_TOKEN}",
                },
            },
            "messages": {"queue": {"mode": "interrupt"}},
            "agents": {
                "defaults": {
                    "workspace": ".",
                    "skipBootstrap": True,
                    "heartbeat": {"every": "0m"},
                    "sandbox": {"mode": "off"},
                    "model": {"primary": ctx.model},
                }
            },
            "tools": {
                "profile": "coding",
                "fs": {"workspaceOnly": True},
                "exec": {"host": "gateway", "mode": "full"},
                "deny": self.config.disabled_tools or [],
            },
            "models": {
                "providers": {
                    provider: {
                        "baseUrl": endpoint,
                        "apiKey": "${OPENCLAW_INTERCEPT_KEY}",
                        "request": {"allowPrivateNetwork": True},
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
        if self.config.skills:
            config["skills"] = {"load": {"extraDirs": [skills_dir]}}
        if not self.config.use_bundled_skill:
            # OpenClaw treats an empty allowlist as all; a no-match key disables the catalog.
            config.setdefault("skills", {})["allowBundled"] = ["__none__"]
        await runtime.write(config_path, json.dumps(config).encode())
        if self.config.skills:
            copied = await runtime.run(
                ["cp", "-R", self._staged_skills_dir, skills_dir], {}
            )
            if copied.exit_code != 0:
                raise RuntimeError(
                    f"failed to isolate OpenClaw skills: {copied.stderr.strip()[-500:]}"
                )

        env = {
            **self.config.resolved_env,
            "OPENCLAW_CONFIG_PATH": config_path,
            "OPENCLAW_STATE_DIR": state_dir,
            "OPENCLAW_GATEWAY_TOKEN": trace.id,
            "OPENCLAW_INTERCEPT_KEY": secret,
            "OPENCLAW_HIDE_BANNER": "1",
            "OPENCLAW_SUPPRESS_NOTES": "1",
            "NO_COLOR": "1",
            "VF_OPENCLAW_BIN": OPENCLAW_BIN.format(version=self.config.version),
        }
        # OpenClaw rejects ACP per-session MCP declarations; the isolated Gateway
        # config owns the equivalent task-scoped server definitions.
        return ACPConfig(
            env=env,
            command=OPENCLAW_COMMAND,
            prompt=prompt,
            mcp_urls={},
            system_prompt=system_prompt,
        )

    async def cleanup(self, trace: Trace, runtime: Runtime) -> None:
        state_dir = f".vf-openclaw/{trace.id}"
        await remove_dir(runtime, state_dir, "OpenClaw state")
