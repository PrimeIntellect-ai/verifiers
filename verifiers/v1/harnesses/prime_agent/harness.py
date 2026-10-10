"""Prime Agent over its native ACP mode."""

import hashlib
import json
import logging
from typing import Literal, NamedTuple

from verifiers.v1.acp import ACPConfig, ACPHarness, ACPTurn
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig, skill_destination
from verifiers.v1.harnesses.node import NODE_BIN_DIR, ensure_node
from verifiers.v1.harnesses.utils.install import ensure_installed, remove_dir
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

GITHUB_RELEASE_URL = (
    "https://github.com/PrimeIntellect-ai/prime-agent/releases/download"
)
# Two release shapes: `0.9.5` is the npm-era TS bundle (four npm tarballs,
# installed by commit), and from the Rust rewrite on, releases are one platform
# tarball per OS (the `prime-agent` binary, the `prime-agent-runtime/` kernel
# sidecar, `models.bundled.json`, `mcp-services.bundled.json`, and the built-in
# skills, all flat at the archive root).
TS_VERSION = "0.9.5"
TS_COMMIT: Literal["a7d791bc1be09793ed5f3ec05bf4cccbc60679ea"] = (
    "a7d791bc1be09793ed5f3ec05bf4cccbc60679ea"
)
RUST_VERSION = "0.10.0"
# The v0.10.0 release SHA256SUMS rows for the published tarballs.
RUST_RELEASE_SHA256: dict[str, str] = {
    "linux-x64": "c16bd2af5e77b53f49b914a44742c4cf6a67c5ed5000041430b78dbd4c3bcbec",
    "linux-arm64": "f3cab3530a4d7ca43dbef8321bf13f260d1ba05d8b5dde057165a20bd4f675c4",
    "darwin-x64": "af4866b5ba82f3419b964290e3f023ddb9b99958a89e16ee856962b901ec0fc6",
    "darwin-arm64": "e418bdf62fcb0002777bf3ac43bcc136512b26763f2ab5c500792f26e37294e5",
}
PRIME_AGENT_DIR = "/var/tmp/vf-prime-agent"
STATE_ROOT = "/tmp/vf-prime-agent-runs"
PROVIDER = "intercept"
LIFECYCLE_META_NAMESPACE = "ai.primeintellect.prime-agent"
KEY_VAR = "PRIME_AGENT_INTERCEPT_KEY"
ENV_AGENT_DIR = "PRIME_AGENT_CODING_AGENT_DIR"


TS_INSTALL = r"""
set -e
export PATH="/var/tmp/vf-node/bin:$PATH"
prefix="$VF_PRIME_AGENT_DIR/$PRIME_AGENT_COMMIT"
[ -x "$prefix/bin/prime-agent" ] && [ -f "$HOME/.prime/agent/kernel-venv/.bootstrap-version" ] && exit 0
export NPM_CONFIG_PREFIX="$prefix"
export PRIME_AGENT_BOOTSTRAP_KERNEL_ON_INSTALL=1
release_url="$VF_PRIME_AGENT_GITHUB_RELEASE_URL/v$PRIME_AGENT_RELEASE_VERSION"
agent_tarball="prime-agent-$PRIME_AGENT_RELEASE_VERSION.tgz"
ai_tarball="prime-agent-ai-$PRIME_AGENT_RELEASE_VERSION.tgz"
core_tarball="prime-agent-core-$PRIME_AGENT_RELEASE_VERSION.tgz"
tui_tarball="prime-agent-tui-$PRIME_AGENT_RELEASE_VERSION.tgz"
download_dir="$(mktemp -d "$VF_PRIME_AGENT_DIR/install.XXXXXX")"
trap 'rm -rf "$download_dir"' EXIT
for tarball in "$agent_tarball" "$ai_tarball" "$core_tarball" "$tui_tarball"; do
    curl -fsSL --retry 5 --retry-all-errors \
        "$release_url/$tarball" -o "$download_dir/$tarball"
done
printf '%s  %s\n' \
    '349f1682c7909550842f1b04a71ba95814341b136474ade736df93f8ec006876' "$agent_tarball" \
    '9ad0184b7b5f3d5c3b1677c0663335de32c092cbdb72697aeaf997f17910bb92' "$ai_tarball" \
    '59aafeffb4b64eb997399d56a03f529b2f6604b228e5674372195277db52f04a' "$core_tarball" \
    'e49f41170edfd1d0e72418729d2c117969ccf8df784dc55fe8c1a9d91778e103' "$tui_tarball" \
    > "$download_dir/SHA256SUMS"
(cd "$download_dir" && sha256sum -c SHA256SUMS)
mkdir "$download_dir/package-root"
tar -xzf "$download_dir/$agent_tarball" -C "$download_dir/package-root"
node - \
    "$download_dir/package-root/package/package.json" \
    "$download_dir" \
    "$ai_tarball" \
    "$core_tarball" \
    "$tui_tarball" <<'NODE'
const fs = require("node:fs");
const [manifestPath, downloadDir, ai, core, tui] = process.argv.slice(2);
const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
for (const [name, file] of [
    ["@earendil-works/pi-ai", ai],
    ["@earendil-works/pi-agent-core", core],
    ["@earendil-works/pi-tui", tui],
]) {
    manifest.dependencies[name] = `file:${downloadDir}/${file}`;
}
fs.writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
NODE
mkdir "$download_dir/repacked"
repacked="$(npm pack "$download_dir/package-root/package" \
    --pack-destination "$download_dir/repacked" --silent)"
PRIME_AGENT_BOOTSTRAP_TOOLS_ON_INSTALL=1 npm install -g \
    --no-fund --no-audit --loglevel=error --progress=false \
    "$download_dir/repacked/$repacked"
[ -f "$HOME/.prime/agent/kernel-venv/.bootstrap-version" ]
"""

RUST_INSTALL = r"""
set -e
prefix="$VF_PRIME_AGENT_DIR/$PRIME_AGENT_RELEASE_VERSION"
[ -x "$prefix/prime-agent" ] && [ -f "$HOME/.prime/agent/kernel-venv/.bootstrap-version" ] && exit 0
release_url="$VF_PRIME_AGENT_GITHUB_RELEASE_URL/v$PRIME_AGENT_RELEASE_VERSION"
tarball="prime-agent-$PRIME_AGENT_RELEASE_VERSION-$VF_PRIME_AGENT_PLATFORM.tar.gz"
download_dir="$(mktemp -d "$VF_PRIME_AGENT_DIR/install.XXXXXX")"
trap 'rm -rf "$download_dir"' EXIT
# Slim task images may not carry curl; the download needs it.
if ! command -v curl >/dev/null 2>&1; then
    (apt-get update -qq && apt-get install -y -qq curl ca-certificates >/dev/null) \
        || apk add --no-cache curl ca-certificates >/dev/null \
        || { echo "prime-agent install needs curl" >&2; exit 1; }
fi
curl -fsSL --retry 5 --retry-all-errors \
    "$release_url/$tarball" -o "$download_dir/$tarball"
printf '%s  %s\n' "$VF_PRIME_AGENT_RELEASE_SHA256" "$tarball" \
    > "$download_dir/SHA256SUMS"
(cd "$download_dir" && sha256sum -c SHA256SUMS)
mkdir -p "$prefix"
tar -xzf "$download_dir/$tarball" -C "$prefix"
chmod +x "$prefix/prime-agent"
# The kernel bootstrap provisions the venv's Python and packages through uv;
# the binary looks for uv on PATH and at ~/.local/bin/uv.
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
    curl -fsSL --retry 5 --retry-all-errors \
        https://astral.sh/uv/install.sh -o "$download_dir/uv-install.sh"
    UV_INSTALL_DIR="$HOME/.local/bin" sh "$download_dir/uv-install.sh" >/dev/null
    [ -x "$HOME/.local/bin/uv" ]
fi
# The release installer's own pre-warm: run the install-time bootstrap entry
# so the first session starts with the kernel venv already built.
PATH="$HOME/.local/bin:$PATH" "$prefix/prime-agent" --prime-agent-bootstrap
[ -f "$HOME/.prime/agent/kernel-venv/.bootstrap-version" ]
"""


class ReleasePlan(NamedTuple):
    """How one Prime Agent release is installed and launched."""

    install: str
    bin: str
    launch_path: str


def release_plan(version: str, platform: str) -> ReleasePlan:
    """Choose the install script, binary path, and launch PATH dirs for one
    release version: the 0.9.x npm era keeps the TS bundle, the Rust rewrite's
    versions install the platform tarball."""
    if version == TS_VERSION:
        return ReleasePlan(
            TS_INSTALL,
            f"{PRIME_AGENT_DIR}/{TS_COMMIT}/bin/prime-agent",
            f"{NODE_BIN_DIR}:$HOME/.local/bin",
        )
    if version == RUST_VERSION:
        if platform not in RUST_RELEASE_SHA256:
            raise ValueError(
                f"prime-agent {version} has no pinned {platform} tarball "
                f"(pinned: {', '.join(sorted(RUST_RELEASE_SHA256))})"
            )
        return ReleasePlan(
            RUST_INSTALL,
            f"{PRIME_AGENT_DIR}/{version}/prime-agent",
            "$HOME/.local/bin",
        )
    raise ValueError(
        f"prime-agent {version} has no pinned install "
        f"(known: {TS_VERSION} npm era, {RUST_VERSION} Rust tarball)"
    )


class PrimeAgentHarnessConfig(HarnessConfig):
    version: Literal["0.9.5", "0.10.0"] = RUST_VERSION
    """Prime Agent release to install: the Rust platform tarball, or `0.9.5`
    for the npm-era TS bundle."""

    platform: Literal["linux-x64", "linux-arm64", "darwin-x64", "darwin-arm64"] = (
        "linux-x64"
    )
    """Platform suffix of the Rust release tarball to install."""

    autonomous: bool = False
    """Enable Prime Agent's autonomous continuation loop."""


class PrimeAgentHarness(ACPHarness[PrimeAgentHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_SKILLS = True

    def acp_turn_result(self, trace: Trace, result: ACPTurn) -> None:
        events = [
            event
            for metadata in result.update_metadata
            if isinstance(event := metadata.get(LIFECYCLE_META_NAMESPACE), dict)
        ]
        terminal = next(
            (
                event
                for event in reversed(events)
                if event.get("phase") == "terminalQuiescence"
            ),
            None,
        )
        prompt_turn_id = terminal.get("promptTurnId") if terminal else None
        boundary = next(
            (
                event
                for event in reversed(events)
                if event.get("phase") == "responseBoundary"
                and event.get("promptTurnId") == prompt_turn_id
            ),
            None,
        )
        quiescence = terminal.get("quiescence") if terminal else None
        quiescent = bool(
            isinstance(quiescence, dict) and quiescence.get("outstandingSubagents") == 0
        )
        status = {
            "prompt_turn_id": prompt_turn_id,
            "stop_reason": result.stop_reason,
            "infrastructure_status": "ok" if boundary and quiescent else "unverified",
            "autonomous_completion": bool(
                quiescent and terminal and terminal.get("outcome") == "result"
            ),
            "terminal_quiescence_observed": terminal is not None,
            "last_lifecycle_phase": terminal.get("phase")
            if terminal
            else boundary.get("phase")
            if boundary
            else None,
            "response_boundary": boundary,
            "terminal_quiescence": terminal,
        }
        lifecycle = trace.info.get("acp_lifecycle")
        if not isinstance(lifecycle, dict):
            lifecycle = {}
            trace.info["acp_lifecycle"] = lifecycle
        statuses = lifecycle.get(LIFECYCLE_META_NAMESPACE)
        if not isinstance(statuses, list):
            statuses = []
            lifecycle[LIFECYCLE_META_NAMESPACE] = statuses
        statuses.append(status)

    async def setup(self, runtime: Runtime) -> None:
        plan = release_plan(self.config.version, self.config.platform)
        if self.config.version == TS_VERSION:
            await ensure_node(runtime)
        logger.info(
            "prime-agent: ensuring release %s (%s) is installed",
            self.config.version,
            self.config.platform,
        )
        await ensure_installed(
            runtime,
            directory=PRIME_AGENT_DIR,
            install=plan.install,
            env={
                **self.config.resolved_env,
                "VF_PRIME_AGENT_DIR": PRIME_AGENT_DIR,
                "VF_PRIME_AGENT_GITHUB_RELEASE_URL": GITHUB_RELEASE_URL,
                "PRIME_AGENT_COMMIT": TS_COMMIT,
                "PRIME_AGENT_RELEASE_VERSION": self.config.version,
                "VF_PRIME_AGENT_PLATFORM": self.config.platform,
                "VF_PRIME_AGENT_RELEASE_SHA256": RUST_RELEASE_SHA256.get(
                    self.config.platform, ""
                ),
            },
            label="prime-agent",
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
            raise ValueError(
                "prime-agent has no per-tool disable flag; its model-facing tool "
                "surface is ipython"
            )

        root = self._root(trace)
        agent_dir = f"{root}/agent"
        skills_dir = f"{agent_dir}/skills"
        created = await runtime.run(
            [
                "mkdir",
                "-p",
                "-m",
                "700",
                root,
                agent_dir,
                f"{root}/tmp",
            ],
            {},
        )
        if created.exit_code != 0:
            raise RuntimeError(
                f"prime-agent state directory failed: {created.stderr.strip()[-500:]}"
            )
        await self.install_skills(runtime, skills_dir)
        reasoning = ctx.sampling.reasoning_effort not in (
            None,
            "none",
        ) or ctx.model.rsplit("/", 1)[-1].startswith(("gpt-5", "o1", "o3", "o4"))
        models = {
            "providers": {
                PROVIDER: {
                    "baseUrl": endpoint,
                    "api": "openai-completions",
                    "apiKey": KEY_VAR,
                    "models": [
                        {
                            "id": ctx.model,
                            "reasoning": reasoning,
                            "input": ["text", "image"],
                        }
                    ],
                }
            }
        }
        models_path = f"{agent_dir}/models.json"
        await runtime.write(models_path, json.dumps(models).encode())
        secured = await runtime.run(["chmod", "600", models_path], {})
        if secured.exit_code != 0:
            raise RuntimeError(
                f"prime-agent model config chmod failed: {secured.stderr.strip()[-500:]}"
            )

        system_prompt, prompt = self.resolve_prompt(data)
        plan = release_plan(self.config.version, self.config.platform)
        args = [
            plan.bin,
            "--mode",
            "acp",
            "--provider",
            PROVIDER,
            "--model",
            ctx.model,
            "--daemon-socket",
            f"{root}/daemon.sock",
            "--offline",
        ]
        if self.config.autonomous:
            args.append("--autonomous")
        for skill in self.config.skills:
            args += ["--skill", skill_destination(skill, skills_dir)]
        if system_prompt:
            args += ["--append-system-prompt", system_prompt]

        return ACPConfig(
            env=self._env(trace, secret),
            # Expand the sandbox's PATH while keeping every agent argument literal.
            command=[
                "/bin/sh",
                "-eu",
                "-c",
                f'export PATH="{plan.launch_path}:$PATH"; exec "$@"',
                "prime-agent",
                *args,
            ],
            prompt=prompt,
        )

    async def cleanup(self, trace: Trace, runtime: Runtime) -> None:
        root = self._root(trace)
        await remove_dir(runtime, root, "prime-agent state")

    @staticmethod
    def _root(trace: Trace) -> str:
        digest = hashlib.sha256(trace.id.encode()).hexdigest()[:16]
        return f"{STATE_ROOT}/{digest}"

    def _env(self, trace: Trace, secret: str) -> dict[str, str]:
        root = self._root(trace)
        return {
            **self.config.resolved_env,
            KEY_VAR: secret,
            ENV_AGENT_DIR: f"{root}/agent",
            "TMPDIR": f"{root}/tmp",
            "PRIME_AGENT_TELEMETRY": "0",
        }
