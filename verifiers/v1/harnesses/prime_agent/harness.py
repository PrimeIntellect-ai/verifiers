"""Prime Agent over its native ACP mode."""

import hashlib
import json
import logging
import re
import shlex
from dataclasses import dataclass

import httpx
from pydantic import field_validator, model_validator

from verifiers.v1.acp import ACPConfig, ACPHarness, ACPTurn
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harnesses.node import NODE_BIN_DIR, ensure_node
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

GITHUB_RELEASE_URL = (
    "https://github.com/PrimeIntellect-ai/prime-agent/releases/download"
)
LATEST_RELEASE_URL = "https://pub-728493de92a943e2a9b2d17b4719f318.r2.dev/latest.json"
MINIMUM_VERSION = (0, 8, 1)
RELEASE_PACKAGES = (
    "prime-agent",
    "prime-agent-ai",
    "prime-agent-core",
    "prime-agent-tui",
)
PRIME_AGENT_DIR = "/var/tmp/vf-prime-agent"
STATE_ROOT = "/tmp/vf-prime-agent-runs"
SKILLS_DIR = ".agents/skills"
PROVIDER = "intercept"
LIFECYCLE_META_NAMESPACE = "ai.primeintellect.prime-agent"
KEY_VAR = "PRIME_AGENT_INTERCEPT_KEY"
ENV_AGENT_DIR = "PRIME_AGENT_CODING_AGENT_DIR"


@dataclass(frozen=True)
class PrimeAgentRelease:
    version: str
    sha256: dict[str, str]

    @property
    def cache_key(self) -> str:
        manifest = json.dumps(
            {"version": self.version, "sha256": self.sha256},
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(manifest.encode()).hexdigest()[:16]
        return f"{self.version}-{digest}"

    def file(self, package: str) -> str:
        return f"{package}-{self.version}.tgz"


def _version(value: object) -> str:
    raw = value.removeprefix("v") if isinstance(value, str) else ""
    if not re.fullmatch(r"\d+\.\d+\.\d+", raw):
        raise ValueError(f"invalid Prime Agent stable version: {value!r}")
    version = tuple(int(part) for part in raw.split("."))
    if raw != ".".join(str(part) for part in version):
        raise ValueError(f"invalid Prime Agent stable version: {value!r}")
    if version < MINIMUM_VERSION:
        minimum = ".".join(str(part) for part in MINIMUM_VERSION)
        raise ValueError(f"Prime Agent {raw} is older than the required {minimum}")
    return raw


def _release(version: object, entries: object) -> PrimeAgentRelease:
    resolved = _version(version)
    if not isinstance(entries, list):
        raise TypeError("Prime Agent release metadata has no tarballs")
    files: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError("invalid Prime Agent tarball entry")
        file = entry.get("file")
        sha256 = entry.get("sha256")
        if isinstance(file, str) and isinstance(sha256, str):
            if file in files:
                raise ValueError(f"duplicate Prime Agent tarball {file!r}")
            files[file] = sha256.lower()
    hashes: dict[str, str] = {}
    for package in RELEASE_PACKAGES:
        file = f"{package}-{resolved}.tgz"
        sha256 = files.get(file, "")
        if not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise ValueError(f"missing or invalid SHA-256 for {file}")
        hashes[package] = sha256
    return PrimeAgentRelease(version=resolved, sha256=hashes)


def _source(requested: str) -> str:
    if requested == "stable":
        return LATEST_RELEASE_URL
    return f"{GITHUB_RELEASE_URL}/v{requested}/SHA256SUMS"


async def _fetch_release(requested: str) -> PrimeAgentRelease:
    source = _source(requested)
    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        response = await client.get(source)
        response.raise_for_status()
        if requested == "stable":
            try:
                metadata = response.json()
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"invalid Prime Agent release metadata from {source}"
                ) from e
            if not isinstance(metadata, dict):
                raise ValueError(f"invalid Prime Agent release metadata from {source}")
            return _release(metadata.get("version"), metadata.get("tarballs"))
        entries = []
        for line in response.text.splitlines():
            fields = line.split()
            if len(fields) == 2:
                entries.append({"sha256": fields[0], "file": fields[1]})
        return _release(requested, entries)


INSTALL = r"""
set -e
export PATH="/var/tmp/vf-node/bin:$PATH"
prefix="$VF_PRIME_AGENT_DIR/$PRIME_AGENT_CACHE_KEY"
[ -x "$prefix/bin/prime-agent" ] && exit 0
export NPM_CONFIG_PREFIX="$prefix"
export PRIME_AGENT_BOOTSTRAP_KERNEL_ON_INSTALL=0
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
    "$PRIME_AGENT_SHA256" "$agent_tarball" \
    "$PRIME_AGENT_AI_SHA256" "$ai_tarball" \
    "$PRIME_AGENT_CORE_SHA256" "$core_tarball" \
    "$PRIME_AGENT_TUI_SHA256" "$tui_tarball" \
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
"""


class PrimeAgentHarnessConfig(HarnessConfig):
    release: str = "stable"
    """Prime Agent stable channel, or an exact stable version such as ``0.8.1``."""

    resolved_release: PrimeAgentRelease | None = None
    """Release frozen before the evaluation is saved or workers start."""

    autonomous: bool = False
    """Enable Prime Agent's autonomous continuation loop."""

    @field_validator("release", mode="before")
    @classmethod
    def _validate_release(cls, value: object) -> str:
        if not isinstance(value, str):
            raise TypeError("Prime Agent release must be 'stable' or an exact version")
        value = value.strip().lower()
        return value if value == "stable" else _version(value)

    @model_validator(mode="after")
    def _validate_resolved_release(self) -> "PrimeAgentHarnessConfig":
        resolved = self.resolved_release
        if resolved is None:
            return self
        entries = [
            {"file": resolved.file(package), "sha256": sha256}
            for package, sha256 in resolved.sha256.items()
        ]
        if _release(resolved.version, entries) != resolved:
            raise ValueError("invalid frozen Prime Agent release")
        if self.release != "stable" and self.release != resolved.version:
            raise ValueError("frozen Prime Agent release does not match exact version")
        return self


class PrimeAgentHarness(ACPHarness[PrimeAgentHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_RESUME = True
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

    async def _resolve(self) -> PrimeAgentRelease:
        if self.config.resolved_release is None:
            try:
                self.config.resolved_release = await _fetch_release(self.config.release)
            except (httpx.HTTPError, TypeError, ValueError) as e:
                raise RuntimeError(
                    f"failed to resolve Prime Agent release "
                    f"{self.config.release!r}: {e}"
                ) from e
        return self.config.resolved_release

    def _resolved(self) -> PrimeAgentRelease:
        release = self.config.resolved_release
        if release is None:
            raise RuntimeError("Prime Agent release was not resolved during setup")
        return release

    async def prepare(self) -> None:
        await self._resolve()

    async def setup(self, runtime: Runtime) -> None:
        release = await self._resolve()
        await self.install_skills(runtime, SKILLS_DIR)
        await ensure_node(runtime)
        logger.info(
            "prime-agent: release %s resolved to %s (%s)",
            self.config.release,
            release.version,
            release.cache_key,
        )
        lock = f"{PRIME_AGENT_DIR}/install.lock"
        guarded = (
            f"mkdir -p {PRIME_AGENT_DIR} && "
            f'"$(command -v flock || command -v lockf)" {lock} '
            f"sh -c {shlex.quote(INSTALL)}"
        )
        result = await runtime.run(
            ["sh", "-c", guarded],
            {
                **self.config.resolved_env,
                "VF_PRIME_AGENT_DIR": PRIME_AGENT_DIR,
                "VF_PRIME_AGENT_GITHUB_RELEASE_URL": GITHUB_RELEASE_URL,
                "PRIME_AGENT_CACHE_KEY": release.cache_key,
                "PRIME_AGENT_RELEASE_VERSION": release.version,
                "PRIME_AGENT_SHA256": release.sha256["prime-agent"],
                "PRIME_AGENT_AI_SHA256": release.sha256["prime-agent-ai"],
                "PRIME_AGENT_CORE_SHA256": release.sha256["prime-agent-core"],
                "PRIME_AGENT_TUI_SHA256": release.sha256["prime-agent-tui"],
            },
        )
        if result.exit_code != 0:
            raise RuntimeError(
                f"prime-agent install failed: {result.stderr.strip()[-500:]}"
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

        release = self._resolved()
        trace.info["prime_agent_release"] = {
            "requested": self.config.release,
            "version": release.version,
            "source": _source(self.config.release),
            "cache_key": release.cache_key,
            "artifacts": [
                {
                    "file": release.file(package),
                    "sha256": release.sha256[package],
                }
                for package in RELEASE_PACKAGES
            ],
        }
        root = self._root(trace)
        agent_dir = f"{root}/agent"
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
        args = [
            self._bin(),
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
            args += ["--skill", f"{SKILLS_DIR}/{skill.resolve().name}"]
        if system_prompt:
            args += ["--append-system-prompt", system_prompt]

        wrapper = f"{root}/prime-agent"
        await runtime.write(
            wrapper,
            (
                "#!/bin/sh\n"
                "set -eu\n"
                f'export PATH="{NODE_BIN_DIR}:$HOME/.local/bin:$PATH"\n'
                f'exec {shlex.join(args)} "$@"\n'
            ).encode(),
        )
        executable = await runtime.run(["chmod", "700", wrapper], {})
        if executable.exit_code != 0:
            raise RuntimeError(
                f"prime-agent wrapper chmod failed: {executable.stderr.strip()[-500:]}"
            )

        return ACPConfig(
            env=self._env(trace, secret),
            command=[wrapper],
            prompt=prompt,
        )

    async def cleanup(self, trace: Trace, runtime: Runtime) -> None:
        root = self._root(trace)
        removed = await runtime.run(["rm", "-rf", root], {})
        if removed.exit_code != 0:
            raise RuntimeError(
                f"prime-agent state cleanup failed: {removed.stderr.strip()[-500:]}"
            )

    def _bin(self) -> str:
        return f"{PRIME_AGENT_DIR}/{self._resolved().cache_key}/bin/prime-agent"

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
