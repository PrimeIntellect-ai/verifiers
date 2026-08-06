"""Run Prime Agent through its native ACP mode."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shlex
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from verifiers.v1.acp import ACP
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness, HarnessSession
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

ENV_AGENT_DIR = "PRIME_AGENT_CODING_AGENT_DIR"
PROVIDER = "intercept"
KEY_VAR = "PRIME_AGENT_INTERCEPT_KEY"

RELEASE_BASE_URL = "https://pub-728493de92a943e2a9b2d17b4719f318.r2.dev"
DEFAULT_VERSION = "0.7.0"
DEFAULT_TARBALL_SHA256 = (
    "88b6578518c72cd51a825bc80f28e0fef9a64c67de4a7d6fd7afd7ca1b34da0b"
)
INSTALL_ROOT = "/var/tmp/vf-prime-agent"
STATE_ROOT = "/tmp/vf-prime-agent"
HARNESS_STATE_FILENAME = "harness_state.json"
REFINEMENT_HISTORY_FILENAME = "refinements.jsonl"
NODE_VERSION = "22.19.0"
THINKING_LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh", "max")
ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"]

_SEMVER_IDENTIFIER = r"(?:0|[1-9][0-9]*|[0-9A-Za-z-]*[A-Za-z-][0-9A-Za-z-]*)"
_SEMVER = re.compile(
    rf"(?:0|[1-9][0-9]*)\."
    rf"(?:0|[1-9][0-9]*)\."
    rf"(?:0|[1-9][0-9]*)"
    rf"(?:-{_SEMVER_IDENTIFIER}(?:\.{_SEMVER_IDENTIFIER})*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)

INSTALL = r"""
set -eu
root="$VF_PRIME_AGENT_INSTALL_DIR"
package_sha="$VF_PRIME_AGENT_TARBALL_SHA256"
stamp="$root/.installed"
node_root="$VF_NODE_ROOT"
export PATH="$VF_NODE_BIN_DIR:$PATH"

node_ok() {
    node -e 'const [a,b]=process.versions.node.split(".").map(Number); process.exit(a>22 || a===22 && b>=8 ? 0 : 1)'
}

# Prime VMs mount /tmp as a small tmpfs. Keep Node and the Prime Agent install
# on /var/tmp, which is backed by the sandbox filesystem.
if [ -f /etc/alpine-release ]; then
    apk add --no-cache curl ca-certificates nodejs-current npm >/dev/null
    if ! node_ok; then
        sed -E -i 's/v[0-9]+\.[0-9]+/v3.22/g' /etc/apk/repositories
        apk upgrade --available --no-cache >/dev/null
        apk add --no-cache nodejs-current npm >/dev/null
    fi
else
    if ! command -v curl >/dev/null 2>&1; then
        command -v apt-get >/dev/null 2>&1 || {
            echo "prime-agent setup needs curl, or apt-get to install it" >&2
            exit 1
        }
        apt-get update -qq
        apt-get install -y -qq curl ca-certificates >/dev/null
    fi
    if [ ! -x "$VF_NODE_BIN_DIR/node" ] \
        || [ "$("$VF_NODE_BIN_DIR/node" --version 2>/dev/null)" != "v$VF_NODE_VERSION" ]; then
        case "$(uname -s)" in
            Linux) node_os=linux ;;
            Darwin) node_os=darwin ;;
            *) echo "unsupported os: $(uname -s)" >&2; exit 1 ;;
        esac
        case "$(uname -m)" in
            aarch64|arm64) node_arch=arm64 ;;
            x86_64|amd64) node_arch=x64 ;;
            *) echo "unsupported architecture: $(uname -m)" >&2; exit 1 ;;
        esac
        rm -rf "$node_root"
        mkdir -p "$node_root"
        curl -fsSL \
            "https://nodejs.org/dist/v$VF_NODE_VERSION/node-v$VF_NODE_VERSION-${node_os}-${node_arch}.tar.gz" \
            | tar -xz -C "$node_root" --strip-components=1
    fi
fi
node_ok || { echo "prime-agent requires Node.js 22.8 or newer" >&2; exit 1; }

if [ -x "$VF_PRIME_AGENT_BIN" ] \
    && [ "$(cat "$stamp" 2>/dev/null)" = "$package_sha" ]; then
    exit 0
fi

staging="${root}.staging.$$"
backup="${root}.previous.$$"
had_root=0
committed=0
cleanup_install() {
    status=$?
    trap - EXIT HUP INT TERM
    rm -rf "$staging"
    if [ "$committed" -eq 0 ] && [ "$had_root" -eq 1 ] && [ -e "$backup" ]; then
        rm -rf "$root"
        mv "$backup" "$root" || \
            echo "prime-agent: failed to restore previous install" >&2
    elif [ "$committed" -eq 1 ]; then
        rm -rf "$backup"
    fi
    exit "$status"
}
trap cleanup_install EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
rm -rf "$staging" "$backup"
mkdir -p "$staging"

curl -fsSL "$VF_PRIME_AGENT_TARBALL" -o "$staging/prime-agent.tgz"
if command -v sha256sum >/dev/null 2>&1; then
    actual="$(sha256sum "$staging/prime-agent.tgz" | cut -d ' ' -f 1)"
else
    actual="$(shasum -a 256 "$staging/prime-agent.tgz" | cut -d ' ' -f 1)"
fi
if [ "$actual" != "$package_sha" ]; then
    echo "prime-agent tarball checksum mismatch" >&2
    exit 1
fi

export PATH="$VF_NODE_BIN_DIR:$PATH"
export PRIME_AGENT_BOOTSTRAP_KERNEL_ON_INSTALL=1
export PRIME_AGENT_KERNEL_VENV="$VF_PRIME_AGENT_KERNEL_VENV"
npm install --no-audit --no-fund --prefix "$staging" \
    "$staging/prime-agent.tgz" >/dev/null
printf %s "$package_sha" > "$staging/.installed"

if [ -e "$root" ]; then
    had_root=1
    mv "$root" "$backup"
fi
mv "$staging" "$root"
committed=1
rm -rf "$backup"
trap - EXIT HUP INT TERM
"""

PRIME_AGENT_ACP = ACP()


def _guarded_install(root: str, lock: str, command: str) -> str:
    """Serialize an install without PID files or stale-lock deletion races."""
    return (
        "set -eu\n"
        f"mkdir -p {shlex.quote(root)}\n"
        f"lock={shlex.quote(lock)}\n"
        "if command -v flock >/dev/null 2>&1; then\n"
        "  (\n"
        "    flock -x 9\n"
        f"    {command}\n"
        '  ) 9>"$lock"\n'
        "else\n"
        '  lock_dir="${lock}.d"\n'
        "  waited=0\n"
        '  while ! mkdir "$lock_dir" 2>/dev/null; do\n'
        '    if [ "$waited" -ge 300 ]; then\n'
        "      echo 'prime-agent: timed out waiting for install lock' >&2\n"
        "      exit 1\n"
        "    fi\n"
        "    waited=$((waited + 1))\n"
        "    sleep 1\n"
        "  done\n"
        "  release_install_lock() {\n"
        '    rmdir "$lock_dir" 2>/dev/null || true\n'
        "  }\n"
        "  trap release_install_lock EXIT\n"
        "  trap 'exit 129' HUP\n"
        "  trap 'exit 130' INT\n"
        "  trap 'exit 143' TERM\n"
        f"  {command}\n"
        '  rmdir "$lock_dir"\n'
        "  trap - EXIT HUP INT TERM\n"
        "fi\n"
    )


class PrimeAgentHarnessConfig(HarnessConfig):
    id: Literal["prime-agent"] = "prime-agent"

    version: str = DEFAULT_VERSION
    """Prime Agent release to install."""

    tarball_url: str | None = None
    """Override the release tarball URL."""

    tarball_sha256: str | None = None
    """SHA-256 for a non-default release or custom tarball."""

    thinking_level: ThinkingLevel | None = None
    """Override the sampling reasoning effort passed to Prime Agent."""

    model_context_window: int = Field(default=128_000, strict=True, gt=0)
    model_max_tokens: int = Field(default=16_384, strict=True, gt=0)
    load_context_files: bool = False
    save_session: bool = False
    max_depth: int = Field(default=0, strict=True, ge=0)

    harness_state_dir: Path | None = None
    """Host directory with ``harness_state.json`` and optional refinement history.

    The files seed Prime Agent's global continual harness state for this run. An
    environment can harvest the corresponding runtime directory before cleanup
    when it owns cross-run publication.
    """

    autonomous: bool = False
    gates: list[str] = Field(default_factory=list)
    autonomous_gate_retries: int | None = Field(default=None, strict=True, gt=0)
    autonomous_gate_timeout_ms: int | None = Field(default=None, strict=True, gt=0)
    autonomous_max_continuations: int | None = Field(default=None, strict=True, gt=0)
    autonomous_max_turns: int | None = Field(default=None, strict=True, gt=0)
    autonomous_max_tokens: int | None = Field(default=None, strict=True, gt=0)
    autonomous_timeout_ms: int | None = Field(default=None, strict=True, gt=0)

    goal: str | None = None
    goal_token_budget: int | None = Field(default=None, strict=True, gt=0)

    @field_validator("version")
    @classmethod
    def normalize_version(cls, value: str) -> str:
        version = value.removeprefix("v")
        if not _SEMVER.fullmatch(version):
            raise ValueError("version must be a semantic version")
        return version

    @field_validator("tarball_sha256")
    @classmethod
    def normalize_sha256(cls, value: str | None) -> str | None:
        if value is None:
            return None
        digest = value.lower()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("tarball_sha256 must be a 64-character hexadecimal digest")
        return digest

    @model_validator(mode="after")
    def validate_options(self) -> PrimeAgentHarnessConfig:
        if self.tarball_url and not self.tarball_sha256:
            raise ValueError("tarball_url requires tarball_sha256")
        if self.version != DEFAULT_VERSION and not self.tarball_sha256:
            raise ValueError("a non-default version requires tarball_sha256")
        autonomous_options = (
            self.gates,
            self.autonomous_gate_retries,
            self.autonomous_gate_timeout_ms,
            self.autonomous_max_continuations,
            self.autonomous_max_turns,
            self.autonomous_max_tokens,
            self.autonomous_timeout_ms,
        )
        if not self.autonomous and any(value for value in autonomous_options):
            raise ValueError("autonomous options require autonomous=true")
        if any(not gate.strip() for gate in self.gates):
            raise ValueError("gates must contain non-empty commands")
        if self.goal is not None and not self.goal.strip():
            raise ValueError("goal must be non-empty")
        if self.goal_token_budget is not None and not self.goal:
            raise ValueError("goal_token_budget requires goal")
        unknown_tools = set(self.disabled_tools or []) - {"ipython"}
        if unknown_tools:
            names = ", ".join(sorted(unknown_tools))
            raise ValueError(f"prime-agent cannot disable unknown tools: {names}")
        return self


class PrimeAgentHarness(Harness[PrimeAgentHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = False
    SUPPORTS_RESUME = True
    SUPPORTS_SKILLS = True

    def tarball(self) -> str:
        if self.config.tarball_url:
            return self.config.tarball_url
        version = self.config.version
        return f"{RELEASE_BASE_URL}/releases/v{version}/prime-agent-{version}.tgz"

    def tarball_sha256(self) -> str:
        return self.config.tarball_sha256 or DEFAULT_TARBALL_SHA256

    def install_dir(self) -> str:
        return f"{INSTALL_ROOT}/{self.config.version}-{self.tarball_sha256()[:16]}"

    def prime_agent_bin(self) -> str:
        return f"{self.install_dir()}/node_modules/.bin/prime-agent"

    def kernel_venv(self) -> str:
        return (
            f"{INSTALL_ROOT}/kernels/{self.config.version}-{self.tarball_sha256()[:16]}"
        )

    def node_root(self) -> str:
        identity = f"{self.config.version}-{self.tarball_sha256()[:16]}"
        return f"{INSTALL_ROOT}/nodes/{identity}/node-{NODE_VERSION}"

    def node_bin_dir(self) -> str:
        return f"{self.node_root()}/bin"

    @staticmethod
    def trace_root(trace: Trace) -> str:
        # Keep nested Unix socket paths short while retaining a 128-bit key.
        trace_key = hashlib.sha256(trace.id.encode()).hexdigest()[:32]
        return f"{STATE_ROOT}/{trace_key}"

    @classmethod
    def agent_dir(cls, trace: Trace) -> str:
        return f"{cls.trace_root(trace)}/agent"

    @classmethod
    def daemon_socket(cls, trace: Trace) -> str:
        return f"{cls.trace_root(trace)}/daemon.sock"

    @classmethod
    def temp_dir(cls, trace: Trace) -> str:
        return f"{cls.trace_root(trace)}/tmp"

    @classmethod
    def runtime_harness_state_dir(cls, trace: Trace) -> str:
        return f"{cls.agent_dir(trace)}/harness"

    @classmethod
    def harness_state_path(cls, trace: Trace) -> str:
        return f"{cls.runtime_harness_state_dir(trace)}/{HARNESS_STATE_FILENAME}"

    @classmethod
    def refinement_history_path(cls, trace: Trace) -> str:
        return f"{cls.runtime_harness_state_dir(trace)}/{REFINEMENT_HISTORY_FILENAME}"

    async def install_harness_state(self, runtime: Runtime, trace: Trace) -> None:
        source = self.config.harness_state_dir
        if source is None:
            return
        if source.is_symlink():
            raise ValueError(f"harness_state_dir {str(source)!r} is not a folder")
        source = source.resolve()
        if not source.is_dir():
            raise ValueError(f"harness_state_dir {str(source)!r} is not a folder")
        destination = self.runtime_harness_state_dir(trace)
        created = await runtime.run(["mkdir", "-p", "-m", "700", destination], {})
        if created.exit_code != 0:
            raise RuntimeError(
                "prime-agent harness state directory failed: "
                f"{created.stderr.strip()[-500:]}"
            )
        for name in (HARNESS_STATE_FILENAME, REFINEMENT_HISTORY_FILENAME):
            path = source / name
            if not path.exists():
                continue
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"harness state file {str(path)!r} is not a file")
            target = f"{destination}/{name}"
            present = await runtime.run(
                [
                    "sh",
                    "-c",
                    'if [ -L "$1" ]; then exit 2; fi; [ -e "$1" ]',
                    "prime-agent-harness-state",
                    target,
                ],
                {},
            )
            if present.exit_code == 0:
                continue
            if present.exit_code == 2:
                raise RuntimeError(
                    f"prime-agent harness state target is a symlink: {target}"
                )
            if present.exit_code != 1:
                raise RuntimeError(
                    f"prime-agent harness state could not be checked: {target}"
                )
            await runtime.write(target, path.read_bytes())
        restricted = await runtime.run(["chmod", "-R", "go-rwx", destination], {})
        if restricted.exit_code != 0:
            raise RuntimeError(
                "prime-agent harness state permissions failed: "
                f"{restricted.stderr.strip()[-500:]}"
            )

    async def setup(self, runtime: Runtime) -> None:
        logger.info("prime-agent: ensuring %s is installed", self.config.version)
        install_dir = self.install_dir()
        lock = f"{install_dir}.install.flock"
        guarded = _guarded_install(
            INSTALL_ROOT,
            lock,
            f"sh -c {shlex.quote(INSTALL)}",
        )
        install = await runtime.run(
            ["sh", "-c", guarded],
            {
                **self.config.resolved_env,
                "VF_NODE_BIN_DIR": self.node_bin_dir(),
                "VF_NODE_ROOT": self.node_root(),
                "VF_NODE_VERSION": NODE_VERSION,
                "VF_PRIME_AGENT_BIN": self.prime_agent_bin(),
                "VF_PRIME_AGENT_INSTALL_DIR": install_dir,
                "VF_PRIME_AGENT_KERNEL_VENV": self.kernel_venv(),
                "VF_PRIME_AGENT_TARBALL": self.tarball(),
                "VF_PRIME_AGENT_TARBALL_SHA256": self.tarball_sha256(),
            },
        )
        if install.exit_code != 0:
            detail = (install.stderr or install.stdout).strip()[-1000:]
            raise RuntimeError(f"prime-agent install failed: {detail}")
        await PRIME_AGENT_ACP.setup(self, runtime)

    async def session(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
    ) -> HarnessSession:
        if not runtime.supports_live_processes:
            return await super().session(
                ctx, trace, runtime, endpoint, secret, mcp_urls, data
            )
        system_prompt, prompt = self.resolve_prompt(data)
        env, command = await self.prepare_run(
            ctx, trace, runtime, endpoint, secret, system_prompt
        )
        return PRIME_AGENT_ACP.session(
            self,
            ctx,
            trace,
            runtime,
            endpoint,
            secret,
            {},
            data,
            env=env,
            command=command,
            prompt=prompt,
        )

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
        env, command = await self.prepare_run(
            ctx, trace, runtime, endpoint, secret, system_prompt
        )
        return await PRIME_AGENT_ACP.run(runtime, env, command, prompt)

    async def prepare_run(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        system_prompt: str | None,
    ) -> tuple[dict[str, str], list[str]]:
        root = self.trace_root(trace)
        agent_dir = self.agent_dir(trace)
        temp_dir = self.temp_dir(trace)
        created = await runtime.run(
            ["mkdir", "-p", "-m", "700", root, agent_dir, temp_dir], {}
        )
        if created.exit_code != 0:
            raise RuntimeError(
                f"prime-agent state directory failed: {created.stderr.strip()[-500:]}"
            )

        skills_dir = f"{agent_dir}/skills"
        await self.install_skills(runtime, skills_dir)
        await self.install_harness_state(runtime, trace)
        thinking = self.thinking_level(ctx)
        reasoning = thinking not in (None, "off") or (
            thinking is None
            and ctx.model.rsplit("/", 1)[-1].startswith(("gpt-5", "o1", "o3", "o4"))
        )
        models = {
            "providers": {
                PROVIDER: {
                    "baseUrl": endpoint,
                    "api": "openai-completions",
                    "apiKey": KEY_VAR,
                    "models": [
                        {
                            "id": ctx.model,
                            "name": ctx.model,
                            "reasoning": reasoning,
                            "input": ["text", "image"],
                            "contextWindow": self.config.model_context_window,
                            "maxTokens": (
                                ctx.sampling.max_tokens
                                if ctx.sampling.max_tokens is not None
                                else self.config.model_max_tokens
                            ),
                        }
                    ],
                }
            }
        }
        models_path = f"{agent_dir}/models.json"
        await runtime.write(models_path, json.dumps(models).encode())
        restricted = await runtime.run(["chmod", "700", root, agent_dir, temp_dir], {})
        if restricted.exit_code != 0:
            raise RuntimeError(
                f"prime-agent state permissions failed: {restricted.stderr.strip()[-500:]}"
            )
        restricted = await runtime.run(["chmod", "600", models_path], {})
        if restricted.exit_code != 0:
            raise RuntimeError(
                f"prime-agent model permissions failed: {restricted.stderr.strip()[-500:]}"
            )

        wrapper = f"{root}/prime-agent"
        await runtime.write(
            wrapper,
            (
                "#!/bin/sh\n"
                f'export PATH="{self.node_bin_dir()}:$PATH"\n'
                f'exec {shlex.quote(self.prime_agent_bin())} "$@"\n'
            ).encode(),
        )
        restricted = await runtime.run(["chmod", "700", wrapper], {})
        if restricted.exit_code != 0:
            raise RuntimeError(
                f"prime-agent wrapper permissions failed: {restricted.stderr.strip()[-500:]}"
            )

        env = self.run_env(trace, secret)
        args = [
            wrapper,
            "--mode",
            "acp",
            "--daemon-socket",
            self.daemon_socket(trace),
            "--provider",
            PROVIDER,
            "--model",
            ctx.model,
        ]
        if thinking is not None:
            args += ["--thinking", thinking]
        if not self.config.save_session:
            args.append("--no-session")
        if not self.config.load_context_files:
            args.append("--no-context-files")
        if self.config.disabled_tools:
            args.append("--no-builtin-tools")
        for skill in self.config.skills:
            args += ["--skill", f"{skills_dir}/{skill.resolve().name}"]
        if self.config.autonomous:
            args.append("--autonomous")
            for gate in self.config.gates:
                args += ["--autonomous-gate", gate]
            for flag, value in (
                ("--autonomous-gate-retries", self.config.autonomous_gate_retries),
                (
                    "--autonomous-gate-timeout-ms",
                    self.config.autonomous_gate_timeout_ms,
                ),
                (
                    "--autonomous-max-continuations",
                    self.config.autonomous_max_continuations,
                ),
                ("--autonomous-max-turns", self.config.autonomous_max_turns),
                ("--autonomous-max-tokens", self.config.autonomous_max_tokens),
                ("--autonomous-timeout-ms", self.config.autonomous_timeout_ms),
            ):
                if value is not None:
                    args += [flag, str(value)]
        if self.config.goal is not None:
            args += ["--goal", self.config.goal]
            if self.config.goal_token_budget is not None:
                args += ["--goal-token-budget", str(self.config.goal_token_budget)]
        if system_prompt:
            args += ["--append-system-prompt", system_prompt]
        return env, args

    def run_env(self, trace: Trace, secret: str | None = None) -> dict[str, str]:
        agent_dir = self.agent_dir(trace)
        temp_dir = self.temp_dir(trace)
        return {
            **self.config.resolved_env,
            **({KEY_VAR: secret} if secret is not None else {}),
            ENV_AGENT_DIR: agent_dir,
            "NO_COLOR": "1",
            "PI_OFFLINE": "1",
            "PRIME_AGENT_KERNEL_VENV": self.kernel_venv(),
            "RLM_MAX_DEPTH": str(self.config.max_depth),
            "TEMP": temp_dir,
            "TMP": temp_dir,
            "TMPDIR": temp_dir,
        }

    def thinking_level(self, ctx: ModelContext) -> ThinkingLevel | None:
        if self.config.thinking_level is not None:
            return self.config.thinking_level
        effort = ctx.sampling.reasoning_effort
        if effort is None:
            return None
        normalized = effort.lower()
        if normalized == "none":
            return "off"
        if normalized not in THINKING_LEVELS:
            supported = ", ".join(("none", *THINKING_LEVELS))
            raise ValueError(
                f"prime-agent does not support reasoning_effort={effort!r}; "
                f"expected one of: {supported}"
            )
        return normalized  # type: ignore[return-value]

    async def cleanup(self, trace: Trace, runtime: Runtime) -> None:
        root = self.trace_root(trace)
        wrapper = f"{root}/prime-agent"
        shutdown = await runtime.run(
            [
                "sh",
                "-c",
                'if [ -x "$1" ]; then "$1" shutdown --force; fi',
                "prime-agent-cleanup",
                wrapper,
            ],
            self.run_env(trace),
        )
        if shutdown.exit_code != 0:
            detail = (shutdown.stderr or shutdown.stdout).strip()[-500:]
            raise RuntimeError(f"prime-agent daemon shutdown failed: {detail}")
        removed = await runtime.run(["rm", "-rf", root], {})
        if removed.exit_code != 0:
            raise RuntimeError(
                f"prime-agent state cleanup failed: {removed.stderr.strip()[-500:]}"
            )
