"""Run Prime Agent through its native ACP mode."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shlex
from typing import Literal

from pydantic import Field, field_validator, model_validator

from verifiers.v1.acp import ACP
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness, HarnessSession
from verifiers.v1.harnesses.node import NODE_BIN_DIR, ensure_node
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
THINKING_LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh", "max")
ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"]

INSTALL = r"""
set -eu
root="$VF_PRIME_AGENT_INSTALL_DIR"
package_sha="$VF_PRIME_AGENT_TARBALL_SHA256"
stamp="$root/.installed"

if [ -x "$VF_PRIME_AGENT_BIN" ] \
    && [ "$(cat "$stamp" 2>/dev/null)" = "$package_sha" ]; then
    exit 0
fi

staging="${root}.staging.$$"
backup="${root}.previous.$$"
trap 'rm -rf "$staging" "$backup"' EXIT
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
    mv "$root" "$backup"
fi
mv "$staging" "$root"
rm -rf "$backup"
trap - EXIT
"""

PRIME_AGENT_ACP = ACP()


def _guarded_install(root: str, lock: str, command: str) -> str:
    """Guard an install with a lock tied to one Linux process lifetime."""
    return (
        "set -eu\n"
        f"mkdir -p {shlex.quote(root)}\n"
        f"lock={shlex.quote(lock)}\n"
        "self_start=$(awk '{print $22}' \"/proc/$$/stat\")\n"
        'case "$self_start" in\n'
        "  ''|*[!0-9]*) echo 'prime-agent: cannot identify installer process' >&2; "
        "exit 1 ;;\n"
        "esac\n"
        'owner="$$:$self_start"\n'
        'while ! ln -s "$owner" "$lock" 2>/dev/null; do\n'
        '  current=$(readlink "$lock" 2>/dev/null || true)\n'
        "  stale=0\n"
        '  case "$current" in\n'
        "    *:*)\n"
        "      pid=${current%%:*}\n"
        "      start=${current#*:}\n"
        '      case "$pid" in\n'
        "        ''|*[!0-9]*) stale=1 ;;\n"
        "      esac\n"
        '      case "$start" in\n'
        "        ''|*[!0-9]*) stale=1 ;;\n"
        "      esac\n"
        '      if [ "$stale" -eq 0 ]; then\n'
        "        live_start=$(awk '{print $22}' \"/proc/$pid/stat\" "
        "2>/dev/null || true)\n"
        '        [ "$live_start" = "$start" ] || stale=1\n'
        "      fi\n"
        "      ;;\n"
        "    *) stale=1 ;;\n"
        "  esac\n"
        '  if [ "$stale" -eq 1 ] \\\n'
        '      && [ "$(readlink "$lock" 2>/dev/null || true)" = "$current" ]; then\n'
        '    rm -f "$lock"\n'
        "  fi\n"
        "  sleep 0.1\n"
        "done\n"
        "release_install_lock() {\n"
        '  if [ "$(readlink "$lock" 2>/dev/null || true)" = "$owner" ]; then\n'
        '    rm -f "$lock"\n'
        "  fi\n"
        "}\n"
        "trap release_install_lock EXIT\n"
        f"{command}\n"
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
        if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?", version):
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

    async def setup(self, runtime: Runtime) -> None:
        await ensure_node(runtime)
        logger.info("prime-agent: ensuring %s is installed", self.config.version)
        install_dir = self.install_dir()
        lock = f"{install_dir}.install.lock"
        guarded = _guarded_install(
            INSTALL_ROOT,
            lock,
            f"sh -c {shlex.quote(INSTALL)}",
        )
        install = await runtime.run(
            ["sh", "-c", guarded],
            {
                **self.config.resolved_env,
                "VF_NODE_BIN_DIR": NODE_BIN_DIR,
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
                f'export PATH="{NODE_BIN_DIR}:$PATH"\n'
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
