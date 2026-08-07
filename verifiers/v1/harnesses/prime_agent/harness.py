"""Run Prime Agent against interception through its native ACP mode."""

import hashlib
import json
import logging
import re
import shlex
from pathlib import Path

from pydantic import Field, field_validator, model_validator

from verifiers.v1.acp import ACP
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.errors import RolloutError
from verifiers.v1.harness import Harness, HarnessSession
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

INSTALL_SOURCE = (Path(__file__).resolve().parent / "install.sh").read_text()

PRIME_AGENT_ACP = ACP()
PROVIDER = "intercept"
# models.json stores this variable NAME, never the secret: Prime Agent resolves
# `process.env[apiKey] || apiKey`, so the token reaches the agent only through
# the process environment.
KEY_VAR = "PRIME_AGENT_INTERCEPT_KEY"
# Prime Agent reads its agent directory from its packaged config name.
ENV_AGENT_DIR = "PRIME_AGENT_CODING_AGENT_DIR"

DEFAULT_VERSION = "0.7.0"
DEFAULT_TARBALL_SHA256 = (
    "88b6578518c72cd51a825bc80f28e0fef9a64c67de4a7d6fd7afd7ca1b34da0b"
)
RELEASE_BASE_URL = "https://pub-728493de92a943e2a9b2d17b4719f318.r2.dev"
NODE_VERSION = "22.19.0"

INSTALL_ROOT = "/var/tmp/vf-prime-agent"
STATE_ROOT = "/tmp/vf-prime-agent-state"
# The daemon builds worker sockets as
# $TMPDIR/prime-agent-<uid>/worker-<12>-<12>.sock, which adds 54 characters. A
# per-trace TMPDIR under STATE_ROOT pushed that past the 108-byte sun_path limit,
# so listen() returned EINVAL and the supervisor blocked for its full 30s worker
# timeout -- surfacing only as an opaque ACP "create" timeout. Keep the socket
# root short and separate from the (longer) state root.
TMP_ROOT = "/tmp/vfpa"

THINKING_LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh", "max")
# Prime Agent's model-facing tool surface is IPython; there is no per-tool
# disable flag, only the allowlist forms (--tools/--no-tools/--no-builtin-tools).
KNOWN_TOOLS = ("ipython",)


class PrimeAgentHarnessConfig(HarnessConfig):
    version: str = Field(default=DEFAULT_VERSION)
    """Prime Agent release to install, pinned for reproducibility."""

    tarball_url: str | None = None
    """Override the release tarball URL. Requires `tarball_sha256`."""

    tarball_sha256: str | None = None
    """SHA-256 digest for a non-default release or a custom tarball."""

    thinking: str | None = None
    """Reasoning level passed as `--thinking`; Prime Agent has no `--effort`."""

    autonomous: bool = False
    """Run with `--autonomous` so the agent continues until its limits or gates stop it."""

    gates: list[str] = Field(default_factory=list)
    """Autonomous quality-gate commands. Requires `autonomous`."""

    @field_validator("version")
    @classmethod
    def _semver(cls, value: str) -> str:
        if not re.fullmatch(
            r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
            r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?",
            value,
        ):
            raise ValueError("version must be a semantic version, e.g. 0.7.0")
        return value

    @field_validator("tarball_sha256")
    @classmethod
    def _digest(cls, value: str | None) -> str | None:
        if value is None:
            return None
        digest = value.strip().lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("tarball_sha256 must be a 64-character hexadecimal digest")
        return digest

    @field_validator("thinking")
    @classmethod
    def _thinking(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value not in THINKING_LEVELS:
            raise ValueError(f"thinking must be one of {', '.join(THINKING_LEVELS)}")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> "PrimeAgentHarnessConfig":
        # An unpinned artifact must never be installed unverified.
        if self.tarball_url and not self.tarball_sha256:
            raise ValueError("tarball_url requires tarball_sha256")
        if self.version != DEFAULT_VERSION and not self.tarball_sha256:
            raise ValueError("a non-default version requires tarball_sha256")
        if self.gates and not self.autonomous:
            raise ValueError("prime-agent gates require autonomous=true")
        unknown = set(self.disabled_tools or []) - set(KNOWN_TOOLS)
        if unknown:
            raise ValueError(
                "prime-agent has no per-tool disable flag; unknown tools: "
                + ", ".join(sorted(unknown))
            )
        return self


class PrimeAgentHarness(Harness[PrimeAgentHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    # Prime Agent's ACP mode ignores `session/new` mcpServers on the pinned
    # release, so claiming support would let a tool-bearing taskset run with its
    # tools silently absent. Once a release advertises
    # `agentCapabilities.mcpCapabilities.http`, sniff it rather than hardcoding.
    SUPPORTS_MCP = False
    SUPPORTS_RESUME = True
    SUPPORTS_SKILLS = True

    def tarball_url(self) -> str:
        if self.config.tarball_url:
            return self.config.tarball_url
        version = self.config.version
        return f"{RELEASE_BASE_URL}/releases/v{version}/prime-agent-{version}.tgz"

    def tarball_sha256(self) -> str:
        return self.config.tarball_sha256 or DEFAULT_TARBALL_SHA256

    def install_dir(self) -> str:
        # Key the shared install on version+digest so a changed pin cannot reuse
        # whatever an earlier rollout left behind.
        return f"{INSTALL_ROOT}/{self.config.version}-{self.tarball_sha256()[:16]}"

    def node_root(self) -> str:
        return f"{INSTALL_ROOT}/node-{NODE_VERSION}"

    def prime_agent_bin(self) -> str:
        return f"{self.install_dir()}/node_modules/.bin/prime-agent"

    def trace_key(self, trace: Trace) -> str:
        # Hash the trace id: it is untrusted input for a filesystem path.
        return hashlib.sha256(trace.id.encode()).hexdigest()[:32]

    def tmp_dir(self, trace: Trace) -> str:
        """Short per-trace TMPDIR, sized for the daemon's worker socket paths.

        16 hex characters keep the longest derived socket well inside sun_path
        while still being collision-free in practice for concurrent rollouts.
        """
        return f"{TMP_ROOT}/{self.trace_key(trace)[:16]}"

    def trace_root(self, trace: Trace) -> str:
        return f"{STATE_ROOT}/{self.trace_key(trace)}"

    async def setup(self, runtime: Runtime) -> None:
        logger.info("prime-agent: ensuring %s is installed", self.config.version)
        lock = f"{INSTALL_ROOT}/install.lock"
        # flock only; a mkdir-based fallback leaves a stale lock directory after
        # SIGKILL that blocks every later install.
        guarded = (
            f"mkdir -p {shlex.quote(INSTALL_ROOT)} && "
            f"flock -x 9 sh -c {shlex.quote(INSTALL_SOURCE)} 9>{shlex.quote(lock)}"
        )
        result = await runtime.run(
            ["sh", "-c", guarded],
            {
                "VF_PA_INSTALL_DIR": self.install_dir(),
                "VF_PA_TARBALL_URL": self.tarball_url(),
                "VF_PA_TARBALL_SHA256": self.tarball_sha256(),
                "VF_PA_NODE_ROOT": self.node_root(),
                "VF_PA_NODE_VERSION": NODE_VERSION,
                "VF_PA_UV_VERSION": "0.8.17",
            },
        )
        if result.exit_code != 0:
            raise RuntimeError(
                f"prime-agent install failed: {result.stderr.strip()[-500:]}"
            )
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
        # One live ACP process per trace keeps a single Prime Agent session, and
        # with it one IPython kernel, across turns. Prime Agent advertises
        # `loadSession: false` and refuses a second `session/new`, so a relaunch
        # per segment cannot preserve kernel state.
        #
        # ACP.session has no allow_empty_tool_reply option (unlike ACP.run).
        # The live handle owns turn requests directly, so do not invent or pass
        # that one-shot runner kwarg here.
        # This live path exists only because ACP sessions are currently
        # client-owned: the worker stops when the client disconnects. When Prime
        # Agent gains a resident ACP lifecycle, deleting this override restores
        # the plain per-segment shape.
        if not runtime.supports_live_processes:
            raise RuntimeError(
                "prime-agent harness requires runtime live-process support: "
                "without it, each segment would launch a fresh ACP process and "
                "lose the persistent IPython kernel"
            )
        system_prompt, prompt = self.resolve_prompt(data)
        command = await self._prepare(ctx, trace, runtime, endpoint, system_prompt)
        return PRIME_AGENT_ACP.session(
            self,
            ctx,
            trace,
            runtime,
            endpoint,
            secret,
            mcp_urls,
            data,
            env=self._run_env(trace, secret),
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
        command = await self._prepare(ctx, trace, runtime, endpoint, system_prompt)
        try:
            return await PRIME_AGENT_ACP.run(
                runtime,
                self._run_env(trace, secret),
                command,
                prompt,
                allow_empty_tool_reply=True,
            )
        except Exception as error:
            # ACP reports a daemon that never answers as an opaque 30s "create"
            # timeout, and the daemon log dies with the sandbox. Attach it so the
            # failure is diagnosable from CI output alone.
            tail = await self.daemon_log_tail(runtime, trace)
            if tail:
                # A typed rollout error already carries the authoritative
                # attribution (for example SandboxError). Do not replace it
                # with a generic RuntimeError merely to attach diagnostics.
                if isinstance(error, RolloutError):
                    raise
                raise RuntimeError(
                    f"{error}\n\nprime-agent daemon log:\n{tail}"
                ) from error
            raise

    def _run_env(self, trace: Trace, secret: str) -> dict[str, str]:
        root = self.trace_root(trace)
        return {
            **self.config.resolved_env,
            KEY_VAR: secret,
            ENV_AGENT_DIR: f"{root}/agent",
            "HOME": f"{root}/agent",
            # The daemon derives its socket from TMPDIR
            # (defaultDaemonSocketDir joins tmpdir with prime-agent-<uid>), so a
            # per-trace TMPDIR already isolates the socket without naming a path.
            "TMPDIR": self.tmp_dir(trace),
            # The ACP launch wrapper must resolve uv for daemon workers too;
            # setup()'s subprocess PATH cannot modify this later environment.
            "PATH": f"{self.install_dir()}.uv/bin:{self.node_root()}/bin:"
            + self.config.resolved_env.get(
                "PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
            ),
            "PRIME_AGENT_TELEMETRY": "0",
        }

    async def _prepare(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        system_prompt: str | None,
    ) -> list[str]:
        root = self.trace_root(trace)
        agent_dir = f"{root}/agent"
        created = await runtime.run(
            ["mkdir", "-p", "-m", "700", root, agent_dir, self.tmp_dir(trace)], {}
        )
        if created.exit_code != 0:
            raise RuntimeError(
                f"prime-agent state directory failed: {created.stderr.strip()[-500:]}"
            )

        skills_dir = f"{agent_dir}/skills"
        if self.config.skills:
            await self.install_skills(runtime, skills_dir)

        thinking = self.config.thinking
        reasoning = thinking not in (None, "off") or ctx.model.rsplit("/", 1)[
            -1
        ].startswith(("gpt-5", "o1", "o3", "o4"))
        models = {
            "providers": {
                PROVIDER: {
                    "baseUrl": endpoint,
                    "api": "openai-completions",
                    # The variable name, not the secret. Never interpolate
                    # untrusted text here: a leading "!" is executed as a command.
                    "apiKey": KEY_VAR,
                    "models": [
                        {
                            "id": ctx.model,
                            "reasoning": reasoning,
                            "input": ["text", "image"],
                            **(
                                {"maxTokens": ctx.sampling.max_tokens}
                                if ctx.sampling.max_tokens is not None
                                else {}
                            ),
                        }
                    ],
                }
            }
        }
        models_path = f"{agent_dir}/models.json"
        await runtime.write(models_path, json.dumps(models).encode())
        # The endpoint the agent must reach is rewritten by the runtime
        # (127.0.0.1 becomes a proxy-only alias under egress restriction), and a
        # rollout that cannot reach it fails as an opaque provider error. Log it
        # so a failure names the address that was actually configured.
        logger.info(
            "prime-agent: model endpoint for trace %s is %s", trace.id, endpoint
        )
        for command in (
            ["chmod", "700", root, agent_dir, self.tmp_dir(trace)],
            ["chmod", "600", models_path],
        ):
            restricted = await runtime.run(command, {})
            if restricted.exit_code != 0:
                raise RuntimeError(
                    f"prime-agent permissions failed: {restricted.stderr.strip()[-500:]}"
                )

        args = [
            self.prime_agent_bin(),
            "--mode",
            "acp",
            "--provider",
            PROVIDER,
            "--model",
            ctx.model,
            "--daemon-socket",
            self.trace_root(trace) + "/daemon.sock",
        ]
        if self.config.disabled_tools:
            args.append("--no-builtin-tools")
        if thinking is not None:
            args += ["--thinking", thinking]
        for skill in self.config.skills:
            args += ["--skill", f"{skills_dir}/{skill.resolve().name}"]
        if self.config.autonomous:
            args.append("--autonomous")
            for gate in self.config.gates:
                args += ["--autonomous-gate", gate]
        if system_prompt:
            # Sent once per launch as a flag. Folding it into the transcript
            # would re-apply it on every resumed segment.
            args += ["--append-system-prompt", system_prompt]

        # The bundled Node is beside the install, not on the image PATH.
        node_bin = f"{self.node_root()}/bin"
        command = (
            f'export PATH={shlex.quote(node_bin)}:"$PATH"\n'
            f'exec {shlex.join(args)} "$@"\n'
        )
        wrapper = f"{root}/prime-agent"
        await runtime.write(wrapper, f"#!/bin/sh\nset -eu\n{command}".encode())
        wrapper_mode = await runtime.run(["chmod", "700", wrapper], {})
        if wrapper_mode.exit_code != 0:
            raise RuntimeError(
                "prime-agent wrapper permissions failed: "
                f"{wrapper_mode.stderr.strip()[-500:]}"
            )

        return ["sh", "-c", f"exec {shlex.quote(wrapper)}"]

    async def daemon_log_tail(self, runtime: Runtime, trace: Trace) -> str:
        """Best-effort daemon log, for explaining an opaque ACP startup timeout."""
        result = await runtime.run(
            [
                "sh",
                "-c",
                f"tail -n 40 {shlex.quote(self.trace_root(trace))}/agent/logs/*.log 2>/dev/null || true",
            ],
            {},
        )
        return result.stdout.strip()[-1500:]

    async def cleanup(self, trace: Trace, runtime: Runtime) -> None:
        root = self.trace_root(trace)
        socket = f"{root}/daemon.sock"
        try:
            # Stop this trace's daemon before deleting its state: a live worker
            # would keep writing into a removed directory. `daemon` is in the
            # CLI's REMOVED_COMMAND_NAMES, so the subcommand is plain `stop`.
            stopped = await runtime.run(
                [
                    "sh",
                    "-c",
                    # Prepend the bundled Node inside the shell rather than passing
                    # PATH in env: `docker exec --env PATH=...` REPLACES the image
                    # PATH, and resolved_env usually has no PATH to fall back on.
                    (
                        f'export PATH={shlex.quote(f"{self.node_root()}/bin")}:"$PATH"\n'
                        f"exec {shlex.quote(self.prime_agent_bin())} stop "
                        f"--daemon-socket {shlex.quote(socket)}"
                    ),
                ],
                self._run_env(trace, ""),
            )
            if stopped.exit_code != 0:
                # Keep the state directory: a failed stop may leave a live
                # worker writing to it. Do not turn that failure into silent
                # data loss by deleting the directory.
                raise RuntimeError(
                    "prime-agent: stopping the trace daemon failed "
                    f"(exit {stopped.exit_code}): {stopped.stderr.strip()[-300:]}"
                )
        except Exception:
            logger.exception("prime-agent: daemon cleanup failed; retaining %s", root)
            raise
        else:
            # Remove only this trace's state, never the shared install.
            await runtime.run(["rm", "-rf", root], {})
