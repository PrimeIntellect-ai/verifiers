"""Prime Agent over its native ACP mode."""

import hashlib
import json
import shlex
from typing import Literal

from verifiers.v1.acp import ACPConfig, ACPHarness, ACPTurn
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import DialectName, HarnessConfig, skill_destination
from verifiers.v1.harnesses.utils.install import ensure_installed, remove_dir
from verifiers.v1.harnesses.utils.remote_runtime import prepare_remote_bridge
from verifiers.v1.harnesses.utils.replay import drop_blank_text
from verifiers.v1.runtimes import Runtime
from verifiers.v1.runtimes.base import _ENSURE_UV
from verifiers.v1.runtimes.deployment import ExecutionConnection
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace
from verifiers.v1.types import Message
from verifiers.v1.utils.logging import get_logger

logger = get_logger(__name__)

GITHUB_RELEASE_URL = (
    "https://github.com/PrimeIntellect-ai/prime-agent/releases/download"
)
PRIME_AGENT_COMMIT: Literal["033eea1705c5ae953fe7cd7a5e7e7c45317e0dfe"] = (
    "033eea1705c5ae953fe7cd7a5e7e7c45317e0dfe"
)
PRIME_AGENT_VERSION = "0.10.5"
PRIME_AGENT_DIR = "/var/tmp/vf-prime-agent"
STATE_ROOT = "/tmp/vf-prime-agent-runs"
LIFECYCLE_META_NAMESPACE = "ai.primeintellect.prime-agent"
KEY_VAR = "PRIME_AGENT_INTERCEPT_KEY"
ENV_AGENT_DIR = "PRIME_AGENT_CODING_AGENT_DIR"


INSTALL = (
    "set -e\n"
    + _ENSURE_UV
    + r"""
prefix="$AGENT_PRIME_AGENT_DIR/$PRIME_AGENT_COMMIT"
rm -f "$prefix/.ready"
if [ ! -x "$prefix/bin/prime-agent" ]; then
    case "$(uname -s)-$(uname -m)" in
        Linux-x86_64) platform=linux-x64; sha=b9f653f4c32e83a2270f0b52edea6a72c7bdbd1f7b5c3e64c55e75157094f3b3 ;;
        Linux-aarch64|Linux-arm64) platform=linux-arm64; sha=8ddee5f1ed1a4443829944117509212f1294abbf0cbed22765f8a6efefaa2516 ;;
        Darwin-x86_64) platform=darwin-x64; sha=2a19423b21e79ac45050a246a18d507db66801072437e321b1f7063c6bcf9a4b ;;
        Darwin-arm64) platform=darwin-arm64; sha=6eb082c66f5259b953c38f21c72055d11942fa61372718a5eab25603247bd49c ;;
        *) echo "unsupported prime-agent platform: $(uname -s)-$(uname -m)" >&2; exit 1 ;;
    esac
    release_url="$AGENT_PRIME_AGENT_GITHUB_RELEASE_URL/v$PRIME_AGENT_RELEASE_VERSION"
    tarball="prime-agent-$PRIME_AGENT_RELEASE_VERSION-$platform.tar.gz"
    download_dir="$(mktemp -d "$AGENT_PRIME_AGENT_DIR/install.XXXXXX")"
    trap 'rm -rf "$download_dir"' EXIT
    mkdir "$download_dir/bin"
    $(command -v python3 || command -v python) -c '
import hashlib, io, sys, tarfile, urllib.request
req = urllib.request.Request(sys.argv[1], headers={"User-Agent": "verifiers"})
data = urllib.request.urlopen(req, timeout=60).read()
assert hashlib.sha256(data).hexdigest() == sys.argv[2], "prime-agent checksum mismatch"
with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
    tf.extractall(sys.argv[3])
' "$release_url/$tarball" "$sha" "$download_dir/bin"
    [ "$("$download_dir/bin/prime-agent" --version)" = "$PRIME_AGENT_RELEASE_VERSION" ]
    mkdir -p "$prefix"
    mv "$download_dir/bin" "$prefix/bin"
fi
# Let the released binary validate the kernel cache and prepare its Python runtime.
"$prefix/bin/prime-agent" --prime-agent-bootstrap
# Bootstrap installs only the runtime; sessions also import the bundled Python skills.
set --
for pyproject in "$prefix/bin/skills/"*/pyproject.toml; do
    set -- "$@" --editable "${pyproject%/pyproject.toml}"
done
uv pip install --python "$PRIME_AGENT_KERNEL_VENV/bin/python" "$@"
chmod -R a+rX "$AGENT_PRIME_AGENT_DIR"
touch "$prefix/.ready"
"""
)


class PrimeAgentHarnessConfig(HarnessConfig):
    commit: Literal["033eea1705c5ae953fe7cd7a5e7e7c45317e0dfe"] = PRIME_AGENT_COMMIT
    """Prime Agent release commit to install."""

    autonomous: bool = False
    """Enable Prime Agent's autonomous continuation loop."""
    api: str | None = None
    """The API Prime Agent speaks to the model's provider, e.g. `openai-completions` for an
    OpenAI-compatible endpoint that serves only chat completions. Unset, Prime Agent picks the
    provider's native API (Responses for `openai`)."""
    dialect: Literal["auto"] = "auto"


class PrimeAgentHarness(ACPHarness[PrimeAgentHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_SKILLS = True

    async def remote_tools(self, runtime: Runtime, data: TaskData) -> None:
        return None

    # Its chat-completions provider drops whitespace-only assistant text on replay. Recheck on bumps:
    # https://github.com/PrimeIntellect-ai/prime-agent/blob/v0.10.5/crates/pa-ai/src/providers/openai_completions/convert.rs#L106
    def replay_view(self, message: Message, dialect: DialectName | None) -> Message:
        return drop_blank_text(message)

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

    async def setup_execution(
        self, runtime: Runtime, workspace: Runtime, connection: ExecutionConnection
    ) -> None:
        await self._install(connection.runtime, {})
        connection.register(
            "prime_agent",
            [f"{self._kernel_venv()}/bin/python", "-m", "rlm.repl"],
            {},
        )

    async def setup(self, runtime: Runtime) -> None:
        await self._install(runtime, self.config.resolved_env)
        await super().setup(runtime)

    async def _install(self, runtime: Runtime, env: dict[str, str]) -> None:
        logger.debug("Ensuring prime-agent %s is installed", self.config.commit)
        ready = f"{PRIME_AGENT_DIR}/{self.config.commit}/.ready"
        await ensure_installed(
            runtime,
            directory=PRIME_AGENT_DIR,
            # A finished install skips the script: its skills install resolves against
            # the package index on every run, which a runtime reused after its network
            # is restricted can no longer reach.
            ready=(
                f"[ -f {ready} ] && [ -x {self._bin()} ] "
                f"&& [ -x {self._kernel_venv()}/bin/python ]"
            ),
            install=INSTALL,
            env={
                **env,
                "AGENT_PRIME_AGENT_DIR": PRIME_AGENT_DIR,
                "AGENT_PRIME_AGENT_GITHUB_RELEASE_URL": GITHUB_RELEASE_URL,
                "PRIME_AGENT_COMMIT": self.config.commit,
                "PRIME_AGENT_RELEASE_VERSION": PRIME_AGENT_VERSION,
                "PRIME_AGENT_KERNEL_VENV": self._kernel_venv(),
            },
            label="prime-agent",
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
        provider, separator, model = ctx.model.partition("/")
        if not separator:
            provider, model = "openai", ctx.model
        elif provider in ("internal", "dev"):
            provider, model = "prime-inference", ctx.model
        models = {
            "providers": {
                provider: {
                    # The native provider owns its API path, including any version prefix.
                    "baseUrl": endpoint.removesuffix("/v1"),
                    "apiKey": KEY_VAR,
                    **({"api": self.config.api} if self.config.api else {}),
                    "models": [
                        {
                            "id": model,
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
            "--no-session",  # The ACP connection owns the session's lifetime.
            "--provider",
            provider,
            "--model",
            model,
            "--daemon-socket",
            f"{root}/daemon.sock",
            "--offline",
        ]
        if self.config.autonomous:
            args.append("--autonomous")
        for skill in self.config.skills:
            args += ["--skill", skill_destination(skill, skills_dir)]
        if system_prompt:
            # prime-agent reads a path argument as the file's contents when it
            # builds a session, so the file stays until cleanup.
            path = f"{agent_dir}/system-prompt.md"
            await runtime.write(path, system_prompt.encode())
            args += ["--append-system-prompt", path]
        if mcp_urls:
            # ACP servers need discovery guidance; importing the SDK shadows the REPL helper.
            args += [
                "--append-system-prompt",
                (
                    f"Connected MCP servers: {json.dumps(list(mcp_urls))}. "
                    "Use the pre-imported `mcp` helper in `ipython`, or `from rlm import mcp`. "
                    "Do not `import mcp`: that loads the unrelated SDK and overwrites the helper. "
                    "Discover tools with `print(await mcp.list_tools(server))`, then call them "
                    "with `print(await mcp.call_tool(server, tool, arguments))`."
                ),
            ]

        env = self._env(trace, secret)
        cwd = None
        if runtime._execution_bridge is not None:
            cwd = runtime._execution_bridge.runtime.info.workdir
            # The supervisor launches its interpreter with the session cwd.
            # Keep only the directory in A; all task content lives in B.
            created = await runtime.run(["mkdir", "-p", cwd], {})
            if created.exit_code:
                raise RuntimeError("could not prepare Prime Agent supervisor cwd")
            command = await prepare_remote_bridge(
                runtime,
                f"{endpoint.removesuffix('/v1')}/execution",
                secret,
                "prime_agent",
                protocol="prime",
            )
            # The pinned supervisor uses this interpreter for trusted import
            # probes and starts every user cell with `-m rlm.repl`.
            # Only probes use A's installed Python; the persistent kernel is B.
            wrapper = f"{root}/kernel-python"
            source = (
                "#!/bin/sh\nset -eu\n"
                'prev=""\n'
                'for arg in "$@"; do\n'
                '  if [ "$prev" = "-m" ] && [ "$arg" = "rlm.repl" ]; then\n'
                f"    exec {shlex.join(command)}\n"
                "  fi\n"
                '  prev="$arg"\n'
                "done\n"
                f'exec {shlex.quote(self._kernel_venv() + "/bin/python")} "$@"\n'
            )
            await runtime.write(wrapper, source.encode())
            result = await runtime.run(["chmod", "700", wrapper], {})
            if result.exit_code:
                raise RuntimeError("could not install Prime Agent execution bridge")
            env["PRIME_AGENT_KERNEL_PYTHON"] = wrapper
        return ACPConfig(
            env=env,
            cwd=cwd,
            # Expand the sandbox's PATH while keeping every agent argument literal.
            command=[
                "/bin/sh",
                "-eu",
                "-c",
                'export PATH="$HOME/.local/bin:$PATH"; exec "$@"',
                "prime-agent",
                *args,
            ],
            prompt=prompt,
        )

    async def cleanup(self, trace: Trace, runtime: Runtime) -> None:
        root = self._root(trace)
        await remove_dir(runtime, root, "prime-agent state")

    def _bin(self) -> str:
        return f"{PRIME_AGENT_DIR}/{self.config.commit}/bin/prime-agent"

    def _kernel_venv(self) -> str:
        """One venv path for install and session, in the shared install dir so the
        agent user can read the venv the root run bootstrapped."""
        return f"{PRIME_AGENT_DIR}/{self.config.commit}/kernel-venv"

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
            # Sessions use the installed interpreter without writing to the shared venv.
            "PRIME_AGENT_KERNEL_PYTHON": f"{self._kernel_venv()}/bin/python",
            "PRIME_AGENT_TELEMETRY": "0",
        }
