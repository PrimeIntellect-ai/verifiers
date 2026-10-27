"""Prime Agent over its native ACP mode."""

import hashlib
import json
import logging
import uuid
from functools import cached_property
from pathlib import Path
from typing import Literal

from verifiers.v1.acp import ACPConfig, ACPHarness, ACPTurn
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import DialectName, HarnessConfig, skill_destination
from verifiers.v1.harnesses.utils.install import ensure_installed, remove_dir
from verifiers.v1.harnesses.utils.replay import drop_blank_text
from verifiers.v1.runtimes import Runtime
from verifiers.v1.runtimes.base import _ENSURE_UV
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace
from verifiers.v1.types import Message

logger = logging.getLogger(__name__)

GITHUB_RELEASE_URL = (
    "https://github.com/PrimeIntellect-ai/prime-agent/releases/download"
)
PRIME_AGENT_COMMIT: Literal["8aac8692456c00c01d4e53e60e6d3347a3506559"] = (
    "8aac8692456c00c01d4e53e60e6d3347a3506559"
)
PRIME_AGENT_VERSION = "0.10.1"
# Release archive digests by Prime Agent platform name.
PRIME_AGENT_SHA256 = {
    "linux-x64": "21ec219183f4134dd23dce9572102e69456066fe374846525561c776d4d1945c",
    "linux-arm64": "5af68c10a088538fd4990fd18778d8c7fe28c0491083dc069c5177f733770e91",
    "darwin-x64": "9e2812084acead5a2c44209ffac350f762da67a27f4abcc5b417b7a6ef436632",
    "darwin-arm64": "2b47bd9cefa0b531151aaeb7295073ba0a16cf694177c18b4f9f74a54b24e279",
}
# `uname -sm` to Prime Agent platform name.
PLATFORMS = {
    "Linux x86_64": "linux-x64",
    "Linux aarch64": "linux-arm64",
    "Linux arm64": "linux-arm64",
    "Darwin x86_64": "darwin-x64",
    "Darwin arm64": "darwin-arm64",
}
PRIME_AGENT_DIR = "/var/tmp/vf-prime-agent"
STATE_ROOT = "/tmp/vf-prime-agent-runs"
PROVIDER = "intercept"
LIFECYCLE_META_NAMESPACE = "ai.primeintellect.prime-agent"
KEY_VAR = "PRIME_AGENT_INTERCEPT_KEY"
ENV_AGENT_DIR = "PRIME_AGENT_CODING_AGENT_DIR"


INSTALL = (
    "set -e\n"
    + _ENSURE_UV
    + r"""
prefix="$VF_PRIME_AGENT_DIR/$PRIME_AGENT_BUILD"
if [ ! -x "$prefix/bin/prime-agent" ]; then
    download_dir="$(mktemp -d "$VF_PRIME_AGENT_DIR/install.XXXXXX")"
    trap 'rm -rf "$download_dir"' EXIT
    archive="${PRIME_AGENT_ARCHIVE:-$download_dir/prime-agent.tar.gz}"
    if [ -z "${PRIME_AGENT_ARCHIVE:-}" ]; then
        command -v curl >/dev/null || { apt-get update -qq && apt-get install -y -qq curl ca-certificates; } || apk add --no-cache curl ca-certificates
        curl -fsSL --retry 5 --retry-all-errors "$PRIME_AGENT_ARCHIVE_URL" -o "$archive"
    fi
    printf '%s  %s\n' "$PRIME_AGENT_ARCHIVE_SHA256" "$archive" > "$download_dir/SHA256SUMS"
    if command -v sha256sum >/dev/null; then
        sha256sum -c "$download_dir/SHA256SUMS"
    else
        shasum -a 256 -c "$download_dir/SHA256SUMS"
    fi
    mkdir "$download_dir/bin"
    tar -xzf "$archive" -C "$download_dir/bin"
    if [ -n "${PRIME_AGENT_RELEASE_VERSION:-}" ]; then
        reported="$("$download_dir/bin/prime-agent" --version)"
        [ "$reported" = "$PRIME_AGENT_RELEASE_VERSION" ] || {
            echo "prime-agent reports version $reported, expected $PRIME_AGENT_RELEASE_VERSION" >&2
            exit 1
        }
    fi
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
"""
)


class PrimeAgentHarnessConfig(HarnessConfig):
    commit: Literal["8aac8692456c00c01d4e53e60e6d3347a3506559"] = PRIME_AGENT_COMMIT
    """Prime Agent release commit to install."""

    release: Path | None = None
    """A locally built Prime Agent release to install instead of `commit`: the directory
    holding its `manifest.json`, such as the `target/release/dist` that
    `make continuous-dry-run` writes."""

    autonomous: bool = False
    """Enable Prime Agent's autonomous continuation loop."""
    # Over Responses it replays an earlier turn without its reasoning.
    dialect: Literal["chat_completions", "anthropic_messages"] = "chat_completions"


class PrimeAgentHarness(ACPHarness[PrimeAgentHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_SKILLS = True

    # Its chat-completions provider drops whitespace-only assistant text on replay. Recheck on bumps:
    # https://github.com/PrimeIntellect-ai/prime-agent/blob/v0.10.1/crates/pa-ai/src/providers/openai_completions/convert.rs#L106
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

    async def setup(self, runtime: Runtime) -> None:
        logger.info("prime-agent: ensuring build %s is installed", self._build)
        uname = await runtime.run(["uname", "-sm"], {})
        platform = PLATFORMS.get(uname.stdout.strip())
        if uname.exit_code != 0 or platform is None:
            raise RuntimeError(
                f"unsupported prime-agent platform: {uname.stdout.strip() or uname.stderr.strip()}"
            )
        archive: dict[str, str] = {}
        upload: tuple[str, Path] | None = None
        root = runtime.with_user(None)
        if (local := self._local_release) is None:
            version = PRIME_AGENT_VERSION
            archive["PRIME_AGENT_RELEASE_VERSION"] = version
            archive["PRIME_AGENT_ARCHIVE_SHA256"] = PRIME_AGENT_SHA256[platform]
            archive["PRIME_AGENT_ARCHIVE_URL"] = (
                f"{GITHUB_RELEASE_URL}/v{version}/prime-agent-{version}-{platform}.tar.gz"
            )
        else:
            release, manifest_bytes = local
            manifest = json.loads(manifest_bytes)
            entry = next(
                (b for b in manifest["binaries"] if b["platform"] == platform), None
            )
            if entry is None:
                raise RuntimeError(
                    f"prime-agent release {release} has no {platform} archive"
                )
            # The archive digest pins the build; the version stamp differs by release channel.
            archive["PRIME_AGENT_ARCHIVE_SHA256"] = entry["sha256"]
            if (await root.run(["test", "-x", self._bin()], {})).exit_code != 0:
                upload = (
                    f"{PRIME_AGENT_DIR}/upload.{uuid.uuid4().hex}",
                    release / entry["file"],
                )
                archive["PRIME_AGENT_ARCHIVE"] = f"{upload[0]}/{entry['file']}"
        try:
            if upload is not None:
                await root.write(
                    archive["PRIME_AGENT_ARCHIVE"], self._archive(upload[1])
                )
            await ensure_installed(
                runtime,
                directory=PRIME_AGENT_DIR,
                install=INSTALL,
                env={
                    **self.config.resolved_env,
                    **archive,
                    "VF_PRIME_AGENT_DIR": PRIME_AGENT_DIR,
                    "PRIME_AGENT_BUILD": self._build,
                    "PRIME_AGENT_KERNEL_VENV": self._kernel_venv(),
                },
                label="prime-agent",
            )
        finally:
            if upload is not None:
                await remove_dir(root, upload[0], "prime-agent upload")
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
                    # Its Anthropic provider appends `/v1/messages` to the base URL.
                    "baseUrl": endpoint.removesuffix("/v1")
                    if self.config.dialect == "anthropic_messages"
                    else endpoint,
                    "api": {
                        "chat_completions": "openai-completions",
                        "responses": "openai-responses",
                        "anthropic_messages": "anthropic-messages",
                    }[self.config.dialect],
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
            args += ["--skill", skill_destination(skill, skills_dir)]
        if system_prompt:
            args += ["--append-system-prompt", system_prompt]
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

        return ACPConfig(
            env=self._env(trace, secret),
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

    @cached_property
    def _local_release(self) -> tuple[Path, bytes] | None:
        """The configured local release and its `manifest.json`, read once: every rollout
        installs the build the install directory is named for, and a dist rebuilt mid-run
        fails its checksum instead of landing under the old name."""
        release = self.config.release
        return (
            None
            if release is None
            else (release, (release / "manifest.json").read_bytes())
        )

    @cached_property
    def _build(self) -> str:
        """The install directory name: the pinned commit, or a local release's manifest
        digest, which changes with every archive it lists."""
        if self._local_release is None:
            return self.config.commit
        return f"local-{hashlib.sha256(self._local_release[1]).hexdigest()[:16]}"

    @cached_property
    def _archives(self) -> dict[Path, bytes]:
        return {}

    def _archive(self, path: Path) -> bytes:
        """A local release archive, read once and shared by every rollout's upload."""
        if path not in self._archives:
            self._archives[path] = path.read_bytes()
        return self._archives[path]

    def _bin(self) -> str:
        return f"{PRIME_AGENT_DIR}/{self._build}/bin/prime-agent"

    def _kernel_venv(self) -> str:
        """One venv path for install and session, in the shared install dir so the
        agent user can read the venv the root run bootstrapped."""
        return f"{PRIME_AGENT_DIR}/{self._build}/kernel-venv"

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
