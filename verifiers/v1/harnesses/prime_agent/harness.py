"""Prime Agent over its native ACP mode."""

import hashlib
import json
import logging
from typing import Literal

from verifiers.v1.acp import ACPConfig, ACPHarness, ACPTurn
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig, skill_destination
from verifiers.v1.harnesses.utils.install import ensure_installed, remove_dir
from verifiers.v1.runtimes import Runtime
from verifiers.v1.runtimes.base import _ENSURE_UV
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

GITHUB_RELEASE_URL = (
    "https://github.com/PrimeIntellect-ai/prime-agent/releases/download"
)
PRIME_AGENT_COMMIT: Literal["763094e1b6a7c17450ee8399b06cda1853847ca5"] = (
    "763094e1b6a7c17450ee8399b06cda1853847ca5"
)
PRIME_AGENT_VERSION = "0.10.0"
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
prefix="$VF_PRIME_AGENT_DIR/$PRIME_AGENT_COMMIT"
if [ ! -x "$prefix/bin/prime-agent" ]; then
    case "$(uname -s)-$(uname -m)" in
        Linux-x86_64) platform=linux-x64; sha=c16bd2af5e77b53f49b914a44742c4cf6a67c5ed5000041430b78dbd4c3bcbec ;;
        Linux-aarch64|Linux-arm64) platform=linux-arm64; sha=f3cab3530a4d7ca43dbef8321bf13f260d1ba05d8b5dde057165a20bd4f675c4 ;;
        Darwin-x86_64) platform=darwin-x64; sha=af4866b5ba82f3419b964290e3f023ddb9b99958a89e16ee856962b901ec0fc6 ;;
        Darwin-arm64) platform=darwin-arm64; sha=e418bdf62fcb0002777bf3ac43bcc136512b26763f2ab5c500792f26e37294e5 ;;
        *) echo "unsupported prime-agent platform: $(uname -s)-$(uname -m)" >&2; exit 1 ;;
    esac
    command -v curl >/dev/null || (apt-get update -qq && apt-get install -y -qq curl ca-certificates >/dev/null)
    release_url="$VF_PRIME_AGENT_GITHUB_RELEASE_URL/v$PRIME_AGENT_RELEASE_VERSION"
    tarball="prime-agent-$PRIME_AGENT_RELEASE_VERSION-$platform.tar.gz"
    download_dir="$(mktemp -d "$VF_PRIME_AGENT_DIR/install.XXXXXX")"
    trap 'rm -rf "$download_dir"' EXIT
    curl -fsSL --retry 5 --retry-all-errors "$release_url/$tarball" -o "$download_dir/$tarball"
    printf '%s  %s\n' "$sha" "$tarball" > "$download_dir/SHA256SUMS"
    if command -v sha256sum >/dev/null; then
        (cd "$download_dir" && sha256sum -c SHA256SUMS)
    else
        (cd "$download_dir" && shasum -a 256 -c SHA256SUMS)
    fi
    mkdir "$download_dir/bin"
    tar -xzf "$download_dir/$tarball" -C "$download_dir/bin"
    [ "$("$download_dir/bin/prime-agent" --version)" = "$PRIME_AGENT_RELEASE_VERSION" ]
    mkdir -p "$prefix"
    mv "$download_dir/bin" "$prefix/bin"
fi
# Let the released binary validate the kernel cache and prepare its Python runtime.
"$prefix/bin/prime-agent" --prime-agent-bootstrap
"""
)


class PrimeAgentHarnessConfig(HarnessConfig):
    commit: Literal["763094e1b6a7c17450ee8399b06cda1853847ca5"] = PRIME_AGENT_COMMIT
    """Prime Agent release commit to install."""

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
        logger.info("prime-agent: ensuring commit %s is installed", self.config.commit)
        await ensure_installed(
            runtime,
            directory=PRIME_AGENT_DIR,
            install=INSTALL,
            env={
                **self.config.resolved_env,
                "VF_PRIME_AGENT_DIR": PRIME_AGENT_DIR,
                "VF_PRIME_AGENT_GITHUB_RELEASE_URL": GITHUB_RELEASE_URL,
                "PRIME_AGENT_COMMIT": self.config.commit,
                "PRIME_AGENT_RELEASE_VERSION": PRIME_AGENT_VERSION,
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
                "surface is a Python REPL"
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

    def _bin(self) -> str:
        return f"{PRIME_AGENT_DIR}/{self.config.commit}/bin/prime-agent"

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
