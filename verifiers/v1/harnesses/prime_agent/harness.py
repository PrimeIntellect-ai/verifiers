"""Prime Agent over its native ACP mode."""

import hashlib
import json
import logging
import shlex

from pydantic import Field

from verifiers.v1.acp import ACPConfig, ACPHarness, ACPTurnResult
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harnesses.node import NODE_BIN_DIR, ensure_node
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

INSTALL_URL = "https://pub-728493de92a943e2a9b2d17b4719f318.r2.dev/install.sh"
PRIME_AGENT_DIR = "/var/tmp/vf-prime-agent"
STATE_ROOT = "/tmp/vf-prime-agent-runs"
SKILLS_DIR = ".agents/skills"
PROVIDER = "intercept"
LIFECYCLE_META_NAMESPACE = "ai.primeintellect.prime-agent"
KEY_VAR = "PRIME_AGENT_INTERCEPT_KEY"
ENV_AGENT_DIR = "PRIME_AGENT_CODING_AGENT_DIR"


INSTALL = r"""
set -e
export PATH="/var/tmp/vf-node/bin:$PATH"
prefix="$VF_PRIME_AGENT_DIR/$PRIME_AGENT_VERSION"
[ -x "$prefix/bin/prime-agent" ] && exit 0
export NPM_CONFIG_PREFIX="$prefix"
export PRIME_AGENT_BOOTSTRAP_KERNEL_ON_INSTALL=0
installer="$(mktemp "$VF_PRIME_AGENT_DIR/install.XXXXXX")"
trap 'rm -f "$installer"' EXIT
attempt=1
max_attempts=10
while ! { curl -fsSL "$VF_PRIME_AGENT_INSTALL_URL" -o "$installer" && sh "$installer"; }; do
    [ "$attempt" -ge "$max_attempts" ] && exit 1
    jitter="$(od -An -N2 -tu2 /dev/urandom | tr -d ' ')"
    delay=$((attempt * 5 + jitter % 30))
    printf 'Prime Agent install failed; retrying in %ss (%s/%s)\n' \
        "$delay" "$attempt" "$max_attempts" >&2
    sleep "$delay"
    attempt=$((attempt + 1))
done
"""


class PrimeAgentHarnessConfig(HarnessConfig):
    version: str = Field(
        default="0.8.0-beta.548.1.9bc0055",
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._+-]*$",
    )
    """Prime Agent release to install, pinned for reproducibility."""

    autonomous: bool = False
    """Enable Prime Agent's autonomous continuation loop."""


class PrimeAgentHarness(ACPHarness[PrimeAgentHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_RESUME = True
    SUPPORTS_SKILLS = True

    def acp_turn_result(self, trace: Trace, result: ACPTurnResult) -> None:
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
        await self.install_skills(runtime, SKILLS_DIR)
        await ensure_node(runtime)
        logger.info("prime-agent: ensuring %s is installed", self.config.version)
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
                "VF_PRIME_AGENT_INSTALL_URL": INSTALL_URL,
                "PRIME_AGENT_VERSION": self.config.version,
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
        return f"{PRIME_AGENT_DIR}/{self.config.version}/bin/prime-agent"

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
