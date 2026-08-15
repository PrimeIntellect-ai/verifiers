"""Run Grok Build's native ACP server against interception."""

import json
import logging
import shlex

import tomli_w
from pydantic import Field

from verifiers.v1.acp import ACPConfig, ACPHarness
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

GROK_DIR = "/var/tmp/vf-grok-build"
BINARY = f"{GROK_DIR}/bin/grok"
KEY_VAR = "VF_GROK_INTERCEPT_KEY"
MODEL_ALIAS = "verifiers"
SUMMARY_MODEL_ALIAS = "verifiers-session-summary"
# Grok's Responses watchdog intentionally ignores lifecycle and empty-delta events. Verifiers
# owns the rollout deadline and remote sandboxes live at most 24 hours, so keep Grok's independent
# content watchdog just beyond that bound instead of fabricating model tokens to satisfy it.
INFERENCE_IDLE_TIMEOUT_SECONDS = 25 * 60 * 60

INSTALL = r"""
set -e
root=/var/tmp/vf-grok-build
bin="$root/bin/grok"
version_file="$root/.version"
if [ -x "$bin" ] && [ "$(cat "$version_file" 2>/dev/null)" = "$VF_GROK_BUILD_VERSION" ]; then
    exit 0
fi
if ! command -v bash >/dev/null || ! command -v curl >/dev/null; then
    if command -v apk >/dev/null; then
        apk add --no-cache bash curl ca-certificates >/dev/null
    elif command -v apt-get >/dev/null; then
        apt-get update -qq
        apt-get install -y -qq bash curl ca-certificates >/dev/null
    else
        echo "Grok Build installation requires bash and curl" >&2
        exit 1
    fi
fi
installer="$root/install.sh"
curl -fsSL https://x.ai/cli/install.sh -o "$installer"
mkdir -p "$root/install-home"
env \
    HOME="$root/install-home" \
    GROK_BIN_DIR="$root/bin" \
    GROK_DISABLE_AUTOUPDATER=1 \
    SHELL= \
    bash "$installer" "$VF_GROK_BUILD_VERSION"
test -x "$bin"
printf %s "$VF_GROK_BUILD_VERSION" > "$version_file"
"""


class GrokBuildHarnessConfig(HarnessConfig):
    version: str = Field(
        default="1.0.3",
        pattern=r"^[0-9]+\.[0-9]+\.[0-9]+(?:-[A-Za-z0-9._]+)?$",
    )
    """Grok Build release to install, pinned for reproducibility."""


class GrokBuildHarness(ACPHarness[GrokBuildHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_SKILLS = True

    async def setup(self, runtime: Runtime) -> None:
        logger.info(
            "grok-build: ensuring Grok Build %s is installed", self.config.version
        )
        lock = f"{GROK_DIR}/install.lock"
        guarded = (
            f"mkdir -p {GROK_DIR} && "
            f'until ln -s "$$" {lock} 2>/dev/null; do '
            f"owner=$(readlink {lock}); "
            f'if ! kill -0 "$owner" 2>/dev/null; then '
            f'[ "$(readlink {lock})" != "$owner" ] || rm -f {lock}; fi; '
            f"sleep 0.1; done; "
            f'trap \'[ "$(readlink {lock})" != "$$" ] || rm -f {lock}\' EXIT; '
            f"sh -c {shlex.quote(INSTALL)}"
        )
        install = await runtime.run(
            ["sh", "-c", guarded],
            {"VF_GROK_BUILD_VERSION": self.config.version},
        )
        if install.exit_code != 0:
            detail = (install.stderr or install.stdout).strip()[-500:]
            raise RuntimeError(f"Grok Build install failed: {detail}")
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
        grok_home = self.grok_home(trace)
        await self.install_skills(runtime, f"{grok_home}/skills")

        model = {
            "model": ctx.model,
            "base_url": endpoint,
            "name": ctx.model,
            "env_key": KEY_VAR,
            "api_backend": "responses",
            "inference_idle_timeout_secs": INFERENCE_IDLE_TIMEOUT_SECONDS,
            "supports_backend_search": False,
        }
        if ctx.sampling.temperature is not None:
            model["temperature"] = ctx.sampling.temperature
        if ctx.sampling.top_p is not None:
            model["top_p"] = ctx.sampling.top_p
        if ctx.sampling.max_tokens is not None:
            model["max_completion_tokens"] = ctx.sampling.max_tokens
        config = {
            "cli": {"auto_update": False, "use_leader": False},
            "features": {"telemetry": False, "feedback": False},
            "memory": {"enabled": False},
            "models": {
                "default": MODEL_ALIAS,
                "web_search": MODEL_ALIAS,
                # Grok has no switch for UI session-title inference. Keep that
                # auxiliary request out of the rollout trace; Grok falls back to
                # a title derived locally from the first user message.
                "session_summary": SUMMARY_MODEL_ALIAS,
                "image_description": MODEL_ALIAS,
                "prompt_suggestion": MODEL_ALIAS,
                "allowed_models": [MODEL_ALIAS],
            },
            "model": {
                MODEL_ALIAS: model,
                SUMMARY_MODEL_ALIAS: {
                    "model": SUMMARY_MODEL_ALIAS,
                    "base_url": "http://127.0.0.1:1",
                    "name": "Disabled session summary",
                    "env_key": KEY_VAR,
                    "api_backend": "responses",
                },
            },
            # An explicitly selected subagent model could leave interception; keep
            # Grok's native child-agent surface off in the eval harness.
            "subagents": {"enabled": False},
        }
        await runtime.write(f"{grok_home}/config.toml", tomli_w.dumps(config).encode())

        profile_args: list[str] = []
        if self.config.disabled_tools:
            profile = (
                "---\n"
                "name: verifiers\n"
                "description: Grok Build running under Verifiers\n"
                f"disallowedTools: {json.dumps(self.config.disabled_tools)}\n"
                "---\n"
            )
            profile_path = f"{grok_home}/agent.md"
            await runtime.write(profile_path, profile.encode())
            profile_args = ["--agent-profile", profile_path]

        system_prompt, prompt = self.resolve_prompt(data)
        env = {
            **self.config.resolved_env,
            "HOME": grok_home,
            "USERPROFILE": grok_home,
            "GIT_CONFIG_GLOBAL": f"{grok_home}/gitconfig",
            "GROK_HOME": grok_home,
            KEY_VAR: secret,
            # Prevent ambient first-party credentials from selecting an xAI endpoint.
            "XAI_API_KEY": "",
            "GROK_CODE_XAI_API_KEY": "",
            "GROK_DEPLOYMENT_KEY": "",
            "GROK_SUBAGENTS": "0",
            "GROK_MEMORY": "0",
            "GROK_MANAGED_MCPS_ENABLED": "false",
            "GROK_MANAGED_MCP_GATEWAY_TOOLS_ENABLED": "false",
            "GROK_TELEMETRY_ENABLED": "false",
            "GROK_TELEMETRY_TRACE_UPLOAD": "false",
            "GROK_FEEDBACK_ENABLED": "false",
            "GROK_TRACE_UPLOAD": "false",
            "GROK_INSTRUMENTATION": "disabled",
            "GROK_DISABLE_AUTOUPDATER": "1",
            "GROK_PROMPT_SUGGESTIONS": "false",
            "GROK_TURN_SUMMARY": "0",
            "OTEL_SDK_DISABLED": "true",
            "DISABLE_TELEMETRY": "1",
            "DISABLE_FEEDBACK_COMMAND": "1",
        }
        command = [
            BINARY,
            "--no-auto-update",
            "--disable-web-search",
            "agent",
            "--no-leader",
            "--always-approve",
            "--model",
            MODEL_ALIAS,
            *profile_args,
            "stdio",
        ]
        return ACPConfig(
            env=env,
            command=command,
            prompt=prompt,
            system_prompt=system_prompt,
        )

    async def cleanup(self, trace: Trace, runtime: Runtime) -> None:
        result = await runtime.run(["rm", "-rf", self.grok_home(trace)], {})
        if result.exit_code != 0:
            raise RuntimeError(
                f"failed to clean up Grok home: {result.stderr.strip()[-500:]}"
            )

    @staticmethod
    def grok_home(trace: Trace) -> str:
        return f".vf-grok-build/{trace.id}"
