"""Run Poolside's `pool exec` against interception as an OpenAI-compatible provider."""

import json
import shlex

from pydantic import Field

from verifiers.v1.clients import ModelContext
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace
from verifiers.v1.types import SystemMessage, TextContentPart, UserMessage

POOL_DIR = "/tmp/vf-pool-{version}"
SETTINGS_PATH = ".poolside/settings.local.yaml"
INSTALL = r"""
set -e
command -v curl >/dev/null || (apt-get update -qq && apt-get install -y -qq curl ca-certificates >/dev/null)
command -v tar >/dev/null || (apt-get update -qq && apt-get install -y -qq tar >/dev/null)
case "$(uname -s)" in Linux) os=linux ;; Darwin) os=darwin ;; *) echo "unsupported os: $(uname -s)" >&2; exit 1 ;; esac
case "$(uname -m)" in aarch64|arm64) arch=arm64 ;; x86_64|amd64) arch=amd64 ;; *) echo "unsupported arch: $(uname -m)" >&2; exit 1 ;; esac
mkdir -p {dir}
curl -fsSL "https://github.com/poolsideai/pool/releases/download/v{version}/pool-$os-$arch.tar.gz" | tar -xz -C {dir}
mv "{dir}/pool-$os-$arch" "{dir}/pool"
chmod +x "{dir}/pool"
"""


class PoolHarnessConfig(HarnessConfig):
    version: str = Field(default="1.0.11", pattern=r"^[A-Za-z0-9._+-]+$")
    """Pool release to install, pinned for reproducibility."""


class PoolHarness(Harness[PoolHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_MESSAGE_PROMPT = True

    async def setup(self, runtime: Runtime) -> None:
        directory = POOL_DIR.format(version=self.config.version)
        binary = f"{directory}/pool"
        script = INSTALL.replace("{version}", self.config.version).replace(
            "{dir}", directory
        )
        ensure = shlex.quote(f"[ -x {binary} ] || ({script})")
        # Cache the pinned binary across local rollouts; Linux has flock, macOS has lockf.
        guarded = (
            f"mkdir -p {directory} && "
            f'"$(command -v flock || command -v lockf)" {directory}/install.lock '
            f"bash -o pipefail -c {ensure}"
        )
        result = await runtime.run(["sh", "-c", guarded], self.config.resolved_env)
        if result.exit_code != 0:
            detail = (result.stderr or result.stdout).strip()[-500:]
            raise RuntimeError(f"Pool install failed: {detail}")

    async def launch(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> ProgramResult:
        system_prompt, prompt = self.resolve_prompt(trace.task.data)
        if prompt is None:
            raise ValueError("Pool requires a task prompt (it has no user simulator)")
        texts = [system_prompt] if system_prompt else []
        if isinstance(prompt, str):
            texts.append(prompt)
        else:
            for message in prompt:
                if not isinstance(message, (SystemMessage, UserMessage)):
                    raise ValueError(
                        "pool exec only supports system and user initial messages"
                    )
                parts = (
                    [TextContentPart(text=message.content)]
                    if isinstance(message.content, str)
                    else message.content
                )
                for part in parts:
                    if not isinstance(part, TextContentPart):
                        raise ValueError("pool exec does not support image prompts")
                    texts.append(part.text)
        text = "\n\n".join(text for text in texts if text)

        settings = {
            "mcp_servers": {
                name: {"transport": {"type": "http", "url": url}}
                for name, url in mcp_urls.items()
            },
            # Values are Pool tool names such as `shell`, `read`, or `edit`.
            "tools": {
                name: {"disabled": True} for name in self.config.disabled_tools or []
            },
        }
        await runtime.write(SETTINGS_PATH, json.dumps(settings).encode())

        prompt_path = f".vf-pool/prompt-{trace.id}.txt"
        await runtime.write(prompt_path, text.encode())
        env = {
            **self.config.resolved_env,
            # Standalone provider mode sends this bearer and model to interception.
            "POOLSIDE_API_KEY": secret,
            "POOLSIDE_STANDALONE_BASE_URL": endpoint,
            "POOLSIDE_STANDALONE_MODEL": ctx.model,
        }
        directory = POOL_DIR.format(version=self.config.version)
        argv = [
            f"{directory}/pool",
            "exec",
            "--unsafe-auto-allow",
            "--sandbox",
            "disabled",
            "--prompt-file",
            prompt_path,
        ]
        # Keep Pool's global config, logs, and trajectories inside the rollout workspace.
        isolate = (
            'export XDG_CONFIG_HOME="$PWD/.vf-pool/config" '
            'XDG_STATE_HOME="$PWD/.vf-pool/state"; exec "$@"'
        )
        return await runtime.run_program(["sh", "-c", isolate, "pool", *argv], env)
