"""Run Poolside's native ACP server against interception."""

import json
import shlex

from pydantic import Field

from verifiers.v1.acp import ACP
from verifiers.v1.clients import ModelContext
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

POOL_DIR = "/tmp/vf-pool-{version}"
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

POOL_ACP = ACP()


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
        await POOL_ACP.setup(self, runtime)

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
        env = {
            **self.config.resolved_env,
            # Standalone provider mode sends this bearer and model to interception.
            "POOLSIDE_API_KEY": secret,
            "POOLSIDE_STANDALONE_BASE_URL": endpoint,
            "POOLSIDE_STANDALONE_MODEL": ctx.model,
        }
        pool_home = f".vf-pool/{trace.id}"
        # Values are Pool tool names such as `shell`, `read`, or `edit`.
        tools = {name: {"disabled": True} for name in self.config.disabled_tools or []}
        settings = shlex.quote(json.dumps({"tools": tools}))
        command = [
            "sh",
            "-c",
            f'export HOME="$PWD/{pool_home}/home" '
            f'XDG_CONFIG_HOME="$PWD/{pool_home}/config" '
            f'XDG_STATE_HOME="$PWD/{pool_home}/state"; '
            f"exec {POOL_DIR.format(version=self.config.version)}/pool acp "
            f"--sandbox disabled --settings {settings}",
        ]
        return await POOL_ACP.run(
            runtime,
            env,
            command,
            prompt,
            mcp_urls=mcp_urls,
            system_prompt=system_prompt,
            session_path=f"{pool_home}/acp-session",
        )
