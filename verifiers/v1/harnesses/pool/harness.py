"""Run Poolside's native ACP server against interception."""

import json
import shlex

from verifiers.v1.acp import ACPConfig, ACPHarness
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig, PinnedVersion
from verifiers.v1.harnesses.utils.install import ensure_installed
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

POOL_DIR = "/tmp/vf-pool-{version}"
SKILLS_DIR = ".poolside/skills"
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
    version: PinnedVersion = "1.0.15"
    """Pool release to install, pinned for reproducibility."""


class PoolHarness(ACPHarness[PoolHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_SKILLS = True

    async def setup(self, runtime: Runtime) -> None:
        await self.install_skills(runtime, SKILLS_DIR)
        directory = POOL_DIR.format(version=self.config.version)
        binary = f"{directory}/pool"
        script = INSTALL.replace("{version}", self.config.version).replace(
            "{dir}", directory
        )
        await ensure_installed(
            runtime,
            directory=directory,
            ready=f"[ -x {binary} ]",
            install=script,
            env=self.config.resolved_env,
            label="Pool",
            shell=("bash", "-o", "pipefail", "-c"),
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
            (
                f'export HOME="$PWD/{pool_home}/home" '
                f'XDG_CONFIG_HOME="$PWD/{pool_home}/config" '
                f'XDG_STATE_HOME="$PWD/{pool_home}/state"; '
                f"exec {POOL_DIR.format(version=self.config.version)}/pool acp "
                f"--sandbox disabled --settings {settings}"
            ),
        ]
        return ACPConfig(
            env=env,
            command=command,
            prompt=prompt,
            mcp_urls=mcp_urls,
            system_prompt=system_prompt,
        )
