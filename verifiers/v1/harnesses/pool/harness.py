"""Run Poolside's native ACP server against interception."""

import json
import shlex
import uuid
from pathlib import Path

from pydantic import Field

from verifiers.v1.acp import ACPConfig, ACPHarness
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.errors import HarnessError
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

POOL_DIR = "/tmp/vf-pool-{version}"
POOL_VERSION = "1.0.16"
SKILLS_DIR = ".poolside/skills"
HOOK_NAME = "verifiers-tool-interception"
HOOK_SOURCE = Path(__file__).with_name("tool_hook.py").read_text()
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
    version: str = Field(default=POOL_VERSION, pattern=r"^[A-Za-z0-9._+-]+$")
    """Pool release to install, pinned for reproducibility."""


class PoolHarness(ACPHarness[PoolHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_SKILLS = True
    SUPPORTS_PRE_TOOL_INTERCEPTION = True
    SUPPORTS_POST_TOOL_INTERCEPTION = True

    async def setup(self, runtime: Runtime) -> None:
        await self.install_skills(runtime, SKILLS_DIR)
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
        pool_home = f".vf-pool/{trace.id}"
        # Values are Pool tool names such as `shell`, `read`, or `edit`.
        tools = {name: {"disabled": True} for name in self.config.disabled_tools or []}
        settings = json.dumps({"tools": tools})
        env = {
            **self.config.resolved_env,
            # Standalone provider mode sends this bearer and model to interception.
            "POOLSIDE_API_KEY": secret,
            "POOLSIDE_STANDALONE_BASE_URL": endpoint,
            "POOLSIDE_STANDALONE_MODEL": ctx.model,
            "VF_POOL_SETTINGS": settings,
            "VF_POOL_STATE_DIR": pool_home,
        }
        command = [
            "sh",
            "-c",
            (
                f'export HOME="$PWD/{pool_home}/home" '
                f'XDG_CONFIG_HOME="$PWD/{pool_home}/config" '
                f'XDG_STATE_HOME="$PWD/{pool_home}/state"; '
                f"exec {POOL_DIR.format(version=self.config.version)}/pool acp "
                '--sandbox disabled --settings "$VF_POOL_SETTINGS"'
            ),
        ]
        return ACPConfig(
            env=env,
            command=command,
            prompt=prompt,
            mcp_urls=mcp_urls,
            system_prompt=system_prompt,
        )

    async def configure_tool_interception(
        self,
        config: ACPConfig,
        runtime: Runtime,
        url: str,
        secret: str,
    ) -> None:
        if self.config.version != POOL_VERSION:
            raise HarnessError(
                f"Pool tool interception is verified only for version {POOL_VERSION}"
            )

        stateDir = config.env["VF_POOL_STATE_DIR"]
        credentialsPath = f"{stateDir}/{uuid.uuid4().hex}.credentials"
        payload = json.dumps({"url": url, "secret": secret}).encode()
        result = await runtime.run_with_input(
            [
                "sh",
                "-c",
                'umask 077; mkdir -p "$1"; set -C; head -c "$2" > "$3"',
                "write-tool-credentials",
                stateDir,
                str(len(payload)),
                credentialsPath,
            ],
            {},
            payload,
        )
        if result.exit_code != 0:
            raise RuntimeError(
                "failed to write Pool interception credentials privately: "
                f"{result.stderr.strip()[-500:]}"
            )

        hookProgram = await runtime.prepare_uv_script(HOOK_SOURCE, activate=False)
        hook = {
            "name": HOOK_NAME,
            "matcher": "*",
            "command": shlex.join(["exec", *hookProgram, credentialsPath]),
            "timeout": 35,
        }
        settings = json.loads(config.env["VF_POOL_SETTINGS"])
        settings["hooks"] = {
            "PreToolUse": [hook],
            "PostToolUse": [hook],
        }
        config.env["VF_POOL_SETTINGS"] = json.dumps(settings)
        # Pool's closed ACP server cannot originate our metadata request, so its
        # synchronous command hooks call the rollout policy endpoint directly.
        config.toolInterception = None

    async def cleanup(self, trace: Trace, runtime: Runtime) -> None:
        result = await runtime.run(["rm", "-rf", f".vf-pool/{trace.id}"], {})
        if result.exit_code != 0:
            raise RuntimeError(
                f"failed to clean up Pool state: {result.stderr.strip()[-500:]}"
            )
