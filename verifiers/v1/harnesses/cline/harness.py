"""Run the public Cline CLI headlessly through interception."""

import json
import logging
import shlex
from typing import Literal

from pydantic import Field

from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness
from verifiers.v1.harnesses.node import NODE_BIN_DIR, ensure_node
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

CLINE_DIR = "/var/tmp/vf-cline-{version}"
PACKAGES_DIR = f"{CLINE_DIR}/packages"
# Cline's package publishes its own platform resolver. npm 11 does not create a
# `.bin/cline` shim for this package in the minimal prefix install, so address the
# declared binary directly.
CLINE_BIN = f"{PACKAGES_DIR}/node_modules/cline/bin/cline"
CLINE_DATA_ROOT = "/tmp/vf-cline"

INSTALL = r"""
set -e
export PATH="/var/tmp/vf-node/bin:$PATH"
rm -f {ready}
npm install --prefix {packages} --no-audit --no-fund --omit=dev \
    "cline@$VF_CLINE_VERSION" >/dev/null
touch {ready}
"""


class ClineHarnessConfig(HarnessConfig):
    version: str = Field(default="3.0.57", pattern=r"^[A-Za-z0-9._+-]+$")
    """Public Cline CLI release to install, pinned for reproducibility."""

    compaction: Literal["agentic", "basic", "off"] = "basic"
    """Cline's context-compaction mode."""

    max_retries: int = Field(default=6, ge=1)
    """Maximum consecutive Cline mistakes before the CLI exits."""


class ClineHarness(Harness[ClineHarnessConfig]):
    async def setup(self, runtime: Runtime) -> None:
        await ensure_node(runtime)
        directory = CLINE_DIR.format(version=self.config.version)
        packages = PACKAGES_DIR.format(version=self.config.version)
        cline_bin = CLINE_BIN.format(version=self.config.version)
        ready = f"{directory}/.ready"
        script = INSTALL.replace("{packages}", packages).replace("{ready}", ready)
        ensure = shlex.quote(f"[ -f {ready} ] && [ -x {cline_bin} ] || ({script})")
        guarded = (
            f"mkdir -p {directory} && "
            f'"$(command -v flock || command -v lockf)" {directory}/install.lock '
            f"sh -c {ensure}"
        )
        logger.info("cline: ensuring Cline CLI %s is installed", self.config.version)
        result = await runtime.run(
            ["sh", "-c", guarded],
            {
                **self.config.resolved_env,
                "VF_CLINE_VERSION": self.config.version,
            },
        )
        if result.exit_code != 0:
            detail = (result.stderr or result.stdout).strip()[-500:]
            raise RuntimeError(f"Cline CLI install failed: {detail}")

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
        if mcp_urls:
            raise ValueError("Cline harness v1 does not support MCP servers")
        _, prompt = self.resolve_text_prompt(data)
        if prompt is None:
            raise ValueError("Cline requires a task prompt")

        data_dir = self.data_dir(trace)
        settings_dir = f"{data_dir}/settings"
        mcp_settings = f"{settings_dir}/cline_mcp_settings.json"
        await runtime.write(
            f"{settings_dir}/global-settings.json",
            json.dumps({"disabledTools": self.config.disabled_tools or []}).encode(),
        )
        await runtime.write(mcp_settings, b'{"mcpServers":{}}')

        env = {
            **self.config.resolved_env,
            "PATH": (
                f"{NODE_BIN_DIR}:/usr/local/sbin:/usr/local/bin:"
                "/usr/sbin:/usr/bin:/sbin:/bin"
            ),
            "CLINE_DATA_DIR": data_dir,
            "CLINE_MCP_SETTINGS_PATH": mcp_settings,
            "CLINE_TELEMETRY_DISABLED": "1",
            "CLINE_NO_AUTO_UPDATE": "1",
            "NO_UPDATE_NOTIFIER": "1",
        }
        # Execute npm's launcher through its shebang. Passing the `.bin` symlink
        # to `node` directly bypasses the package launcher's intended resolution
        # path for the platform-specific compiled binary.
        cline = [CLINE_BIN.format(version=self.config.version)]
        auth = await runtime.run(
            [
                *cline,
                "auth",
                "--provider",
                "openai-compatible",
                "--apikey",
                secret,
                "--modelid",
                ctx.model,
                "--baseurl",
                endpoint,
                "--data-dir",
                data_dir,
            ],
            env,
        )
        if auth.exit_code != 0:
            detail = (auth.stderr or auth.stdout).strip()[-500:]
            raise RuntimeError(f"Cline provider configuration failed: {detail}")

        args = [
            *cline,
            "--json",
            "--auto-approve",
            "true",
            "--cwd",
            ".",
            "--provider",
            "openai-compatible",
            "--key",
            secret,
            "--model",
            ctx.model,
            "--compaction",
            self.config.compaction,
            "--retries",
            str(self.config.max_retries),
            "--data-dir",
            data_dir,
        ]
        if effort := ctx.sampling.reasoning_effort:
            args += ["--thinking", effort]
        return await runtime.run_program([*args, prompt], env)

    async def cleanup(self, trace: Trace, runtime: Runtime) -> None:
        result = await runtime.run(["rm", "-rf", self.data_dir(trace)], {})
        if result.exit_code != 0:
            detail = (result.stderr or result.stdout).strip()[-500:]
            raise RuntimeError(f"failed to clean up Cline data: {detail}")

    @staticmethod
    def data_dir(trace: Trace) -> str:
        return f"{CLINE_DATA_ROOT}/{trace.id}"
