"""Run NAC through a Verifiers-owned ACP adapter."""

import logging
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import tomli_w
from pydantic import Field

from verifiers.v1.acp import ACPConfig, ACPHarness
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

PROGRAM_SOURCE = (Path(__file__).resolve().parent / "program.py").read_text()
NAC_DIR = "/var/tmp/vf-nac-{version}"
NAC_BIN = f"{NAC_DIR}/nac-web"
SKILLS_DIR = ".agents/skills"
KEY_VAR = "NAC_INTERCEPT_KEY"
MCP_SYSTEM_PROMPT = (
    "Configured MCP tools are available to worker threads and use NAC names of the "
    "form `mcp__<server>__<tool>`. When a task requires an MCP tool, delegate it to "
    "a worker and copy the worker's tool result verbatim into the final response."
)

INSTALL_SOURCE = r"""# /// script
# requires-python = ">=3.11"
# ///
import os
import platform
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

version, destination = sys.argv[1:]
target = {
    ("Linux", "x86_64"): "x86_64-unknown-linux-musl",
    ("Linux", "amd64"): "x86_64-unknown-linux-musl",
    ("Darwin", "arm64"): "aarch64-apple-darwin",
    ("Darwin", "aarch64"): "aarch64-apple-darwin",
}.get((platform.system(), platform.machine()))
if target is None:
    raise SystemExit(
        f"NAC has no release for {platform.system()} {platform.machine()}"
    )

path = Path(destination)
if path.is_file() and os.access(path, os.X_OK):
    raise SystemExit()
path.parent.mkdir(parents=True, exist_ok=True)
url = f"https://github.com/arcee-ai/nac/releases/download/v{version}/nac-{target}.tar.gz"
tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
try:
    with tempfile.TemporaryDirectory() as directory:
        archive = Path(directory) / "nac.tar.gz"
        urllib.request.urlretrieve(url, archive)
        with tarfile.open(archive, "r:gz") as bundle:
            binary = bundle.extractfile("nac-web")
            if binary is None:
                raise RuntimeError("NAC release does not contain nac-web")
            with tmp.open("wb") as output:
                shutil.copyfileobj(binary, output)
    tmp.chmod(0o755)
    tmp.replace(path)
finally:
    tmp.unlink(missing_ok=True)
"""


class NacHarnessConfig(HarnessConfig):
    version: str = Field(default="0.1.1", pattern=r"^[A-Za-z0-9._+-]+$")
    """NAC release to install, pinned for reproducibility."""
    transport: Literal["responses", "anthropic_messages"] = "responses"
    """Model API transport."""


class NacHarness(ACPHarness[NacHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_SKILLS = True

    async def setup(self, runtime: Runtime) -> None:
        await self.install_skills(runtime, SKILLS_DIR)
        logger.info("nac: ensuring NAC %s is installed", self.config.version)
        binary = NAC_BIN.format(version=self.config.version)
        install = await runtime.run_uv_script(
            INSTALL_SOURCE,
            [self.config.version, binary],
            self.config.resolved_env,
        )
        if install.exit_code != 0:
            raise RuntimeError(f"NAC install failed: {install.stderr.strip()[-500:]}")
        await runtime.prepare_uv_script(PROGRAM_SOURCE, self.config.resolved_env)
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
            raise ValueError("NAC does not support disabling individual tools")

        home = self.trace_home(trace)
        created = await runtime.run(["mkdir", "-p", f"{home}/home"], {})
        if created.exit_code != 0:
            raise RuntimeError(
                f"failed to create NAC home: {created.stderr.strip()[-500:]}"
            )

        config: dict[str, object] = {
            "mcp_servers": {
                name: {"transport": "streamable_http", "url": url}
                for name, url in mcp_urls.items()
            }
        }
        if hostname := urlparse(endpoint).hostname:
            config["security"] = {"trusted_base_url_hosts": [hostname]}
        await runtime.write(f"{home}/config.toml", tomli_w.dumps(config).encode())

        backend = {
            "responses": "openai-responses",
            "anthropic_messages": "anthropic-messages",
        }[self.config.transport]
        base_url = (
            endpoint.removesuffix("/v1")
            if self.config.transport == "anthropic_messages"
            else endpoint
        )
        env = {
            **self.config.resolved_env,
            "BROWSER": "none",
            "HOME": f"{home}/home",
            "MODELS_DEV_URL": "http://127.0.0.1:1",
            "NAC_HOME": home,
            "NO_COLOR": "1",
            KEY_VAR: secret,
            "VF_NAC_API_KEY_ENV": KEY_VAR,
            "VF_NAC_BACKEND": backend,
            "VF_NAC_BASE_URL": base_url,
            "VF_NAC_BIN": NAC_BIN.format(version=self.config.version),
            "VF_NAC_MODEL": ctx.model,
            "VF_NAC_VERSION": self.config.version,
        }
        system_prompt, prompt = self.resolve_prompt(data)
        if mcp_urls:
            system_prompt = "\n\n".join(
                part for part in (system_prompt, MCP_SYSTEM_PROMPT) if part
            )
        return ACPConfig(
            env=env,
            command=await runtime.prepare_uv_script(PROGRAM_SOURCE, env),
            prompt=prompt,
            # NAC workers load the rollout-scoped MCP config above.
            mcp_urls={},
            system_prompt=system_prompt,
        )

    async def cleanup(self, trace: Trace, runtime: Runtime) -> None:
        result = await runtime.run(["rm", "-rf", self.trace_home(trace)], {})
        if result.exit_code != 0:
            raise RuntimeError(
                f"failed to clean up NAC home: {result.stderr.strip()[-500:]}"
            )

    @staticmethod
    def trace_home(trace: Trace) -> str:
        return f"/tmp/vf-nac-home-{trace.id}"
