"""The codex harness: installs the Codex CLI into the runtime and runs `codex exec`.

Codex only speaks the streaming OpenAI Responses API, so it reaches the interception server as a
custom model provider (`wire_api = responses`) pointed at the rollout endpoint — served by the
Responses dialect + SSE relay (see `verifiers.v1.dialects.responses`). The binary is the static
musl release, so it drops into any linux container with no runtime deps; its bearer token (the
session secret) is read from an env var.
"""

import logging
import shlex

from verifiers.v1.clients import ModelContext
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

# The provider id we register on the fly via `-c` overrides — arbitrary, internal to codex.
PROVIDER = "intercept"
# The env var codex reads the provider api key (its bearer = the session secret) from.
KEY_VAR = "CODEX_INTERCEPT_KEY"

CODEX_DIR = "/tmp/vf-codex"
CODEX_BIN = f"{CODEX_DIR}/bin/codex"
INSTALL = r"""
set -e
mkdir -p {dir}/bin
command -v curl >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq curl >/dev/null; }
case "$(uname -m)" in aarch64|arm64) arch=aarch64 ;; *) arch=x86_64 ;; esac
triple="${arch}-unknown-linux-musl"
curl -fsSL "https://github.com/openai/codex/releases/download/rust-v{version}/codex-${triple}.tar.gz" | tar -xz -C {dir}/bin
mv "{dir}/bin/codex-${triple}" {bin}
chmod +x {bin}
"""


class CodexHarnessConfig(HarnessConfig):
    """The Codex CLI harness — which codex release to install in the runtime."""

    version: str = "0.137.0"
    """Codex release to install (the `rust-v<version>` GitHub release); pinned for reproducibility."""


class CodexHarness(Harness[CodexHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = False  # TODO
    SUPPORTS_MCP = False  # TODO

    async def setup(self, runtime: Runtime) -> None:
        logger.info("codex: ensuring codex %s is installed", self.config.version)
        script = (
            INSTALL.replace("{version}", self.config.version)
            .replace("{dir}", CODEX_DIR)
            .replace("{bin}", CODEX_BIN)
        )
        ensure = shlex.quote(f"[ -x {CODEX_BIN} ] || ({script})")
        # Shared local runtimes may provision concurrently; only the first downloads.
        guarded = (
            f"mkdir -p {CODEX_DIR} && flock {CODEX_DIR}/install.lock sh -c {ensure}"
        )
        install = await runtime.run(["sh", "-c", guarded], {})
        if install.exit_code != 0:
            raise RuntimeError(f"codex install failed: {install.stderr.strip()[-500:]}")

    async def launch(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> ProgramResult:
        _, prompt = self.resolve_prompt(trace.task)
        # codex authenticates to the interception server with the session secret (its provider
        # api key) and posts Responses calls to `{endpoint}/responses`.
        env = {**self.config.resolved_env, KEY_VAR: secret}
        # Values are Codex feature names such as `shell_tool`; Codex owns validation.
        # https://developers.openai.com/codex/config-reference#features
        tool_config = [
            arg
            for tool in self.config.disabled_tools or []
            for arg in ("--disable", tool)
        ]
        # `-c` values parse as TOML, falling back to a raw string (so the url / `responses`
        # come through literally); `requires_openai_auth=false` parses as a bool.
        argv = [
            CODEX_BIN,
            "exec",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "-m",
            ctx.model,
            "-c",
            f"model_provider={PROVIDER}",
            "-c",
            f"model_providers.{PROVIDER}.name={PROVIDER}",
            "-c",
            f"model_providers.{PROVIDER}.base_url={endpoint}",
            "-c",
            f"model_providers.{PROVIDER}.env_key={KEY_VAR}",
            "-c",
            f"model_providers.{PROVIDER}.wire_api=responses",
            "-c",
            f"model_providers.{PROVIDER}.requires_openai_auth=false",
            *tool_config,
            "--",
            prompt,
        ]
        return await runtime.run_program(argv, env)
