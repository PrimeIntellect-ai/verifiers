"""The codex harness: installs the Codex CLI into the runtime and runs `codex exec`.

Codex only speaks the streaming OpenAI Responses API, so it reaches the interception server as a
custom model provider (`wire_api = responses`) pointed at the rollout endpoint — served by the
Responses dialect + SSE relay (see `verifiers.v1.dialects.responses`). The binary is the static
musl release, so it drops into any linux container with no runtime deps; its bearer token (the
session secret) is read from an env var.
"""

import logging

from verifiers.v1.clients import RolloutContext
from verifiers.v1.errors import ProgramError
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

# The provider id we register on the fly via `-c` overrides — arbitrary, internal to codex.
PROVIDER = "intercept"
# The env var codex reads the provider api key (its bearer = the session secret) from.
KEY_VAR = "CODEX_INTERCEPT_KEY"

# Install the static-musl codex release (`rust-v<version>`) onto PATH, idempotently — fetching
# curl first if the image lacks it. musl => no libc dep, so it runs in any linux container.
INSTALL = r"""
command -v codex >/dev/null 2>&1 && exit 0
set -e
command -v curl >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq curl >/dev/null; }
case "$(uname -m)" in aarch64|arm64) arch=aarch64 ;; *) arch=x86_64 ;; esac
triple="${arch}-unknown-linux-musl"
curl -fsSL "https://github.com/openai/codex/releases/download/rust-v{version}/codex-${triple}.tar.gz" | tar -xz -C /usr/local/bin
mv "/usr/local/bin/codex-${triple}" /usr/local/bin/codex
chmod +x /usr/local/bin/codex
"""


class CodexHarnessConfig(HarnessConfig):
    """The Codex CLI harness — which codex release to install in the runtime."""

    id: str = "codex"
    version: str = "0.137.0"
    """Codex release to install (the `rust-v<version>` GitHub release); pinned for reproducibility."""


class CodexHarness(Harness[CodexHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = False  # TODO
    SUPPORTS_TASK_TOOLS = False  # TODO

    async def launch(
        self,
        ctx: RolloutContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> ProgramResult:
        _, instruction = self.resolve_prompt(trace.task)
        # codex authenticates to the interception server with the session secret (its provider
        # api key) and posts Responses calls to `{endpoint}/responses`.
        env = {**self.config.env, KEY_VAR: secret}
        logger.info("codex: ensuring codex %s is installed", self.config.version)
        install = await runtime.run(
            ["sh", "-c", INSTALL.replace("{version}", self.config.version)], {}
        )
        if install.exit_code != 0:
            raise ProgramError(f"codex install failed: {install.stderr.strip()[-500:]}")
        # `-c` values parse as TOML, falling back to a raw string (so the url / `responses`
        # come through literally); `requires_openai_auth=false` parses as a bool.
        argv = [
            "codex",
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
            instruction,
        ]
        return await runtime.run(argv, env)


def load_harness(config: CodexHarnessConfig) -> CodexHarness:
    return CodexHarness(config)


__all__ = ["CodexHarness", "CodexHarnessConfig", "load_harness"]
