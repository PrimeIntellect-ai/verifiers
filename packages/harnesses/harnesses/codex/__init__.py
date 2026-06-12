"""The codex harness: installs the Codex CLI into the runtime and runs `codex exec`.

Codex only speaks the streaming OpenAI Responses API, so it reaches the interception server as a
custom model provider (`wire_api = responses`) pointed at the rollout endpoint — the server
fetches the completion unary and fake-streams it back (see `verifiers.v1.dialects.responses`).
The binary is the static musl release, so it drops into any linux container with no runtime
deps; its bearer token (the session secret) is read from an env var.
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

# Install the static-musl codex release onto PATH, idempotently. musl => no libc dep, so it runs
# in any linux container; the arch is read from `uname -m`. curl + tar are installed if missing.
# `{version}` is filled from the harness config.
_INSTALL = r"""
command -v codex >/dev/null 2>&1 && exit 0
set -e
case "$(uname -m)" in
  x86_64|amd64) triple=x86_64-unknown-linux-musl ;;
  aarch64|arm64) triple=aarch64-unknown-linux-musl ;;
  *) echo "codex: unsupported arch $(uname -m)" >&2; exit 1 ;;
esac
need=""
for tool in curl tar; do command -v $tool >/dev/null 2>&1 || need="$need $tool"; done
if [ -n "$need" ]; then
  { apt-get update -qq && apt-get install -y -qq $need; } >/dev/null 2>&1 \
    || apk add --no-cache $need >/dev/null 2>&1 || true
fi
url="https://github.com/openai/codex/releases/download/rust-v{version}/codex-${triple}.tar.gz"
mkdir -p /tmp/codex-dl
curl -fsSL "$url" -o /tmp/codex-dl/codex.tgz
tar -xzf /tmp/codex-dl/codex.tgz -C /tmp/codex-dl
mv "$(find /tmp/codex-dl -type f -name 'codex*' ! -name '*.tgz' | head -1)" /usr/local/bin/codex
chmod +x /usr/local/bin/codex
"""


class CodexHarnessConfig(HarnessConfig):
    """The Codex CLI harness — which codex release to install in the runtime."""

    id: str = "codex"
    version: str = "0.137.0"
    """Codex release to install (the `rust-v<version>` GitHub release); pinned for reproducibility."""


class CodexHarness(Harness[CodexHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = False  # fold any system prompt into the instruction
    SUPPORTS_TASK_TOOLS = (
        False  # codex runs its own shell tools, not the taskset's MCP servers
    )

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
        env = {KEY_VAR: secret}
        logger.info("codex: ensuring codex %s is installed", self.config.version)
        install = await runtime.run(
            ["sh", "-c", _INSTALL.replace("{version}", self.config.version)], {}
        )
        if install.exit_code != 0:
            raise ProgramError(f"codex install failed: {install.stderr.strip()[-500:]}")
        # `-c` values parse as TOML, falling back to a raw string (so the url / `responses`
        # come through literally); `requires_openai_auth=false` parses as a bool.
        argv = [
            "codex",
            "exec",
            "--dangerously-bypass-approvals-and-sandbox",  # headless: no prompts, no sandbox
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
