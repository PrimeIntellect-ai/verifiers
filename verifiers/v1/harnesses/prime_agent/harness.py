"""Prime Agent harness driving prime-agent's native ACP mode."""

import json
import logging
import shlex

from pydantic import Field

from verifiers.v1.acp import ACP
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

# Model traffic is routed through the interception endpoint, which prime-agent
# sees as an ordinary OpenAI-compatible provider.
PROVIDER = "intercept"
KEY_VAR = "PRIME_AGENT_INTERCEPT_KEY"

PRIME_AGENT_DIR = "/tmp/vf-prime-agent"
PRIME_AGENT_BIN = f"{PRIME_AGENT_DIR}/node_modules/.bin/prime-agent"
SKILLS_DIR = ".agents/skills"
INSTALLER_URL = "https://pub-728493de92a943e2a9b2d17b4719f318.r2.dev/install.sh"

INSTALL = r"""
set -e
if [ -x "$VF_PRIME_AGENT_BIN" ]; then
    exit 0
fi
mkdir -p "$VF_PRIME_AGENT_DIR"
cd "$VF_PRIME_AGENT_DIR"
# The published release tarball is a plain npm package, so install it directly
# instead of running the interactive installer script.
npm install --no-audit --no-fund --prefix "$VF_PRIME_AGENT_DIR" \
    "$VF_PRIME_AGENT_TARBALL"
"""

PRIME_AGENT_ACP = ACP()


class PrimeAgentHarnessConfig(HarnessConfig):
    version: str = "0.5.1"
    """Prime Agent release to install, pinned for reproducibility."""

    tarball_url: str | None = None
    """Override the release tarball URL (defaults to the pinned public release)."""

    autonomous: bool = False
    """Run with `--autonomous`, so the agent continues until its limits or gates stop it."""

    gates: list[str] = Field(default_factory=list)
    """Autonomous quality-gate commands. Requires `autonomous`."""


class PrimeAgentHarness(Harness[PrimeAgentHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_RESUME = True
    SUPPORTS_SKILLS = True

    def _tarball(self) -> str:
        if self.config.tarball_url:
            return self.config.tarball_url
        version = self.config.version
        base = INSTALLER_URL.rsplit("/", 1)[0]
        return f"{base}/releases/v{version}/prime-agent-{version}.tgz"

    async def setup(self, runtime: Runtime) -> None:
        await self.install_skills(runtime, SKILLS_DIR)
        logger.info("prime-agent: ensuring %s is installed", self.config.version)
        lock = f"{PRIME_AGENT_DIR}/install.lock"
        # Concurrent rollouts share one runtime; serialize the install like the
        # other node-based harnesses do.
        guarded = (
            f"mkdir -p {PRIME_AGENT_DIR} && "
            f'"$(command -v flock || command -v lockf)" {lock} '
            f"sh -c {shlex.quote(INSTALL)}"
        )
        install = await runtime.run(
            ["sh", "-c", guarded],
            {
                "VF_PRIME_AGENT_DIR": PRIME_AGENT_DIR,
                "VF_PRIME_AGENT_BIN": PRIME_AGENT_BIN,
                "VF_PRIME_AGENT_TARBALL": self._tarball(),
            },
        )
        if install.exit_code != 0:
            raise RuntimeError(
                f"prime-agent install failed: {install.stderr.strip()[-500:]}"
            )
        await PRIME_AGENT_ACP.setup(self, runtime)

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
        agent_dir = f".vf-prime-agent-{trace.id}"
        reasoning = ctx.sampling.reasoning_effort not in (
            None,
            "none",
        ) or ctx.model.rsplit("/", 1)[-1].startswith(("gpt-5", "o1", "o3", "o4"))
        models = {
            "providers": {
                PROVIDER: {
                    "baseUrl": endpoint,
                    "api": "openai-completions",
                    "apiKey": f"${KEY_VAR}",
                    "models": [
                        {
                            "id": ctx.model,
                            "reasoning": reasoning,
                            "input": ["text", "image"],
                        }
                    ],
                }
            }
        }
        await runtime.write(f"{agent_dir}/models.json", json.dumps(models).encode())

        env = {
            **self.config.resolved_env,
            KEY_VAR: secret,
            "PI_CODING_AGENT_DIR": agent_dir,
            "PI_OFFLINE": "1",
            "PI_TELEMETRY": "0",
        }

        args = [
            PRIME_AGENT_BIN,
            "--mode",
            "acp",
            "--provider",
            PROVIDER,
            "--model",
            ctx.model,
        ]
        for skill in self.config.skills:
            # Resolve like `install_skills` so the path matches what it wrote.
            args += ["--skill", f"{SKILLS_DIR}/{skill.resolve().name}"]
        if self.config.disabled_tools:
            raise ValueError(
                "prime-agent has no per-tool disable flag; its model-facing tool is "
                "ipython. Use --no-builtin-tools upstream if that is ever needed."
            )
        if self.config.autonomous:
            args.append("--autonomous")
            for gate in self.config.gates:
                args += ["--autonomous-gate", gate]
        elif self.config.gates:
            raise ValueError("prime-agent gates require autonomous=true")
        if system_prompt:
            args += ["--append-system-prompt", system_prompt]

        # ACP owns stdout, so the agent must be exec'd directly with HOME pinned
        # to the per-trace agent dir: prime-agent writes sessions and kernel state
        # under it, and rollouts must not share either.
        wrapper = f"{agent_dir}/prime-agent"
        await runtime.write(
            wrapper,
            (
                "#!/bin/sh\n"
                'PI_CODING_AGENT_DIR="$PWD/$PI_CODING_AGENT_DIR"\n'
                'export PI_CODING_AGENT_DIR HOME="$PI_CODING_AGENT_DIR"\n'
                f'exec {shlex.join(args)} "$@"\n'
            ).encode(),
        )
        await runtime.run(["chmod", "+x", wrapper], {})

        return await PRIME_AGENT_ACP.run(
            runtime,
            env,
            ["sh", "-c", f'exec "$PWD/{wrapper}"'],
            prompt,
            mcp_urls=mcp_urls,
            system_prompt=system_prompt,
        )
