"""Prime Agent over its native ACP mode, installed via the published installer."""

import hashlib
import json
import logging
import shlex

from verifiers.v1.acp import ACP
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness, HarnessSession
from verifiers.v1.harnesses.node import NODE_BIN_DIR, ensure_node
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

INSTALL_URL = "https://pub-728493de92a943e2a9b2d17b4719f318.r2.dev/install.sh"
PRIME_AGENT_DIR = "/var/tmp/vf-prime-agent"
STATE_ROOT = "/tmp/vf-prime-agent-runs"
SKILLS_DIR = ".agents/skills"

PROVIDER = "intercept"
# models.json carries this variable NAME, never the secret: prime-agent resolves
# `process.env[apiKey] || apiKey`, so the token only rides the process env.
KEY_VAR = "PRIME_AGENT_INTERCEPT_KEY"
# Prime Agent reads its agent directory from its packaged config name.
ENV_AGENT_DIR = "PRIME_AGENT_CODING_AGENT_DIR"

# The published installer resolves, verifies, and `npm install -g`s the release
# named by PRIME_AGENT_VERSION; NPM_CONFIG_PREFIX keys the install per version so
# a changed pin never reuses another rollout's tree. Headless it proceeds without
# confirmation. The kernel runtime is prepared on first use instead of at install
# time: it lands under HOME, which is per-trace at run time.
INSTALL = r"""
set -e
export PATH="/var/tmp/vf-node/bin:$PATH"
prefix="$VF_PRIME_AGENT_DIR/$PRIME_AGENT_VERSION"
[ -x "$prefix/bin/prime-agent" ] && exit 0
export NPM_CONFIG_PREFIX="$prefix"
export PRIME_AGENT_BOOTSTRAP_KERNEL_ON_INSTALL=0
curl -fsSL "$VF_PRIME_AGENT_INSTALL_URL" | sh
"""

PRIME_AGENT_ACP = ACP()


class PrimeAgentHarnessConfig(HarnessConfig):
    version: str = "0.7.0"
    """Prime Agent release to install, pinned for reproducibility (the installer
    verifies the published artifact for whatever version is selected)."""


class PrimeAgentHarness(Harness[PrimeAgentHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    # Prime Agent's ACP mode ignores `session/new` mcpServers: its MCP
    # integrations are kernel-side skills, not client-injected tools. Claiming
    # support would run tool-bearing tasksets with their tools silently absent.
    SUPPORTS_MCP = False
    SUPPORTS_RESUME = True
    SUPPORTS_SKILLS = True

    def _bin(self) -> str:
        return f"{PRIME_AGENT_DIR}/{self.config.version}/bin/prime-agent"

    def _root(self, trace: Trace) -> str:
        # Hash the trace id: it is untrusted input for a filesystem path, and the
        # short digest keeps the daemon socket path under sun_path limits.
        return f"{STATE_ROOT}/{hashlib.sha256(trace.id.encode()).hexdigest()[:16]}"

    async def setup(self, runtime: Runtime) -> None:
        await self.install_skills(runtime, SKILLS_DIR)
        # ensure_node also bootstraps curl, which the installer pipe needs.
        await ensure_node(runtime)
        logger.info("prime-agent: ensuring %s is installed", self.config.version)
        lock = f"{PRIME_AGENT_DIR}/install.lock"
        guarded = (
            f"mkdir -p {PRIME_AGENT_DIR} && "
            f'"$(command -v flock || command -v lockf)" {lock} '
            f"sh -c {shlex.quote(INSTALL)}"
        )
        result = await runtime.run(
            ["sh", "-c", guarded],
            {
                **self.config.resolved_env,
                "VF_PRIME_AGENT_DIR": PRIME_AGENT_DIR,
                "VF_PRIME_AGENT_INSTALL_URL": INSTALL_URL,
                "PRIME_AGENT_VERSION": self.config.version,
            },
        )
        if result.exit_code != 0:
            raise RuntimeError(
                f"prime-agent install failed: {result.stderr.strip()[-500:]}"
            )
        await PRIME_AGENT_ACP.setup(self, runtime)

    async def session(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
    ) -> HarnessSession:
        # One live ACP process per trace keeps a single Prime Agent session — and
        # with it one IPython kernel — alive across turns: the agent refuses a
        # second `session/new`, so a relaunch per segment cannot preserve state.
        if not runtime.supports_live_processes:
            return await super().session(
                ctx, trace, runtime, endpoint, secret, mcp_urls, data
            )
        system_prompt, prompt = self.resolve_prompt(data)
        command = await self._prepare(ctx, trace, runtime, endpoint, system_prompt)
        return PRIME_AGENT_ACP.session(
            self,
            ctx,
            trace,
            runtime,
            endpoint,
            secret,
            mcp_urls,
            data,
            env=self._env(trace, secret),
            command=command,
            prompt=prompt,
        )

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
        """Run one standalone ACP segment through the default session adapter."""
        system_prompt, prompt = self.resolve_prompt(data)
        command = await self._prepare(ctx, trace, runtime, endpoint, system_prompt)
        return await PRIME_AGENT_ACP.run(
            runtime,
            self._env(trace, secret),
            command,
            prompt,
            allow_empty_tool_reply=True,
        )

    def _env(self, trace: Trace, secret: str) -> dict[str, str]:
        root = self._root(trace)
        # Prime Agent writes sessions and kernel state under HOME; pin it (and
        # TMPDIR) to the per-trace root so concurrent rollouts share nothing.
        return {
            **self.config.resolved_env,
            KEY_VAR: secret,
            ENV_AGENT_DIR: f"{root}/agent",
            "HOME": f"{root}/agent",
            "TMPDIR": f"{root}/tmp",
            "PRIME_AGENT_TELEMETRY": "0",
        }

    async def _prepare(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        system_prompt: str | None,
    ) -> list[str]:
        root = self._root(trace)
        agent_dir = f"{root}/agent"
        created = await runtime.run(
            ["mkdir", "-p", "-m", "700", root, agent_dir, f"{root}/tmp"], {}
        )
        if created.exit_code != 0:
            raise RuntimeError(
                f"prime-agent state directory failed: {created.stderr.strip()[-500:]}"
            )

        reasoning = ctx.sampling.reasoning_effort not in (
            None,
            "none",
        ) or ctx.model.rsplit("/", 1)[-1].startswith(("gpt-5", "o1", "o3", "o4"))
        models = {
            "providers": {
                PROVIDER: {
                    "baseUrl": endpoint,
                    "api": "openai-completions",
                    # The variable name, not the secret (see KEY_VAR above).
                    "apiKey": KEY_VAR,
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
        models_path = f"{agent_dir}/models.json"
        await runtime.write(models_path, json.dumps(models).encode())
        await runtime.run(["chmod", "600", models_path], {})

        args = [
            self._bin(),
            "--mode",
            "acp",
            "--provider",
            PROVIDER,
            "--model",
            ctx.model,
            "--daemon-socket",
            f"{root}/daemon.sock",
        ]
        for skill in self.config.skills:
            # Resolve like `install_skills` so the path matches what it wrote.
            args += ["--skill", f"{SKILLS_DIR}/{skill.resolve().name}"]
        if self.config.disabled_tools:
            raise ValueError(
                "prime-agent has no per-tool disable flag; its model-facing tool "
                "surface is ipython"
            )
        if system_prompt:
            # Passed once per launch as a flag: folding it into the transcript
            # would re-apply it on every resumed segment.
            args += ["--append-system-prompt", system_prompt]

        # The bundled Node lives beside the install, not on the image PATH.
        wrapper = f"{root}/prime-agent"
        await runtime.write(
            wrapper,
            (
                "#!/bin/sh\n"
                "set -eu\n"
                f'export PATH="{NODE_BIN_DIR}:$PATH"\n'
                f'exec {shlex.join(args)} "$@"\n'
            ).encode(),
        )
        await runtime.run(["chmod", "700", wrapper], {})
        return ["sh", "-c", f"exec {shlex.quote(wrapper)}"]

    async def cleanup(self, trace: Trace, runtime: Runtime) -> None:
        root = self._root(trace)
        # Stop this trace's daemon before deleting its state: a live worker would
        # keep writing into a removed directory.
        stopped = await runtime.run(
            [
                "sh",
                "-c",
                (
                    f"{shlex.quote(self._bin())} stop "
                    f"--daemon-socket {shlex.quote(root + '/daemon.sock')}"
                ),
            ],
            self._env(trace, ""),
        )
        if stopped.exit_code != 0:
            logger.warning(
                "prime-agent: stopping the trace daemon failed (exit %s): %s",
                stopped.exit_code,
                stopped.stderr.strip()[-300:],
            )
        await runtime.run(["rm", "-rf", root], {})
