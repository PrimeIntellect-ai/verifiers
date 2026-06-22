"""The Terminus 2 harness: runs Harbor's tmux agent through LiteLLM."""

import logging
from pathlib import Path

from verifiers.v1.clients import RolloutContext
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace

PROGRAM_SOURCE = (Path(__file__).resolve().parent / "program.py").read_text()
logger = logging.getLogger(__name__)


class Terminus2HarnessConfig(HarnessConfig):
    """The Harbor Terminus 2 harness."""

    version: str = "0.14.0"
    """Harbor release to install, pinned for reproducibility."""


class Terminus2Harness(Harness[Terminus2HarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = False

    async def setup(self, runtime: Runtime) -> None:
        # TODO: Terminus drives tmux; on the host (subprocess) runtime its tmux server — and the
        # `tmux kill-server` cleanup in `launch` — share the host's tmux, so a host run can kill
        # the user's own tmux session. Until tmux is isolated (a dedicated `tmux -L` socket + a
        # created, private TMUX_TMPDIR), refuse the host runtime; run Terminus 2 in a container.
        if runtime.type == "subprocess":
            raise RuntimeError(
                "Terminus 2 drives tmux and is unsafe on the subprocess (host) runtime — its tmux "
                "cleanup can kill the host's tmux server. Run it in a container runtime "
                "(--harness.runtime.type docker|prime|modal)."
            )
        source = PROGRAM_SOURCE.replace("{version}", self.config.version)
        await runtime.prepare_uv_script(source, self.config.env)

    async def launch(
        self,
        ctx: RolloutContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> ProgramResult:
        if self.config.disabled_tools:
            raise ValueError("Terminus 2 does not support disabling tools")
        system_prompt, prompt = self.resolve_prompt(trace.task)
        if prompt is None:
            raise ValueError(
                "Terminus 2 requires a task prompt (it has no user simulator)"
            )
        tmux_dir = f"/tmp/vf-terminus-2-{trace.id}"
        env = {
            **self.config.env,
            "TMUX_TMPDIR": tmux_dir,
        }
        args = [
            f"--base-url={endpoint}",
            f"--api-key={secret}",
            f"--model={ctx.model}",
            f"--system-prompt={system_prompt or ''}",
            f"--task={prompt}",
        ]
        try:
            source = PROGRAM_SOURCE.replace("{version}", self.config.version)
            program = await runtime.prepare_uv_script(source, self.config.env)
            return await runtime.run_program([*program, *args], env)
        finally:
            # Harbor normally destroys its whole sandbox; this adapter borrows the
            # Verifiers runtime, so clean up Terminus's detached tmux server ourselves.
            try:
                await runtime.run(
                    [
                        "sh",
                        "-c",
                        'tmux kill-server >/dev/null 2>&1 || true; rm -rf "$TMUX_TMPDIR"',
                    ],
                    {"TMUX_TMPDIR": tmux_dir},
                )
            except Exception:
                # Runtime teardown is the final backstop; preserve the rollout's
                # result or original failure when this best-effort cleanup cannot run.
                logger.warning(
                    "failed to clean up Terminus 2 tmux server", exc_info=True
                )
