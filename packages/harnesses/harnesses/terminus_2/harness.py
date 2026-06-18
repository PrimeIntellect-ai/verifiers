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

    id: str = "terminus-2"
    version: str = "0.14.0"
    """Harbor release to install, pinned for reproducibility."""


class Terminus2Harness(Harness[Terminus2HarnessConfig]):
    APPENDS_SYSTEM_PROMPT = False
    SUPPORTS_TASK_TOOLS = False

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
        _, prompt = self.resolve_prompt(trace.task)
        tmux_dir = f"/tmp/vf-terminus-2-{trace.id}"
        env = {
            **self.config.env,
            "OPENAI_BASE_URL": endpoint,
            "OPENAI_API_KEY": secret,
            "TMUX_TMPDIR": tmux_dir,
        }
        try:
            return await runtime.run_uv_script(
                PROGRAM_SOURCE.replace("{version}", self.config.version),
                args=[ctx.model, prompt],
                env=env,
            )
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
