import logging
from pathlib import Path

from verifiers.v1.clients import ModelContext
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace
from verifiers.v1.task import TaskData

PROGRAM_SOURCE = (Path(__file__).resolve().parent / "program.py").read_text()
logger = logging.getLogger(__name__)


class Terminus2HarnessConfig(HarnessConfig):
    version: str = "0.14.0"
    """Harbor release to install, pinned for reproducibility."""


class Terminus2Harness(Harness[Terminus2HarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = False
    # Beyond the usual host leaks, Terminus drives tmux: on the host its tmux server —
    # and the `tmux kill-server` cleanup in `launch` — would share the user's own.

    async def setup(self, runtime: Runtime) -> None:
        source = PROGRAM_SOURCE.replace("{version}", self.config.version)
        await runtime.prepare_uv_script(source, self.config.resolved_env)

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
        if self.config.disabled_tools:
            raise ValueError("Terminus 2 does not support disabling tools")
        system_prompt, prompt = self.resolve_text_prompt(data)
        if prompt is None:
            raise ValueError("Terminus 2 requires a task prompt")
        tmux_dir = f"/tmp/vf-terminus-2-{trace.id}"
        env = {
            **self.config.resolved_env,
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
            program = await runtime.prepare_uv_script(source, self.config.resolved_env)
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
