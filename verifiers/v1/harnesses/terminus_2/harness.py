import json
import logging
from pathlib import Path

from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig, PinnedVersion
from verifiers.v1.harness import Harness
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

PROGRAM_SOURCE = (Path(__file__).resolve().parent / "program.py").read_text()
logger = logging.getLogger(__name__)


class Terminus2HarnessConfig(HarnessConfig):
    version: PinnedVersion = "0.23.0"
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
        # The endpoint, its key and the prompts go in a file the program reads and
        # deletes, never argv: the box's `ps` would show the key to the agent, and an
        # agent command such as `pkill -f <word from the task>` would match, and kill,
        # its own harness (as with `launch_chat_program`).
        prompt_file = f".vf-terminus-2-prompt-{trace.id}.json"
        args = [f"--model={ctx.model}", f"--prompt-file={prompt_file}"]
        try:
            await runtime.write(
                prompt_file,
                json.dumps(
                    {
                        "base_url": endpoint,
                        "api_key": secret,
                        "system_prompt": system_prompt,
                        "task": prompt,
                    }
                ).encode(),
            )
            await runtime.run(["chmod", "600", prompt_file], {})
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
