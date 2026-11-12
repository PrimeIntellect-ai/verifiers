from pathlib import Path
from typing import Literal

from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig, PinnedVersion
from verifiers.v1.harness import Harness
from verifiers.v1.harnesses.utils.install import ensure_installed
from verifiers.v1.harnesses.utils.remote_runtime import (
    REMOTE_TOOL_CLIENT_SOURCE,
    remote_shell_tools,
)
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace
from verifiers.v1.utils.logging import get_logger

PROGRAM_SOURCE = (Path(__file__).resolve().parent / "program.py").read_text()
logger = get_logger(__name__)


class Terminus2HarnessConfig(HarnessConfig):
    dialect: Literal["chat_completions"] = "chat_completions"
    version: PinnedVersion = "0.25.0"
    """Harbor release to install, pinned for reproducibility."""


class Terminus2Harness(Harness[Terminus2HarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = False
    SUPPORTS_SPLIT_EXECUTION = True
    SUPPORTS_SPLIT_TOOL_INTERCEPTION = True
    # Beyond the usual host leaks, Terminus drives tmux: on the host its tmux server —
    # and the `tmux kill-server` cleanup in `launch` — would share the user's own.

    def _source(self) -> str:
        return PROGRAM_SOURCE.replace("{version}", self.config.version).replace(
            "# REMOTE_TOOL_CLIENT", REMOTE_TOOL_CLIENT_SOURCE
        )

    async def remote_tools(self, runtime: Runtime, data: TaskData):
        return remote_shell_tools(runtime)

    async def setup(self, runtime: Runtime) -> None:
        source = self._source()
        await runtime.prepare_uv_script(source, self.config.resolved_env)
        if runtime._remote_tools_only:
            assert runtime._execution_bridge is not None
            # Harbor installs tmux during agent.setup(), after the execution
            # network is locked. Provision the system tool in trusted setup.
            await ensure_installed(
                runtime._execution_bridge.setup_runtime,
                directory="/var/tmp/agent-tools/terminus-2",
                ready="command -v tmux >/dev/null 2>&1",
                install=(
                    "if command -v apt-get >/dev/null; then "
                    "apt-get update -qq && apt-get install -y -qq tmux; "
                    "elif command -v dnf >/dev/null; then dnf install -y -q tmux; "
                    "elif command -v apk >/dev/null; then apk add --no-cache tmux; "
                    "else echo 'Terminus 2 requires tmux in the execution image' >&2; exit 1; fi"
                ),
                env={},
                label="Terminus 2 tmux",
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
        tool_interception_url: str | None = None,
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
        execution_runtime = runtime
        if runtime._remote_tools_only:
            if not tool_interception_url or runtime._execution_bridge is None:
                raise ValueError("split Terminus 2 requires a remote tool gate")
            args.append(f"--tool-interception-url={tool_interception_url}")
            execution_runtime = runtime._execution_bridge.runtime
            await execution_runtime.run(["mkdir", "-p", "-m", "700", tmux_dir], {})
        try:
            source = self._source()
            program = await runtime.prepare_uv_script(source, self.config.resolved_env)
            return await runtime.run([*program, *args], env)
        finally:
            # Harbor normally destroys its whole sandbox; this adapter borrows the
            # Verifiers runtime, so clean up Terminus's detached tmux server ourselves.
            try:
                await execution_runtime.run(
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
                    "Failed to clean up the Terminus 2 tmux server", exc_info=True
                )
