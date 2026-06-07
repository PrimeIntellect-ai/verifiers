import verifiers.v1 as vf

from .command import CommandHarness, CommandHarnessConfig, shell_command

PI_DEFAULT_VERSION = "@earendil-works/pi-coding-agent@latest"
PI_DEFAULT_WORKDIR = "/app"
PI_DEFAULT_LOG_PATH = "/logs/agent/pi.txt"
PI_DEFAULT_SYSTEM_PROMPT = "Complete the user's task using the available tools."


class PiConfig(CommandHarnessConfig):
    system_prompt: vf.SystemPrompt | None = PI_DEFAULT_SYSTEM_PROMPT
    version: str = PI_DEFAULT_VERSION
    cwd: str | None = PI_DEFAULT_WORKDIR
    log_path: str = PI_DEFAULT_LOG_PATH
    max_turns: int = 4


class Pi(CommandHarness[PiConfig]):
    config: PiConfig

    def command(self, task: vf.Task, state: vf.State) -> list[str]:
        _ = state
        instruction = str(
            getattr(task, "instruction", None) or getattr(task, "question", None) or ""
        )
        if not instruction:
            instruction = "\n\n".join(
                str(getattr(message, "content", "") or "") for message in task.prompt
            )
        system_prompt = "\n\n".join(
            str(getattr(message, "content", "") or "")
            for message in vf.get_messages(self.system_prompt)
        )
        system_prompt_arg = (
            f"--system-prompt {system_prompt!r}" if system_prompt else ""
        )
        script = f"""
set -eo pipefail
mkdir -p "$(dirname {self.config.log_path!r})"
pi --no-session --no-context-files --provider openai --model "$OPENAI_MODEL" \
  {system_prompt_arg} -p {instruction!r} 2>&1 | tee {self.config.log_path!r}
"""
        return shell_command(script)


def load_harness(config: PiConfig) -> Pi:
    return Pi(config=config)
