import verifiers.v1 as vf

from .command import CommandHarness, CommandHarnessConfig, shell_command

TERMINUS_2_DEFAULT_WORKDIR = "/app"
TERMINUS_2_DEFAULT_LOG_PATH = "/logs/agent/terminus_2.log"
TERMINUS_2_DEFAULT_VERSION = "harbor==0.6.6"
TERMINUS_2_DEFAULT_MODEL_NAME = "openai/gpt-4.1-mini"
TERMINUS_2_DEFAULT_API_BASE_URL = "https://api.pinference.ai/api/v1"


class Terminus2Config(CommandHarnessConfig):
    version: str = TERMINUS_2_DEFAULT_VERSION
    cwd: str | None = TERMINUS_2_DEFAULT_WORKDIR
    log_path: str = TERMINUS_2_DEFAULT_LOG_PATH
    model_name: str = TERMINUS_2_DEFAULT_MODEL_NAME
    api_base_url: str = TERMINUS_2_DEFAULT_API_BASE_URL
    max_turns: int = 4


class Terminus2(CommandHarness[Terminus2Config]):
    config: Terminus2Config

    def command(self, task: vf.Task, state: vf.State) -> list[str]:
        _ = state
        instruction = vf.task_text(task, state, keys=("instruction", "question"))
        system_prompt = vf.messages_text(vf.get_messages(self.system_prompt))
        if system_prompt:
            instruction = f"{system_prompt}\n\n{instruction}"
        script = f"""
set -eo pipefail
mkdir -p "$(dirname {self.config.log_path!r})"
python -m harbor.agents.terminus_2 \
  --model {self.config.model_name!r} \
  --api-base "${{OPENAI_BASE_URL:-{self.config.api_base_url}}}" \
  --instruction {instruction!r} 2>&1 | tee -a {self.config.log_path!r}
"""
        return shell_command(script)


def load_harness(config: Terminus2Config) -> Terminus2:
    return Terminus2(config=config)
