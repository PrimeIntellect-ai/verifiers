import shlex

import verifiers.v1 as vf

from .command import CommandHarness, CommandHarnessConfig, shell_command

MINI_SWE_AGENT_DEFAULT_WORKDIR = "/app"
MINI_SWE_AGENT_DEFAULT_LOG_PATH = "/logs/agent/mini-swe-agent.log"
MINI_SWE_AGENT_DEFAULT_OUTPUT_PATH = "/logs/agent/mini-swe-agent.traj.json"
MINI_SWE_AGENT_DEFAULT_VERSION = "mini-swe-agent@2.2.8"
MINI_SWE_AGENT_DEFAULT_CONFIG_SPEC = "mini"
MINI_SWE_AGENT_DEFAULT_MODEL_CLASS = "litellm"
MINI_SWE_AGENT_DEFAULT_ENVIRONMENT_TIMEOUT = 120


class MiniSWEAgentConfig(CommandHarnessConfig):
    version: str = MINI_SWE_AGENT_DEFAULT_VERSION
    cwd: str | None = MINI_SWE_AGENT_DEFAULT_WORKDIR
    log_path: str = MINI_SWE_AGENT_DEFAULT_LOG_PATH
    output_path: str = MINI_SWE_AGENT_DEFAULT_OUTPUT_PATH
    config_spec: str = MINI_SWE_AGENT_DEFAULT_CONFIG_SPEC
    model_class: str = MINI_SWE_AGENT_DEFAULT_MODEL_CLASS
    environment_timeout: int = MINI_SWE_AGENT_DEFAULT_ENVIRONMENT_TIMEOUT
    parallel_tool_calls: bool = True
    extra_config_specs: list[str] = []
    max_turns: int = 4


class MiniSWEAgent(CommandHarness[MiniSWEAgentConfig]):
    config: MiniSWEAgentConfig

    def command(self, task: vf.Task, state: vf.State) -> list[str]:
        _ = state
        instruction = str(
            getattr(task, "instruction", None) or getattr(task, "question", None) or ""
        )
        if not instruction:
            instruction = "\n\n".join(
                str(getattr(message, "content", "") or "") for message in task.prompt
            )
        config_args = [
            "-c",
            self.config.config_spec,
            "-c",
            "agent.cost_limit=0",
            "-c",
            f"environment.timeout={self.config.environment_timeout}",
            "-c",
            f"model.model_class={self.config.model_class}",
            "-c",
            "model.cost_tracking=ignore_errors",
            "-c",
            "model.model_kwargs.custom_llm_provider=openai",
            "-c",
            f"model.model_kwargs.parallel_tool_calls={str(self.config.parallel_tool_calls).lower()}",
        ]
        for spec in self.config.extra_config_specs:
            config_args.extend(["-c", spec])
        args = " ".join(shlex.quote(arg) for arg in config_args)
        script = f"""
set -eo pipefail
mkdir -p "$(dirname {self.config.log_path!r})" "$(dirname {self.config.output_path!r})"
mini --model "$OPENAI_MODEL" --task {instruction!r} --output {self.config.output_path!r} \
  --exit-immediately --yolo {args} 2>&1 | tee -a {self.config.log_path!r}
"""
        return shell_command(script)


def load_harness(config: MiniSWEAgentConfig) -> MiniSWEAgent:
    return MiniSWEAgent(config=config)
