import verifiers.v1 as vf

from .command import CommandHarness, CommandHarnessConfig, shell_command

RLM_DEFAULT_WORKDIR = "/workspace"
RLM_DEFAULT_TOOLS = ["ipython"]


class RLMConfig(CommandHarnessConfig):
    command: list[str] = []
    cwd: str | None = RLM_DEFAULT_WORKDIR
    tools: list[str] = RLM_DEFAULT_TOOLS
    exec_timeout: int = 300
    max_depth: int = 0
    summarize_at_tokens: int | None = None
    append_to_system_prompt: str = ""


class RLM(CommandHarness[RLMConfig]):
    config: RLMConfig

    def command(self, task: vf.Task, state: vf.State) -> list[str]:
        if self.config.command:
            return list(self.config.command)
        _ = state
        instruction = str(
            getattr(task, "instruction", None) or getattr(task, "question", None) or ""
        )
        if not instruction:
            instruction = "\n\n".join(
                str(getattr(message, "content", "") or "") for message in task.prompt
            )
        return shell_command(f"rlm {instruction!r}")

    def command_env(self, task: vf.Task, state: vf.State) -> dict[str, str]:
        env = super().command_env(task, state)
        env.update(
            {
                "RLM_MODEL": state.model.model if state.model is not None else "",
                "RLM_TOOLS": ",".join(self.config.tools),
                "RLM_EXEC_TIMEOUT": str(self.config.exec_timeout),
                "RLM_MAX_DEPTH": str(self.config.max_depth),
                "RLM_APPEND_TO_SYSTEM_PROMPT": self.config.append_to_system_prompt,
            }
        )
        if self.config.summarize_at_tokens is not None:
            env["RLM_SUMMARIZE_AT_TOKENS"] = str(self.config.summarize_at_tokens)
        return env

    @vf.metric
    async def rlm_sub_llm_call_count(self, state: vf.State) -> float:
        return rlm_metric(state, "sub_llm_call_count")

    @vf.metric
    async def rlm_sub_llm_total_turns(self, state: vf.State) -> float:
        return rlm_metric(state, "sub_llm_total_turns")

    @vf.metric
    async def rlm_sub_llm_total_tool_calls(self, state: vf.State) -> float:
        return rlm_metric(state, "sub_llm_total_tool_calls")


def rlm_metric(state: vf.State, name: str) -> float:
    metrics = state.artifacts.get("rlm_metrics")
    if not isinstance(metrics, dict):
        return 0.0
    value = metrics.get(name)
    return float(value) if isinstance(value, int | float) else 0.0


def load_harness(config: RLMConfig) -> RLM:
    return RLM(config=config)
