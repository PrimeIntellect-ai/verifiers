import sys

import verifiers.v1 as vf

NAME_GROUPS = [
    ["world"],
    ["prime", "verifiers"],
    ["taskset", "harness", "runtime"],
    ["sandbox"],
    ["alpha", "beta"],
    ["delta", "epsilon", "zeta"],
    ["tools", "users"],
    ["group", "reward", "advantage"],
    ["mcp", "search"],
    ["open", "superintelligence", "stack"],
]


class SubagentTasksetConfig(vf.TasksetConfig):
    system_prompt: str = (
        "You are a parent coordinator. Call ask_subagent once for each requested "
        "name. After all tool results are available, join the child answers with "
        "', ' and output only that final joined text."
    )


class SubagentHarnessConfig(vf.HarnessConfig):
    max_turns: int = 8


class SubagentTask(vf.Task):
    names: list[str]
    answer: str


class SubagentTaskset(vf.Taskset[SubagentTasksetConfig]):
    task_type = SubagentTask

    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        if split == "eval":
            return []
        return [
            {
                "names": names,
                "prompt": [{"role": "user", "content": f"Names: {', '.join(names)}"}],
                "answer": ", ".join(f"hello {name}" for name in names),
            }
            for names in NAME_GROUPS
        ]

    def load_toolsets(self, config: SubagentTasksetConfig) -> list[vf.Toolset]:
        _ = config
        return [
            vf.Toolset(
                name="subagent",
                scope="rollout",
                server=vf.MCPServerSpec(
                    command=[sys.executable, "-m", "hello_subagent_v1.servers.tools"]
                ),
            )
        ]

    @vf.metric
    async def subagent_calls(self, state: vf.State) -> float:
        calls = state.scratch.get("subagent_calls")
        return float(len(calls) if isinstance(calls, list) else 0)

    @vf.reward(weight=1.0)
    async def exact_answer(self, task: SubagentTask, state: vf.State) -> float:
        messages = vf.get_messages(state.completion or [], role="assistant")
        answer = str(messages[-1].content or "").strip() if messages else ""
        return float(answer == task.answer)


def load_taskset(config: SubagentTasksetConfig) -> SubagentTaskset:
    return SubagentTaskset(config=config)


def load_harness(config: SubagentHarnessConfig) -> vf.Harness:
    return vf.Harness(config=config)
