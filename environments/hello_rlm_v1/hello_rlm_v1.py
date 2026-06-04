import verifiers as vf
from harnesses import RLM, RLMConfig

TASKS = [
    # 0: plain text
    {"question": "Say 'hello rlm'.", "answer": "hello rlm"},
    # 1: python
    {"question": "Use ipython to print 'hello rlm'.", "answer": "hello rlm"},
    # 2: bash
    {"question": "Use `!echo hello rlm` in ipython.", "answer": "hello rlm"},
]


class HelloRLMTasksetConfig(vf.TasksetConfig):
    task_idx: int | list[int] | None = None
    """Restrict the taskset to a subset of `TASKS` by index. Useful for dev
    runs that only want to exercise one capability (e.g. `task_idx=1` for the
    ipython-tool test). `None` means run all tasks."""


class HelloRLMTaskset(vf.Taskset[HelloRLMTasksetConfig]):
    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        tasks = list(TASKS)
        idx = self.config.task_idx
        if idx is None:
            return tasks
        wanted = [idx] if isinstance(idx, int) else list(idx)
        return [tasks[i] for i in wanted]

    @vf.reward(weight=1.0)
    async def exact_answer(self, task, state) -> float:
        stdout = str(state.get("command", {}).get("stdout") or "")
        return float(str(task["answer"]).lower() in stdout.lower())


def load_taskset(config: HelloRLMTasksetConfig) -> HelloRLMTaskset:
    return HelloRLMTaskset(config=config)


def load_harness(config: RLMConfig) -> RLM:
    return RLM(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=vf.load_taskset(config=config.taskset),
        harness=vf.load_harness(config=config.harness),
    )
