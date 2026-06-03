import verifiers as vf
from harnesses import RLM, RLMConfig


@vf.reward(weight=1.0)
async def exact_answer(task, state) -> float:
    stdout = str(state.get("command", {}).get("stdout") or "")
    return float(str(task["answer"]).lower() in stdout.lower())


# Each task exercises a distinct RLM capability end-to-end. Indices are stable
# so callers can pin a subset via `task_idx` on the taskset config.
TASKS: list[dict[str, str]] = [
    # 0: plain text echo — no tools needed.
    {
        "question": "Reply with exactly 'hello rlm'.",
        "answer": "hello rlm",
    },
    # 1: ipython tool — model must run Python in the persistent REPL to print
    # the answer rather than answering inline.
    {
        "question": (
            "Use your ipython tool to print exactly the string 'hello rlm' "
            "(no quotes, no extra text). Do not answer in plain text."
        ),
        "answer": "hello rlm",
    },
    # 2: ipython shell escape — exercise the `!cmd` shell passthrough.
    {
        "question": (
            "Inside an ipython cell, use the `!` shell escape to run "
            "`echo hello rlm` so the shell prints the literal string."
        ),
        "answer": "hello rlm",
    },
    # 3: ipython arithmetic — answer requires actual computation, not echo.
    {
        "question": (
            "Use the ipython tool to compute 7! (the factorial of 7) and "
            "report the result as a single integer."
        ),
        "answer": "5040",
    },
]


def load_tasks(split: vf.TaskSplit = "train"):
    _ = split
    return list(TASKS)


class HelloRLMTasksetConfig(vf.TasksetConfig):
    rewards: list[str] = ["exact_answer"]
    task_idx: int | list[int] | None = None
    """Restrict the taskset to a subset of `TASKS` by index. Useful for dev
    runs that only want to exercise one capability (e.g. `task_idx=1` for the
    ipython-tool test). `None` means run all tasks."""


class HelloRLMTaskset(vf.Taskset[HelloRLMTasksetConfig]):
    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        tasks = load_tasks(split)
        idx = self.config.task_idx
        if idx is None:
            return tasks
        wanted = [idx] if isinstance(idx, int) else list(idx)
        return [tasks[i] for i in wanted]


def load_taskset(config: HelloRLMTasksetConfig) -> HelloRLMTaskset:
    return HelloRLMTaskset(config=config)


def load_harness(config: RLMConfig) -> RLM:
    return RLM(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=vf.load_taskset(config=config.taskset),
        harness=vf.load_harness(config=config.harness),
    )
