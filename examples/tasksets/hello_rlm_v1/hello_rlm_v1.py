"""hello-rlm: an agentic environment driven by the `rlm` CLI binary.

The model never answers directly — the `RLMHarness` installs and runs the `rlm` CLI,
which makes its own LLM calls (intercepted into the trace), and the reward
checks the requested answer appears in what the harness *produced* (its assistant
turns and tool results — not the instruction).

Select the rlm harness with `--harness.id rlm` (it installs itself into the default docker
runtime). Tune rlm via the harness config, e.g. `--harness.max-depth 2`; run on the host
with `--harness.runtime.type subprocess`.
"""

import verifiers.v1 as vf

# (id, instruction, expected answer)
TASKS = [
    ("echo-reply", "Reply with exactly: hello rlm", "hello rlm"),
    (
        "echo-python-print",
        "Use your ipython tool to print exactly: hello world",
        "hello world",
    ),
    (
        "echo-bash-print",
        "Use your ipython tool to print via bash exactly: hello world",
        "hello world",
    ),
]


class HelloRLMTask(vf.Task):
    answer: str
    """The string the harness's output must contain."""


class HelloRLMTaskset(vf.Taskset[HelloRLMTask, vf.TasksetConfig]):
    def load_tasks(self) -> list[HelloRLMTask]:
        return [
            HelloRLMTask(idx=i, name=tid, instruction=q, answer=a)
            for i, (tid, q, a) in enumerate(TASKS)
        ]

    @vf.reward(weight=1.0)
    async def answer_in_output(
        self, task: HelloRLMTask, trace: vf.Trace, runtime: vf.Runtime
    ) -> float:
        # What the harness produced: its assistant turns + tool results (e.g. the
        # ipython stdout), across the whole trajectory — but not the instruction.
        produced = [
            m.content or ""
            for turn in trace.trajectory
            for m in (*turn.prompt, turn.response.message)
            if m.role in ("assistant", "tool")
        ]
        return float(task.answer.lower() in " ".join(produced).lower())


def load_taskset(config: vf.TasksetConfig) -> HelloRLMTaskset:
    return HelloRLMTaskset(config)
