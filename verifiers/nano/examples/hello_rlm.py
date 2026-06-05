"""hello-rlm: an agentic environment driven by the `rlm` CLI binary.

The model never answers directly — the `rlm` agent runs as a subprocess (via the
`ProgramHarness`), makes its own LLM calls (intercepted into the transcript), and
the reward checks the requested answer appears in the agent's output.

Point the harness at your rlm binary with
`--env.harness.command '["/path/to/rlm"]'`, or have `rlm` on PATH.
"""

import verifiers.nano as vf

# (id, instruction, expected answer)
TASKS = [
    ("0", "Reply with exactly: hello rlm", "hello rlm"),
    ("1", "Reply with exactly: taskset harness", "taskset harness"),
    ("2", "Reply with exactly: runtime boundary", "runtime boundary"),
]


class HelloRLMTask(vf.Task):
    answer: str
    """The string the agent's output must contain."""


class HelloRLMTaskset(vf.Taskset[HelloRLMTask, vf.TasksetConfig]):
    def load_tasks(self) -> list[HelloRLMTask]:
        return [HelloRLMTask(id=tid, instruction=q, answer=a) for tid, q, a in TASKS]

    @vf.reward(weight=1.0)
    async def answer_in_output(
        self, task: HelloRLMTask, transcript: vf.Transcript
    ) -> float:
        output = " ".join(
            m.content or "" for m in transcript.messages if m.role == "assistant"
        )
        return float(task.answer.lower() in output.lower())


class EnvConfig(vf.EnvConfig):
    harness: vf.ProgramConfig = vf.ProgramConfig(command=["rlm"])


def load_taskset(config: vf.TasksetConfig | None = None) -> HelloRLMTaskset:
    return HelloRLMTaskset(config or vf.TasksetConfig())


def load_harness(config: vf.ProgramConfig | None = None) -> vf.ProgramHarness:
    return vf.ProgramHarness(config or vf.ProgramConfig(command=["rlm"]))


def load_environment(config: EnvConfig | None = None) -> vf.Environment:
    config = config or EnvConfig()
    return vf.Environment(
        taskset=load_taskset(config.taskset),
        harness=load_harness(config.harness),
    )
