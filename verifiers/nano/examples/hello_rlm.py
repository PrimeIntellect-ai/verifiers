"""hello-rlm: an agentic environment driven by the `rlm` CLI binary.

The model never answers directly — the `RLMAgent` installs and runs the `rlm` CLI,
which makes its own LLM calls (intercepted into the transcript), and the reward
checks the requested answer appears in what the agent *produced* (its assistant
turns and tool results — not the instruction).

Defaults to docker (rlm installs itself into the container). Tune rlm via the
agent config, e.g. `--env.agent.max-depth 2`; run on the host instead with
`--env.runtime.kind subprocess --env.agent.path /path/to/rlm`.
"""

import verifiers.nano as vf

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
    """The string the agent's output must contain."""


class HelloRLMTaskset(vf.Taskset[HelloRLMTask, vf.TasksetConfig]):
    def load_tasks(self) -> list[HelloRLMTask]:
        return [HelloRLMTask(id=tid, instruction=q, answer=a) for tid, q, a in TASKS]

    @vf.reward(weight=1.0)
    async def answer_in_output(
        self, task: HelloRLMTask, transcript: vf.Transcript
    ) -> float:
        # What the agent produced: its assistant turns + tool results (e.g. the
        # ipython stdout), across the whole trajectory — but not the instruction.
        produced = [
            m.content or ""
            for turn in transcript.trajectory
            for m in (*turn.prompt, turn.response.message)
            if m.role in ("assistant", "tool")
        ]
        return float(task.answer.lower() in " ".join(produced).lower())


class EnvConfig(vf.EnvConfig):
    agent: vf.AgentConfig = vf.RLMAgentConfig()
    # rlm installs itself into a container, so default to docker (self-contained).
    runtime: vf.RuntimeConfig = vf.DockerConfig(image="python:3.11-slim")


def load_taskset(config: vf.TasksetConfig | None = None) -> HelloRLMTaskset:
    return HelloRLMTaskset(config or vf.TasksetConfig())


def load_agent(config: vf.AgentConfig | None = None) -> vf.Agent:
    return vf.make_agent(config or vf.RLMAgentConfig())


def load_environment(config: EnvConfig | None = None) -> vf.Environment:
    config = config or EnvConfig()
    return vf.Environment(
        taskset=load_taskset(config.taskset),
        agent=load_agent(config.agent),
        runtime=config.runtime,
    )
