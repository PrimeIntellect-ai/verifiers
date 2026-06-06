import verifiers.v1 as vf


def load_tasks(split: vf.TaskSplit = "train"):
    _ = split
    return [
        {
            "prompt": "Reply with exactly hello rlm.",
            "answer": "hello rlm",
        },
        {
            "prompt": "Reply with exactly taskset harness.",
            "answer": "taskset harness",
        },
        {
            "prompt": "Reply with exactly runtime boundary.",
            "answer": "runtime boundary",
        },
    ]


class HelloRLMTask(vf.Task):
    answer: str


class HelloRLMTasksetConfig(vf.TasksetConfig):
    pass


class HelloRLMTaskset(vf.Taskset[HelloRLMTasksetConfig]):
    task_type = HelloRLMTask

    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        return [HelloRLMTask.model_validate(record) for record in load_tasks(split)]

    @vf.reward(weight=1.0)
    async def exact_answer(self, task: HelloRLMTask, state: vf.State) -> float:
        stdout = str(state.scratch.get("command", {}).get("stdout") or "")
        return float(task.answer.lower() in stdout.lower())


class HelloRLMHarnessConfig(vf.HarnessConfig):
    max_turns: int = 1


class HelloRLMHarness(vf.Harness[HelloRLMHarnessConfig]):
    async def _run(
        self,
        task: HelloRLMTask,
        state: vf.State,
        *,
        ctx: vf.RolloutContext,
        runtime: vf.RuntimeSession | None = None,
        tools: vf.MCPToolRegistry | None = None,
        user: vf.MCPToolRegistry | None = None,
    ) -> None:
        _ = ctx, runtime, tools, user
        answer = task.answer
        message = vf.AssistantMessage(content=answer)
        state.scratch["command"] = {"stdout": answer, "stderr": "", "returncode": 0}
        state.add_turn(
            vf.Turn(prompt=self.initial_messages(task), completion=[message])
        )
        state.stop("deterministic_harness")


def load_taskset(config: HelloRLMTasksetConfig) -> HelloRLMTaskset:
    return HelloRLMTaskset(config=config)


def load_harness(config: HelloRLMHarnessConfig) -> HelloRLMHarness:
    return HelloRLMHarness(config=config)
