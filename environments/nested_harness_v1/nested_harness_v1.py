from __future__ import annotations

import verifiers.v1 as vf


async def child_program(task, state):
    state["answer"] = str(task["prompt"]).upper()
    state["completion"] = [{"role": "assistant", "content": state["answer"]}]
    return state


async def call_harness(prompt, harness, state):
    _ = state
    task = vf.Task({"prompt": prompt}).freeze()
    child_state = await harness.run(task)
    return child_state["answer"]


@vf.metric
async def child_calls(task, state) -> float:
    return float(len(state["child_answers"]))


@vf.reward(weight=1.0)
async def exact_answer(task, state) -> float:
    return float(state["answer"] == task["answer"])


def source():
    return [
        {
            "prompt": "Ask a child harness to uppercase hello.",
            "child_prompts": ["hello"],
            "answer": "HELLO",
        },
        {
            "prompt": "Ask two child harnesses to uppercase short words.",
            "child_prompts": ["open", "source"],
            "answer": "OPEN SOURCE",
        },
    ]


def load_child_harness():
    return vf.Harness(program=child_program)


def load_toolset(config=None):
    return vf.Toolset(
        tools=[call_harness],
        objects={"child_harness": load_child_harness},
        bindings={
            "call_harness.harness": "objects.child_harness",
        },
        config=config,
    )


async def parent_program(task, state):
    tools = state.get_tools()
    answers = []
    for prompt in task["child_prompts"]:
        answer = await tools["call_harness"](prompt=prompt)
        answers.append(answer)
    state["child_answers"] = answers
    state["answer"] = " ".join(answers)
    state["completion"] = [{"role": "assistant", "content": state["answer"]}]
    return state


def load_taskset(config=None):
    return vf.Taskset(
        source=source,
        rewards=[exact_answer],
        config=config,
    )


def load_harness(config=None):
    return vf.Harness(
        program=parent_program,
        toolsets=[load_toolset(getattr(config, "toolset", None))],
        metrics=[child_calls],
        config=config,
    )


def load_environment(config=None):
    return vf.Env(
        taskset=load_taskset(getattr(config, "taskset", None)),
        harness=load_harness(getattr(config, "harness", None)),
    )
