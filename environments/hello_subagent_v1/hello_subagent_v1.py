from __future__ import annotations

import verifiers.v1 as vf
from verifiers.v1.utils.tool_utils import load_tools_from_state


async def child_program(task, state):
    state["answer"] = f"hello {task['name']}"
    state["completion"] = [{"role": "assistant", "content": state["answer"]}]
    return state


async def ask_subagent(name: str, harness, state):
    task = vf.Task({"name": name}).freeze()
    child_state = await vf.current_runtime().run_harness(
        harness,
        task,
        parent_state=state,
    )
    return {
        "answer": child_state["answer"],
        "trajectory_id": child_state["trajectory_id"],
        "metrics": child_state.get("metrics", {}),
    }


@vf.metric
async def subagent_calls(task, state) -> float:
    return float(len(state.get("subagent_results", [])))


@vf.reward(weight=1.0)
async def exact_answer(task, state) -> float:
    return float(state.get("answer") == task["answer"])


def source():
    return [
        {
            "prompt": (
                "Use the ask_subagent tool for each requested name, then join "
                "the child answers with ', '."
            ),
            "names": ["world"],
            "answer": "hello world",
        },
        {
            "prompt": (
                "Use the ask_subagent tool for each requested name, then join "
                "the child answers with ', '."
            ),
            "names": ["prime", "verifiers"],
            "answer": "hello prime, hello verifiers",
        },
    ]


def load_child_harness():
    return vf.Harness(program=child_program)


def load_toolset(config=None):
    return vf.Toolset(
        tools=[ask_subagent],
        objects={"child_harness": load_child_harness},
        bindings={"ask_subagent.harness": "objects.child_harness"},
        config=config,
    )


async def parent_program(task, state):
    tools = load_tools_from_state(state)
    results = []
    answers = []
    for name in task["names"]:
        result = await tools["ask_subagent"](name=name)
        results.append(result)
        answers.append(result["answer"])
    state["subagent_results"] = results
    state["answer"] = ", ".join(answers)
    state["completion"] = [{"role": "assistant", "content": state["answer"]}]
    return state


def load_taskset(config=None):
    return vf.Taskset(source=source, rewards=[exact_answer], config=config)


def load_harness(config=None):
    return vf.Harness(
        program=parent_program,
        toolsets=[load_toolset(getattr(config, "toolset", None))],
        metrics=[subagent_calls],
        config=config,
    )


def load_environment(config=None):
    return vf.Env(
        taskset=load_taskset(getattr(config, "taskset", None)),
        harness=load_harness(getattr(config, "harness", None)),
    )
