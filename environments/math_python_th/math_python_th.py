from __future__ import annotations

import verifiers.v1 as vf
from verifiers.errors import SandboxError
from verifiers.v1.utils.tool_utils import load_tools_from_state


async def bash(command, sandbox):
    result = await sandbox.execute(command)
    return {
        "stdout": result.stdout or "",
        "stderr": result.stderr or "",
        "returncode": result.exit_code,
    }


async def python(expression, bash):
    result = await bash(command=f"python - <<'PY'\nprint({expression})\nPY")
    if result["returncode"]:
        raise SandboxError(f"Python command failed: {result['stderr']}")
    return result["stdout"].strip()


@vf.reward(weight=1.0)
async def exact_answer(task, state) -> float:
    return float(str(state["answer"]) == str(task["answer"]))


@vf.cleanup(priority=10)
async def collect_sandbox_commands(task, state):
    state["commands"] = list(state.get("sandbox_commands", []))
    state.pop("sandbox_commands", None)


def source():
    return [
        {
            "prompt": "Use Python to calculate 17 * 23.",
            "expression": "17 * 23",
            "answer": "391",
        },
        {
            "prompt": "Use Python to calculate (144 / 12) + 9.",
            "expression": "(144 / 12) + 9",
            "answer": "21.0",
        },
    ]


def load_toolset(config=None):
    return vf.Toolset(
        tools=[bash, python],
        hide=["bash"],
        write=True,
        sandbox={
            "image": "python:3.11-slim",
            "scope": "group",
        },
        bindings={
            "python.bash": "tools.bash",
        },
        cleanup=[collect_sandbox_commands],
        config=config,
    )


async def math_program(task, state):
    tools = load_tools_from_state(state)
    state["answer"] = await tools["python"](expression=task["expression"])
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
        program=math_program,
        toolsets=[load_toolset(getattr(config, "toolset", None))],
        config=config,
    )


def load_environment(config=None):
    return vf.Env(
        taskset=load_taskset(getattr(config, "taskset", None)),
        harness=load_harness(getattr(config, "harness", None)),
    )
