import json

from math_verify import parse, verify

import verifiers as vf
from verifiers.errors import SandboxError
from verifiers.utils.data_utils import extract_boxed_answer, load_example_dataset


def python_session() -> dict[str, list[str]]:
    return {"history": []}


async def python(code: str, sandbox, session) -> str:
    """Execute Python code in the rollout sandbox."""
    history = session.setdefault("history", [])
    script = f"""
import ast
import contextlib
import io
import traceback

history = {json.dumps(history)}
code = {json.dumps(code)}
namespace = {{}}

try:
    for snippet in history:
        exec(compile(snippet, "<history>", "exec"), namespace, namespace)

    tree = ast.parse(code, "<tool>", "exec")
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            prefix = ast.Module(body=tree.body[:-1], type_ignores=[])
            exec(compile(prefix, "<tool>", "exec"), namespace, namespace)
            expression = ast.Expression(tree.body[-1].value)
            result = eval(compile(expression, "<tool>", "eval"), namespace, namespace)
            if result is not None:
                print(repr(result))
        else:
            exec(compile(tree, "<tool>", "exec"), namespace, namespace)
    print(stdout.getvalue(), end="")
except BaseException:
    traceback.print_exc()
    raise SystemExit(1)
"""
    await sandbox.upload_bytes("/tmp/vf_python_tool.py", script.encode())
    result = await sandbox.execute("python /tmp/vf_python_tool.py")
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    if result.exit_code:
        raise SandboxError(f"Python command failed: {stderr}")
    history.append(code)
    return stdout.strip() or "(no output)"


@vf.reward(weight=1.0)
async def correct_answer(task, state) -> float:
    completion = state.get("completion") or []
    messages = vf.get_messages(completion, role="assistant")
    response_text = str(messages[-1].content or "") if messages else ""
    response = extract_boxed_answer(response_text)
    answer = str(task["answer"])
    if not response or len(response) > 50_000:
        return 0.0

    try:
        parsed_answer = parse(rf"\boxed{{{answer}}}", parsing_timeout=5)
        parsed_response = parse(rf"\boxed{{{response}}}", parsing_timeout=5)
        return float(verify(parsed_answer, parsed_response, timeout_seconds=5))
    except BaseException:
        return 0.0


@vf.cleanup(priority=10)
async def collect_python_commands(task, state):
    state["commands"] = list(state.get("sandbox_commands", []))
    state.pop("sandbox_commands", None)


def build_system_prompt(pip_install_packages: str = "numpy sympy scipy") -> str:
    pip_install_prompt = (
        f"In addition to the Python standard library, you have access to: {pip_install_packages}."
        if pip_install_packages.strip()
        else "You may only use the Python standard library."
    )
    return (
        "Use Python for all calculations. Give your answer inside \\boxed{}."
        "\n\n"
        f"{pip_install_prompt}"
    )


class MathPythonTasksetConfig(vf.TasksetConfig):
    source: str = f"{__name__}:source"
    system_prompt: str = build_system_prompt()
    rewards: list[vf.CallableConfig] = [
        vf.CallableConfig(fn=f"{__name__}:correct_answer")
    ]
    dataset_name: str = "math"
    dataset_split: str = "train"
    num_train_examples: int = -1


class MathPythonHarnessConfig(vf.HarnessConfig):
    max_turns: int = 100
    toolsets: dict[str, dict[str, str | int]] = {
        "python": {
            "fn": f"{__name__}:load_toolset",
            "pip_install_packages": "numpy sympy scipy",
            "sandbox_cpu_cores": 1,
            "sandbox_memory_gb": 2,
            "sandbox_disk_size_gb": 5,
            "sandbox_gpu_count": 0,
            "sandbox_timeout_minutes": 60,
            "sandbox_timeout_per_command_seconds": 60,
        }
    }


class MathPythonEnvConfig(vf.EnvConfig):
    taskset: MathPythonTasksetConfig = MathPythonTasksetConfig()
    harness: MathPythonHarnessConfig = MathPythonHarnessConfig()


def source(
    dataset_name: str = "math",
    dataset_split: str = "train",
    num_train_examples: int = -1,
):
    dataset = load_example_dataset(
        dataset_name,
        dataset_split,
        n=num_train_examples,
    )
    for index, row in enumerate(dataset):
        yield {
            **row,
            "example_id": index,
            "prompt": [{"role": "user", "content": row["question"]}],
        }


def load_toolset(
    pip_install_packages: str = "numpy sympy scipy",
    sandbox_cpu_cores: int = 1,
    sandbox_memory_gb: int = 2,
    sandbox_disk_size_gb: int = 5,
    sandbox_gpu_count: int = 0,
    sandbox_timeout_minutes: int = 60,
    sandbox_timeout_per_command_seconds: int = 60,
    config=None,
):
    packages = pip_install_packages.split() if pip_install_packages.strip() else []
    return vf.Toolset(
        tools=[python],
        write=True,
        objects={"python_session": python_session},
        bindings={"python.session": "objects.python_session"},
        sandbox={
            "image": "python:3.11-slim",
            "scope": "group",
            "cpu_cores": sandbox_cpu_cores,
            "memory_gb": sandbox_memory_gb,
            "disk_size_gb": sandbox_disk_size_gb,
            "gpu_count": sandbox_gpu_count,
            "timeout_minutes": sandbox_timeout_minutes,
            "command_timeout": sandbox_timeout_per_command_seconds,
            "packages": packages,
        },
        cleanups=[collect_python_commands],
        config=config,
    )


def load_taskset(
    config: MathPythonTasksetConfig = MathPythonTasksetConfig(),
) -> vf.Taskset:
    return vf.Taskset(config=config)


def load_harness(
    config: MathPythonHarnessConfig = MathPythonHarnessConfig(),
) -> vf.Harness:
    return vf.Harness(config=config)


def load_v1_environment(
    config: MathPythonEnvConfig = MathPythonEnvConfig(),
) -> vf.Env:
    return vf.Env(
        taskset=load_taskset(config=config.taskset),
        harness=load_harness(config=config.harness),
    )


load_environment = load_v1_environment
