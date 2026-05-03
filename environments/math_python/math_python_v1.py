from __future__ import annotations

import json

from math_verify import parse, verify

import verifiers.v1 as vf
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
    response_text = ""
    for message in reversed(completion):
        if message.get("role") == "assistant":
            response_text = str(message.get("content") or "")
            break
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


def source(
    dataset_name: str = "math",
    dataset_split: str = "train",
    num_train_examples: int = -1,
    pip_install_packages: str = "numpy sympy scipy",
):
    dataset = load_example_dataset(
        dataset_name,
        dataset_split,
        n=num_train_examples,
    )
    pip_install_prompt = (
        f"In addition to the Python standard library, you have access to: {pip_install_packages}."
        if pip_install_packages.strip()
        else "You may only use the Python standard library."
    )
    system_prompt = (
        "Use Python for all calculations. Give your answer inside \\boxed{}."
    )
    system_prompt += "\n\n" + pip_install_prompt
    for index, row in enumerate(dataset):
        yield {
            **row,
            "example_id": index,
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": row["question"]},
            ],
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
        cleanup=[collect_python_commands],
        config=config,
    )


def load_taskset(
    dataset_name: str = "math",
    dataset_split: str = "train",
    num_train_examples: int = -1,
    pip_install_packages: str = "numpy sympy scipy",
    config=None,
):
    return vf.Taskset(
        source=lambda: source(
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            num_train_examples=num_train_examples,
            pip_install_packages=pip_install_packages,
        ),
        rewards=[correct_answer],
        config=config,
    )


def load_harness(
    max_turns: int = 100,
    pip_install_packages: str = "numpy sympy scipy",
    sandbox_cpu_cores: int = 1,
    sandbox_memory_gb: int = 2,
    sandbox_disk_size_gb: int = 5,
    sandbox_gpu_count: int = 0,
    sandbox_timeout_minutes: int = 60,
    sandbox_timeout_per_command_seconds: int = 60,
    config=None,
):
    return vf.Harness(
        toolsets=[
            load_toolset(
                pip_install_packages=pip_install_packages,
                sandbox_cpu_cores=sandbox_cpu_cores,
                sandbox_memory_gb=sandbox_memory_gb,
                sandbox_disk_size_gb=sandbox_disk_size_gb,
                sandbox_gpu_count=sandbox_gpu_count,
                sandbox_timeout_minutes=sandbox_timeout_minutes,
                sandbox_timeout_per_command_seconds=sandbox_timeout_per_command_seconds,
            )
        ],
        max_turns=max_turns,
        config=config,
    )


def load_v1_environment(
    dataset_name: str = "math",
    dataset_split: str = "train",
    num_train_examples: int = -1,
    max_turns: int = 100,
    max_startup_wait_seconds: int = 60,
    pip_install_packages: str = "numpy sympy scipy",
    sandbox_cpu_cores: int = 1,
    sandbox_memory_gb: int = 2,
    sandbox_disk_size_gb: int = 5,
    sandbox_gpu_count: int = 0,
    sandbox_timeout_minutes: int = 60,
    sandbox_timeout_per_command_seconds: int = 60,
    sandbox_client_max_workers: int = 50,
    **kwargs,
) -> vf.Env:
    _ = max_startup_wait_seconds, sandbox_client_max_workers
    if kwargs:
        raise TypeError(f"Unsupported v1 args: {sorted(kwargs)}")
    return vf.Env(
        taskset=load_taskset(
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            num_train_examples=num_train_examples,
            pip_install_packages=pip_install_packages,
        ),
        harness=load_harness(
            max_turns=max_turns,
            pip_install_packages=pip_install_packages,
            sandbox_cpu_cores=sandbox_cpu_cores,
            sandbox_memory_gb=sandbox_memory_gb,
            sandbox_disk_size_gb=sandbox_disk_size_gb,
            sandbox_gpu_count=sandbox_gpu_count,
            sandbox_timeout_minutes=sandbox_timeout_minutes,
            sandbox_timeout_per_command_seconds=sandbox_timeout_per_command_seconds,
        ),
    )
