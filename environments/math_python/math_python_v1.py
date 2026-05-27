import json

from math_verify import parse, verify
from pydantic import model_validator

import verifiers as vf
from verifiers.errors import SandboxError
from verifiers.utils.data_utils import extract_boxed_answer, load_example_dataset


async def python(code: str, sandbox, state) -> str:
    """Execute Python code in the rollout sandbox."""
    history = state.setdefault("python_history", [])
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
    rewards: list[str] = ["correct_answer"]
    system_prompt: str | None = None
    dataset_name: str = "math"
    dataset_split: str = "train"
    num_train_examples: int = -1


class MathPythonHarnessConfig(vf.HarnessConfig):
    max_turns: int = 100
    pip_install_packages: str = "numpy sympy scipy"
    sandbox_cpu_cores: int = 1
    sandbox_memory_gb: int = 2
    sandbox_disk_size_gb: int = 5
    sandbox_gpu_count: int = 0
    sandbox_timeout_minutes: int = 60
    sandbox_timeout_per_command_seconds: int = 60


class MathPythonEnvConfig(vf.EnvConfig):
    taskset: MathPythonTasksetConfig = MathPythonTasksetConfig()
    harness: MathPythonHarnessConfig = MathPythonHarnessConfig()

    @model_validator(mode="after")
    def derive_taskset_system_prompt(self) -> "MathPythonEnvConfig":
        if "system_prompt" not in self.taskset.model_fields_set:
            self.taskset = self.taskset.model_copy(
                update={
                    "system_prompt": build_system_prompt(
                        self.harness.pip_install_packages
                    )
                }
            )
        return self


def load_tasks(
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


class MathPythonTaskset(vf.Taskset[MathPythonTasksetConfig]):
    def load_tasks(self) -> vf.Tasks:
        return load_tasks(
            dataset_name=self.config.dataset_name,
            dataset_split=self.config.dataset_split,
            num_train_examples=self.config.num_train_examples,
        )


class MathPythonHarness(vf.Harness[MathPythonHarnessConfig]):
    pass


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


def load_environment(config: MathPythonEnvConfig) -> vf.Env:
    harness = MathPythonHarness(config=config.harness)
    if "toolsets" not in config.harness.model_fields_set:
        harness.add_toolset(
            {
                "python": load_toolset(
                    pip_install_packages=config.harness.pip_install_packages,
                    sandbox_cpu_cores=config.harness.sandbox_cpu_cores,
                    sandbox_memory_gb=config.harness.sandbox_memory_gb,
                    sandbox_disk_size_gb=config.harness.sandbox_disk_size_gb,
                    sandbox_gpu_count=config.harness.sandbox_gpu_count,
                    sandbox_timeout_minutes=config.harness.sandbox_timeout_minutes,
                    sandbox_timeout_per_command_seconds=(
                        config.harness.sandbox_timeout_per_command_seconds
                    ),
                )
            }
        )
    return vf.Env(
        taskset=MathPythonTaskset(config=config.taskset),
        harness=harness,
    )
