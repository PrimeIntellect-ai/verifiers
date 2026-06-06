import ast
import contextlib
import io
import sys
import traceback

from math_verify import parse, verify
import verifiers.v1 as vf
from verifiers.utils.data_utils import extract_boxed_answer, load_example_dataset


def build_system_prompt(pip_install_packages: str = "numpy sympy scipy") -> str:
    pip_install_prompt = (
        f"In addition to the Python standard library, you have access to: {pip_install_packages}."
        if pip_install_packages.strip()
        else "You may only use the Python standard library."
    )
    return (
        "Use the python tool for calculations when useful. Give your answer "
        "inside \\boxed{}.\n\n"
        f"{pip_install_prompt}"
    )


class MathPythonTasksetConfig(vf.TasksetConfig):
    system_prompt: str | None = None
    dataset_name: str = "math"
    dataset_split: str = "train"
    num_train_examples: int = -1


class MathPythonHarnessConfig(vf.HarnessConfig):
    max_turns: int = 100
    pip_install_packages: str = "numpy sympy scipy"


class MathPythonEnvConfig(vf.EnvConfig):
    taskset: MathPythonTasksetConfig = MathPythonTasksetConfig()
    harness: MathPythonHarnessConfig = MathPythonHarnessConfig()


class MathPythonTask(vf.Task):
    question: str
    answer: str


class MathPythonTaskset(vf.Taskset[MathPythonTasksetConfig]):
    task_type = MathPythonTask

    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        if split == "eval":
            return []
        dataset = load_example_dataset(
            self.config.dataset_name,
            self.config.dataset_split,
            n=self.config.num_train_examples,
        )
        return [
            {
                **row,
                "row_id": index,
                "prompt": [{"role": "user", "content": row["question"]}],
            }
            for index, row in enumerate(dataset)
        ]

    def load_toolsets(self, config: MathPythonTasksetConfig) -> list[vf.Toolset]:
        _ = config
        return [
            vf.Toolset(
                name="python",
                server=vf.MCPServerSpec(
                    command=[sys.executable, __file__, "--tool-server"]
                ),
            )
        ]

    @vf.reward(weight=1.0)
    async def correct_answer(self, task: MathPythonTask, state: vf.State) -> float:
        messages = vf.get_messages(state.completion or [], role="assistant")
        response_text = str(messages[-1].content or "") if messages else ""
        response = extract_boxed_answer(response_text)
        if not response or len(response) > 50_000:
            return 0.0
        try:
            parsed_answer = parse(rf"\boxed{{{task.answer}}}", parsing_timeout=5)
            parsed_response = parse(rf"\boxed{{{response}}}", parsing_timeout=5)
            return float(verify(parsed_answer, parsed_response, timeout_seconds=5))
        except BaseException:
            return 0.0


HISTORY: list[str] = []


def execute_python(code: str) -> str:
    namespace: dict[str, object] = {}
    for snippet in HISTORY:
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
    HISTORY.append(code)
    return stdout.getvalue().strip() or "(no output)"


def run_tool_server() -> None:
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("math-python")

    @mcp.tool()
    def python(code: str) -> dict:
        try:
            content = execute_python(code)
        except BaseException:
            content = traceback.format_exc()
        return {
            "content": content,
            "scratch": {"python_history": list(HISTORY)},
        }

    mcp.run(transport="stdio")


def load_environment(config: MathPythonEnvConfig) -> vf.Env:
    taskset_config = config.taskset
    if "system_prompt" not in taskset_config.model_fields_set:
        taskset_config = taskset_config.model_copy(
            update={
                "system_prompt": build_system_prompt(
                    config.harness.pip_install_packages
                )
            }
        )
    return vf.Env(
        taskset=MathPythonTaskset(config=taskset_config),
        harness=vf.Harness(config=config.harness),
        runtime=config.runtime,
    )


if __name__ == "__main__" and sys.argv[1:] == ["--tool-server"]:
    run_tool_server()
