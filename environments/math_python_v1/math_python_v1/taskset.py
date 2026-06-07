from math_verify import parse, verify
import verifiers.v1 as vf
from verifiers.utils.data_utils import extract_boxed_answer, load_example_dataset

from .servers.toolset import PythonToolsetConfig


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
    toolsets: list[vf.ToolsetConfig] = [PythonToolsetConfig()]
    pip_install_packages: str = "numpy sympy scipy"
    dataset_name: str = "math"
    dataset_split: str = "train"
    num_train_examples: int = -1


class MathPythonHarnessConfig(vf.HarnessConfig):
    max_turns: int = 100


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


def load_taskset(config: MathPythonTasksetConfig) -> MathPythonTaskset:
    if "system_prompt" not in config.model_fields_set:
        config = config.model_copy(
            update={"system_prompt": build_system_prompt(config.pip_install_packages)}
        )
    return MathPythonTaskset(config=config)


def load_harness(config: MathPythonHarnessConfig) -> vf.Harness:
    return vf.Harness(config=config)
