import json
from typing import Any, Callable

import verifiers as vf
from datasets import Dataset

from utils.art_adapter import (
    art_config_to_tools,
    build_dataset_from_art_config,
    get_completion_tool_name,
)
from utils.verifiers_adapter import export_verifiers_env


class ARTParser(vf.Parser):
    def __init__(self, completion_tool_name: str):
        super().__init__()
        self.completion_tool_name = completion_tool_name

    def parse_answer(self, completion: vf.Messages) -> str | None:
        if not isinstance(completion, list):
            return super().parse_answer(completion)
        # find the last assistant tool-call with completion tool name
        for msg in reversed(completion):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                tool_calls = msg["tool_calls"] or []
                for tc in tool_calls:
                    try:
                        # handle both typed and dict tool-calls
                        if hasattr(tc, "function"):
                            name = tc.function.name
                            args_s = tc.function.arguments
                        else:
                            name = tc["function"]["name"]
                            args_s = tc["function"]["arguments"]
                        if name == self.completion_tool_name:
                            args = json.loads(args_s)
                            # answer field is any single value or "answer"
                            if isinstance(args, dict):
                                if "answer" in args:
                                    return str(args["answer"])
                                # fallback: stringified dict
                                return json.dumps(args)
                            return str(args)
                    except Exception:
                        continue
        return None


def load_environment(
    task_config_path: str | None = None,
    task_config_dict: dict | None = None,
    dataset: Dataset | None = None,
    eval_dataset: Dataset | None = None,
    max_turns: int = 10,
    use_llm_judge: bool = False,
    judge_model: str = "gpt-4.1-mini",
    judge_client: Any | None = None,
    judge_api_key_var: str = "OPENAI_API_KEY",
    **kwargs,
) -> vf.Environment:
    """Load ART framework adapter environment.

    If no datasets are provided, builds tiny train/eval datasets from the task config examples.
    """

    if task_config_path is None and task_config_dict is None:
        raise ValueError("Provide task_config_path or task_config_dict")
    if task_config_dict is None:
        with open(task_config_path, "r") as f:  # type: ignore[arg-type]
            task_config_dict = json.load(f)

    assert isinstance(task_config_dict, dict)
    completion_tool_name = get_completion_tool_name(task_config_dict)
    tools: list[Callable] = art_config_to_tools(task_config_dict)

    # default datasets from config examples if not supplied
    if dataset is None or eval_dataset is None:
        ds_train, ds_eval = build_dataset_from_art_config(task_config_dict)
        dataset = dataset or ds_train
        eval_dataset = eval_dataset or ds_eval

    parser = ARTParser(completion_tool_name=completion_tool_name)
    if use_llm_judge:
        rubric = vf.JudgeRubric(
            parser=parser, judge_model=judge_model, judge_client=judge_client
        )
    else:

        class ExactMatchRubric(vf.Rubric):
            async def correct_answer(
                self, parser: vf.Parser, completion: vf.Messages, answer: str, **_: Any
            ) -> float:
                pred = parser.parse_answer(completion) or ""
                return 1.0 if str(pred) == str(answer) and pred != "" else 0.0

        rubric = ExactMatchRubric(parser=parser)
        rubric.add_reward_func(rubric.correct_answer, weight=1.0)  # type: ignore

    env = vf.ToolEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        parser=parser,
        rubric=rubric,
        tools=tools,
        max_turns=max_turns,
        env_id="art_framework",
        env_args={
            "task_config": task_config_dict,
            "use_llm_judge": use_llm_judge,
            "judge_model": judge_model,
        },
        **kwargs,
    )
    return env


__all__ = [
    "load_environment",
    "ARTParser",
    "export_verifiers_env",
]
