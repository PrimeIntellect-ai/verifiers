from copy import deepcopy
from typing import Any

from datasets import Dataset, load_dataset

import verifiers as vf


def _as_int_key(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _format_ifbench_row(example: dict[str, Any]) -> dict[str, Any]:
    prompt = example["prompt"]
    return {
        "question": prompt,
        "answer": "",
        "info": {
            "key": example.get("key"),
            "prompt": prompt,
            "instruction_id_list": example["instruction_id_list"],
            "kwargs": example["kwargs"],
        },
    }


def _select_examples(dataset: Dataset, count: int) -> Dataset:
    if count == -1:
        return dataset
    return dataset.select(range(min(count, len(dataset))))


def _clean_kwargs(kwargs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in item.items() if value is not None}
        for item in deepcopy(kwargs)
    ]


class IFBenchRubric(vf.Rubric):
    def __init__(self, mode: str = "loose"):
        if mode not in {"strict", "loose"}:
            raise ValueError("mode must be 'strict' or 'loose'")
        self.mode = mode
        super().__init__(funcs=[self.ifbench_reward])

    def ifbench_reward(self, parser, completion, info, **kwargs) -> float:
        from ifbench_official.evaluation_lib import InputExample
        from ifbench_official.evaluation_lib import test_instruction_following_loose
        from ifbench_official.evaluation_lib import test_instruction_following_strict

        prompt = info["prompt"]
        response = parser.parse_answer(completion) or ""
        example = InputExample(
            key=_as_int_key(info.get("key")),
            instruction_id_list=info["instruction_id_list"],
            prompt=prompt,
            kwargs=_clean_kwargs(info["kwargs"]),
        )
        evaluator = (
            test_instruction_following_strict
            if self.mode == "strict"
            else test_instruction_following_loose
        )
        result = evaluator(example, {prompt: response})
        return float(result.follow_all_instructions)


def load_environment(
    dataset_name: str = "allenai/IFBench_test",
    dataset_split: str = "train",
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
    reward_mode: str = "loose",
    system_prompt: str | None = None,
) -> vf.Environment:
    def build_dataset() -> Dataset:
        dataset = load_dataset(dataset_name, split=dataset_split).map(
            _format_ifbench_row,
            remove_columns=["key", "prompt", "instruction_id_list", "kwargs"],
        )
        return _select_examples(dataset, num_train_examples)

    def build_eval_dataset() -> Dataset:
        dataset = load_dataset(dataset_name, split=dataset_split).map(
            _format_ifbench_row,
            remove_columns=["key", "prompt", "instruction_id_list", "kwargs"],
        )
        return _select_examples(dataset, num_eval_examples)

    parser = vf.Parser()
    rubric = IFBenchRubric(mode=reward_mode)
    return vf.SingleTurnEnv(
        dataset=build_dataset,
        eval_dataset=build_eval_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )
