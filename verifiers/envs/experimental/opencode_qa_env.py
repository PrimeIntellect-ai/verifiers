from typing import Any

from datasets import Dataset, load_dataset

import verifiers as vf
from verifiers.envs.experimental.opencode_env import OpenCodeEnv


class OpenCodeQAEnv(OpenCodeEnv):
    """Solve general QA problems in OpenCode."""

    DEFAULT_QUESTION_KEY = "question"
    DEFAULT_ANSWER_KEY = "answer"
    DEFAULT_INSTRUCTION_PROMPT = ""
    DEFAULT_INSTRUCTION_PROMPT_POST = ""

    def __init__(
        self,
        rubric: vf.Rubric,
        dataset_name: str,
        dataset_subset: str,
        dataset_split: str,
        question_key: str = DEFAULT_QUESTION_KEY,
        answer_key: str = DEFAULT_ANSWER_KEY,
        instruction_prompt: str = DEFAULT_INSTRUCTION_PROMPT,
        instruction_prompt_post: str = DEFAULT_INSTRUCTION_PROMPT_POST,
        **kwargs,
    ):
        dataset = self.construct_dataset(
            dataset_name,
            dataset_subset,
            dataset_split,
            question_key,
            answer_key,
            instruction_prompt,
            instruction_prompt_post,
        )

        super().__init__(dataset=dataset, rubric=rubric, **kwargs)

    def construct_dataset(
        self,
        dataset_name: str,
        dataset_subset: str,
        dataset_split: str,
        question_key: str,
        answer_key: str,
        instruction_prompt: str,
        instruction_prompt_post: str,
    ) -> Dataset:
        """Constructs a general QA dataset."""

        dataset_obj = load_dataset(dataset_name, dataset_subset, split=dataset_split)
        if not isinstance(dataset_obj, Dataset):
            raise TypeError(
                "Expected a Dataset for the requested split, got a different dataset type."
            )
        dataset = dataset_obj

        column_names = dataset.column_names
        if column_names is None:
            raise ValueError("Dataset has no columns.")

        if question_key not in column_names:
            raise ValueError(
                f"Column '{question_key}' not found in dataset: {column_names}"
            )
        if answer_key not in column_names:
            raise ValueError(
                f"Column '{answer_key}' not found in dataset: {column_names}"
            )

        def process_example(example: dict[str, Any]) -> dict[str, Any]:
            question = example[question_key]
            answer = example[answer_key]
            if not isinstance(question, str):
                question = str(question)
            return {
                "question": instruction_prompt + question + instruction_prompt_post,
                "answer": answer,
            }

        mapped_dataset = dataset.map(process_example)
        if not isinstance(mapped_dataset, Dataset):
            raise TypeError("Expected mapped dataset to be a Dataset.")

        return mapped_dataset
