import verifiers as vf
from datasets import Dataset, load_dataset

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

        dataset = load_dataset(dataset_name, dataset_subset, split=dataset_split)

        if question_key not in dataset.column_names:
            raise ValueError(
                f"Column '{question_key}' not found in dataset: {dataset.column_names}"
            )
        if answer_key not in dataset.column_names:
            raise ValueError(
                f"Column '{answer_key}' not found in dataset: {dataset.column_names}"
            )

        def process_example(example):
            question = example[question_key]
            answer = example[answer_key]
            return {
                "question": instruction_prompt + question + instruction_prompt_post,
                "answer": answer,
            }

        dataset = dataset.map(process_example)

        return dataset
