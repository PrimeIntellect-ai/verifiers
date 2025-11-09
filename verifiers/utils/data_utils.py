# NOTE: Helper functions for example datasets. Not intended for core functionality.

import random
from typing import Any, Callable, cast

from datasets import Dataset, concatenate_datasets, load_dataset

from verifiers.types import ChatMessage

### PROMPTS ###

THINK_BOXED_SYSTEM_PROMPT = "Think step-by-step inside <think>...</think> tags. \
    Then, give your final answer inside \\boxed{}."

### https://github.com/huggingface/lighteval/blob/ecef2c662b9418866b6447d33b5e7d5dedd74af8/src/lighteval/tasks/default_prompts.py#L1474
BOXED_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
###############


def format_dataset(
    dataset: Dataset,
    system_prompt: str | None = None,
    few_shot: list[ChatMessage] | None = None,
    question_key: str = "question",
    answer_key: str = "answer",
    map_kwargs: dict = {},
) -> Dataset:
    """
    Create `example_id` and `prompt` columns if not present.
    """
    # if "id" column is present and not int, rename it to "src_id"
    if "example_id" in dataset.column_names and not isinstance(
        dataset["example_id"][0], int
    ):
        dataset = dataset.rename_column("example_id", "src_id")
    if "example_id" not in dataset.column_names:
        dataset = dataset.add_column("example_id", range(len(dataset)))  # type: ignore

    # extract format_prompt as a standalone function to avoid capturing self
    def format_prompt_fn(prompt_str: str) -> list[ChatMessage]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if few_shot:
            messages.extend(few_shot)
        messages.append({"role": "user", "content": prompt_str})
        return messages

    if "prompt" not in dataset.column_names:
        if answer_key == "answer":
            dataset = dataset.map(
                lambda x: {
                    "prompt": format_prompt_fn(x[question_key]),
                },
                **map_kwargs,
            )
        else:
            dataset = dataset.map(
                lambda x: {
                    "prompt": format_prompt_fn(x[question_key]),
                    "answer": x[answer_key],
                },
                **map_kwargs,
            )
    assert "example_id" in dataset.column_names
    assert "prompt" in dataset.column_names
    return dataset


def extract_boxed_answer(text: str) -> str:
    def find_matching_brace(s: str, start: int) -> int:
        count = 1
        i = start
        while i < len(s) and count > 0:
            if s[i] == "{":
                count += 1
            elif s[i] == "}":
                count -= 1
            i += 1
        return i - 1 if count == 0 else -1

    # Find last \boxed{
    boxed_start = text.rfind("\\boxed{")
    if boxed_start == -1:
        return text
    # Find the content between the braces
    content_start = boxed_start + 7  # len('\\boxed{')
    closing_brace = find_matching_brace(text, content_start)

    if closing_brace == -1:
        return text

    return text[content_start:closing_brace]


def strip_non_numeric(text: str) -> str:
    return "".join(c for c in text if c.isdigit() or c == ".")


def extract_hash_answer(text: str) -> str:
    if "####" not in text:
        return text
    return text.split("####")[1].strip()


def get_preprocess_fn(name: str) -> Callable[[dict], dict]:
    if name == "aime2024":

        def preprocess_aime2024(x: dict[str, Any]) -> dict[str, Any]:
            return {
                "question": x["problem"],
                "answer": str(int(x["answer"])),
            }

        return preprocess_aime2024
    elif name == "aime2025":

        def preprocess_aime2025(x: dict[str, Any]) -> dict[str, Any]:
            return {
                "question": x["question"],
                "answer": strip_non_numeric(x["answer"]),
            }

        return preprocess_aime2025
    elif name == "amc2023":

        def preprocess_amc2023(x: dict[str, Any]) -> dict[str, Any]:
            return {
                "question": x["problem"],
                "answer": x["answer"],
            }

        return preprocess_amc2023
    elif name in ["gpqa_diamond", "gpqa_main"]:

        def preprocess_gpqa(x: dict[str, Any]) -> dict[str, Any]:
            q = x["Question"]
            letters = ["A", "B", "C", "D"]
            random.shuffle(letters)
            itos = {k: v for k, v in enumerate(letters)}
            ans = {
                itos[0]: x["Correct Answer"],
                itos[1]: x["Incorrect Answer 1"],
                itos[2]: x["Incorrect Answer 2"],
                itos[3]: x["Incorrect Answer 3"],
            }
            question = f"Question: {q}\n\n"
            question += f"A: {ans['A']}\n"
            question += f"B: {ans['B']}\n"
            question += f"C: {ans['C']}\n"
            question += f"D: {ans['D']}"

            return {
                "question": question,
                "answer": itos[0],
            }

        return preprocess_gpqa
    elif name == "gsm8k":

        def preprocess_gsm8k(x: dict[str, Any]) -> dict[str, Any]:
            return {
                "question": x["question"],
                "answer": extract_hash_answer(x["answer"]),
            }

        return preprocess_gsm8k
    elif name == "math":

        def preprocess_math(x: dict[str, Any]) -> dict[str, Any]:
            return {
                "question": x["problem"],
                "answer": extract_boxed_answer(x["solution"]),
            }

        return preprocess_math
    elif name == "math500":

        def preprocess_math500(x: dict[str, Any]) -> dict[str, Any]:
            return {
                "question": x["problem"],
                "answer": x["answer"],
            }

        return preprocess_math500
    elif name == "mmlu":
        mmlu_map = ["A", "B", "C", "D"]

        def preprocess_mmlu(x: dict[str, Any]) -> dict[str, Any]:
            options = x["choices"]
            answer = x["answer"]
            question = f"Question: {x['question']}\n"
            for i, option in enumerate(options):
                question += f"\n{mmlu_map[i]}: {option}"
            return {
                "question": question,
                "temp_answer": mmlu_map[answer],
            }

        return preprocess_mmlu
    elif name == "mmlu_pro":
        mmlu_map = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

        def preprocess_mmlu(x: dict[str, Any]) -> dict[str, Any]:
            options = x["options"]
            answer = x["answer"]
            question = f"Question: {x['question']}\n"
            for i, option in enumerate(options):
                question += f"\n{mmlu_map[i]}: {option}"
            return {
                "question": question,
                "answer": answer,
            }

        return preprocess_mmlu
    elif name == "openbookqa":

        def preprocess_openbookqa(x: dict[str, Any]) -> dict[str, Any]:
            choices_texts = x["choices"]["text"]
            choices_labels = x["choices"]["label"]

            formatted_choices = []
            for i in range(len(choices_labels)):
                formatted_choices.append(f"{choices_labels[i]}. {choices_texts[i]}")

            question = f"Question: {x['question_stem']}\n\nChoices:\n" + "\n".join(
                formatted_choices
            )
            return {
                "question": question,
                "answer": x["answerKey"],
            }

        return preprocess_openbookqa
    elif name in ["openrs", "openrs_easy", "openrs_hard"]:

        def preprocess_openrs(x: dict[str, Any]) -> dict[str, Any]:
            return {
                "question": x["problem"],
                "answer": x["answer"],
            }

        return preprocess_openrs
    elif name == "prime_code":

        def preprocess_prime_code(x: dict[str, Any]) -> dict[str, Any]:
            return {
                "question": x["prompt"],
                "answer": x["verification_info"],
            }

        return preprocess_prime_code
    else:
        raise ValueError(f"Dataset {name} not supported for preprocess_dataset.")


def load_example_dataset(
    name: str = "gsm8k", split: str | None = None, n: int | None = None, seed: int = 0
) -> Dataset:
    if name == "aime2024":
        if split is None:
            split = "train"
        dataset = load_dataset("HuggingFaceH4/aime_2024")[split]
    elif name == "aime2025":
        if split is None:
            split = "test"
        aime_i = cast(
            Dataset, load_dataset("opencompass/AIME2025", "AIME2025-I")[split]
        )
        aime_ii = cast(
            Dataset, load_dataset("opencompass/AIME2025", "AIME2025-II")[split]
        )
        dataset = concatenate_datasets([aime_i, aime_ii])
    elif name == "amc2023":
        if split is None:
            split = "train"
        dataset = load_dataset("knoveleng/AMC-23")[split]
    elif name == "gpqa_diamond":
        if split is None:
            split = "train"
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond")[split]
    elif name == "gpqa_main":
        if split is None:
            split = "train"
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_main")[split]
    elif name == "gsm8k":
        if split is None:
            split = "test"
        dataset = load_dataset("openai/gsm8k", "main")[split]
    elif name == "math":
        if split is None:
            split = "train"
        dataset = load_dataset("chiayewken/competition_math")[split]
    elif name == "math500":
        if split is None:
            split = "test"
        dataset = load_dataset("HuggingFaceH4/MATH-500")[split]
    elif name == "mmlu":
        if split is None:
            split = "dev"
        dataset = load_dataset("cais/mmlu", "all")[split]
    elif name == "mmlu_pro":
        if split is None:
            split = "validation"
        dataset = load_dataset("TIGER-Lab/MMLU-Pro")[split]
    elif name == "openbookqa":
        if split is None:
            split = "train"
        dataset = load_dataset("allenai/openbookqa", "main")[split]
    elif name == "openrs":
        if split is None:
            split = "train"
        dataset = load_dataset("knoveleng/open-rs")[split]
    elif name == "openrs_easy":
        if split is None:
            split = "train"
        dataset = cast(Dataset, load_dataset("knoveleng/open-rs")[split])
        dataset = dataset.filter(lambda x: x["level"] == "Easy")
    elif name == "openrs_hard":
        if split is None:
            split = "train"
        dataset = cast(Dataset, load_dataset("knoveleng/open-rs")[split])
        dataset = dataset.filter(lambda x: x["level"] == "Hard")
    elif name == "prime_code":
        if split is None:
            split = "train"
        dataset = cast(
            Dataset, load_dataset("PrimeIntellect/verifiable-coding-problems")[split]
        )
        dataset = dataset.filter(
            lambda x: x["prompt"].startswith(
                "Solve the following coding problem using the programming language python:"
            )
        )
    else:
        raise ValueError(
            f"Dataset {name} not supported for preprocess_dataset. \nPlease ensure that the dataset is formatted with 'prompt' (str) and 'answer' (str) keys."
        )

    preprocess_fn = get_preprocess_fn(name)
    dataset = cast(Dataset, dataset)
    if n is not None and n > 0:
        dataset = dataset.shuffle(seed=seed).select(range(n))
    dataset = dataset.map(
        preprocess_fn,
        num_proc=10,
        remove_columns=dataset.column_names,
    )
    if "temp_answer" in dataset.column_names:
        dataset = dataset.rename_column("temp_answer", "answer")
    return dataset
