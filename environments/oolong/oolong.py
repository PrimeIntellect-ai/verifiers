"""
Oolong Long-Context Evaluation Environment.

Implements the Oolong benchmark for evaluating long-context understanding
capabilities of language models using the RLM (Recursive Language Model) environment.

The Oolong benchmark consists of two datasets:
- oolong-synth: Synthetic long-context evaluation tasks
- oolong-real: Real-world long-context evaluation tasks

The model operates in an RLM REPL environment where it can:
- Write Python code to explore the large context
- Make recursive sub-LLM calls if needed
- Return the final answer programmatically
"""

import os
from typing import Literal

from datasets import load_dataset
from openai import AsyncOpenAI

import verifiers as vf
from verifiers.envs.rlm_env import RLMEnv


def load_environment(
    subset: Literal["synth", "synth_with_labels", "real"] = "synth",
    split: Literal["validation", "test"] = "validation",
    shuffle: bool = False,
    max_iterations: int = 30,
    max_output_length: int = 8192,
    judge_model: str = "gpt-5-mini",
    judge_api_key_var: str = "OPENAI_API_KEY",
    **kwargs,
) -> vf.Environment:
    """
    Load the Oolong long-context evaluation environment.

    Args:
        subset: Which subset to use:
            - "synth": Synthetic dataset with context_window_text
            - "synth_with_labels": Synthetic dataset with context_window_text_with_labels
            - "real": Real-world dataset with context_window_text
        split: Dataset split to use ("validation" or "test")
        shuffle: Whether to shuffle the dataset (useful for sampling different examples)
        max_iterations: Maximum REPL iterations before stopping
        max_output_length: Maximum length of code execution output
        judge_model: Model to use for judging answer correctness
        judge_api_key_var: Environment variable containing the API key for the judge model
        **kwargs: Additional arguments passed to RLMEnv (e.g., interception_host)

    Returns:
        Configured RLMEnv instance
    """
    # Determine HuggingFace dataset name and context column based on subset
    if subset in ("synth", "synth_with_labels"):
        hf_dataset_name = "oolongbench/oolong-synth"
        context_column = (
            "context_window_text"
            if subset == "synth"
            else "context_window_text_with_labels"
        )
    else:  # "real"
        hf_dataset_name = "oolongbench/oolong-real"
        context_column = "context_window_text"

    # Load the dataset from HuggingFace
    raw_dataset = load_dataset(hf_dataset_name, split=split)

    # Transform dataset into the required format for RLMEnv
    def transform_example(example, idx):
        return {
            "example_id": idx,
            "prompt": [{"role": "user", "content": example["question"]}],
            "task": "oolong",
            "answer": example["answer"],
            "info": {"context": example[context_column]},
        }

    dataset = raw_dataset.map(transform_example, with_indices=True, remove_columns=raw_dataset.column_names)

    if shuffle:
        dataset = dataset.shuffle()

    # Setup judge for evaluating answer correctness
    judge_client = AsyncOpenAI(api_key=os.getenv(judge_api_key_var))

    judge_prompt_template = """Given a ground truth answer and a response, determine if the response is correct.

Question:
```
{question}
```

Ground truth answer:
```
{ground_truth}
```

Response:
```
{response}
```

Respond either "yes" or "no" only."""

    # Reward function using model-as-judge
    async def judge_reward(
        prompt: vf.Messages,
        completion: vf.Messages,
        answer: str,
        state: vf.State,
        **_kwargs,
    ) -> float:
        """Reward based on judge model evaluation."""
        question = prompt[-1]["content"] if prompt else ""
        response = state.get("final_answer", "")
        ground_truth = state.get("answer", "")

        judge_prompt = judge_prompt_template.format(
            question=question,
            ground_truth=ground_truth,
            response=response,
        )

        judge_response = await judge_client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": judge_prompt}],
        )
        judge_answer = judge_response.choices[0].message.content or ""
        return 1.0 if "yes" in judge_answer.lower() else 0.0

    # Metrics-only reward functions (0-weight) for comparison
    def exact_match_reward(state: vf.State) -> float:
        """Metric: exact match with expected answer."""
        final_answer = state.get("final_answer", "").strip()
        expected = state.get("answer", "").strip()
        return 1.0 if final_answer == expected else 0.0

    def contains_answer_reward(state: vf.State) -> float:
        """Metric: final answer contains the expected value."""
        final_answer = state.get("final_answer", "").strip()
        expected = state.get("answer", "").strip()
        return 1.0 if expected in final_answer else 0.0

    rubric = vf.Rubric(
        funcs=[judge_reward, exact_match_reward, contains_answer_reward],
        weights=[1.0, 0.0, 0.0],
    )

    env = RLMEnv(
        max_iterations=max_iterations,
        max_output_length=max_output_length,
        context_key="context",
        dataset=dataset,
        rubric=rubric,
        **kwargs,
    )

    return env
