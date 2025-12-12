import os

from datasets import load_dataset
from openai import AsyncOpenAI

import verifiers as vf


def load_environment(
    dataset_name: str = "PrimeIntellect/Hendrycks-Math",
    dataset_subset: str = "default",
    dataset_split: str = "train",
    judge_model: str = "gpt-4.1-mini",
    base_url: str = "http://0.0.0.0:8000/v1",
    api_key_var: str = "JUDGE_API_KEY",
):
    dataset = load_dataset(dataset_name, dataset_subset, split=dataset_split)
    judge_prompt = "Q: {question}\nA: {answer}\nGiven: {response}\nRespond with a score between 0.0 and 1.0."
    judge_client = AsyncOpenAI(
        base_url=base_url, api_key=os.getenv(api_key_var, "EMPTY")
    )
    rubric = vf.JudgeRubric(
        judge_client=judge_client,
        judge_model=judge_model,
        judge_prompt=judge_prompt,
    )
    vf_env = vf.SingleTurnEnv(
        dataset=dataset,
        system_prompt="You are a helpful assistant.",
        rubric=rubric,
    )

    return vf_env
