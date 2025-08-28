import os

from datasets import load_dataset
from openai import OpenAI

import verifiers as vf


def load_environment(
    dataset_name: str,
    model_name: str,
    base_url: str = "http://0.0.0.0:8000/v1",
    api_key_var: str = "JUDGE_API_KEY",
):
    judge_prompt = "Q: {question}\nA: {answer}\nGiven: {response}\nRespond with a score between 0.0 and 1.0."
    rubric = vf.JudgeRubric(
        judge_client=OpenAI(base_url=base_url, api_key=os.getenv(api_key_var, "EMPTY")),
        judge_model=model_name,
        judge_prompt=judge_prompt,
    )
    vf_env = vf.SingleTurnEnv(
        dataset=load_dataset(
            dataset_name, data_files="train"
        ),  # HF dataset with "question" and "answer" columns
        system_prompt="You are a helpful assistant.",
        rubric=rubric,
    )

    return vf_env
