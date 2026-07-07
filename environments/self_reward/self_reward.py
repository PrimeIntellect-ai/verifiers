import os
import re

from datasets import load_dataset
from openai import AsyncOpenAI

import verifiers as vf


def load_environment(
    dataset_name: str = "PrimeIntellect/Hendrycks-Math",
    dataset_subset: str = "default",
    dataset_split: str = "train",
    judge_model: str = "gpt-4.1-mini",
    base_url: str = "https://api.openai.com/v1",
    api_key_var: str = "OPENAI_API_KEY",
):
    def build_dataset():
        return load_dataset(dataset_name, dataset_subset, split=dataset_split)

    judge_prompt = "Q: {question}\nA: {answer}\nGiven: {response}\nRespond with a score between 0.0 and 1.0."
    judge_client = AsyncOpenAI(
        base_url=base_url, api_key=os.getenv(api_key_var, "EMPTY")
    )
    rubric = vf.JudgeRubric(
        judge_client=judge_client,
        judge_model=judge_model,
        judge_prompt=judge_prompt,
    )

    async def judge_score(judge, prompt, completion, answer, state) -> float:
        judge_response = await judge(prompt, completion, answer, state)
        # Match either a fractional score ("X/Y", "X out of Y") or a bare
        # number, capturing the intended score (the numerator) so denominators
        # like "/1.0" aren't mistaken for the score and inflate the reward.
        number = r"0(?:\.\d+)?|1(?:\.0+)?"
        scores = re.findall(
            rf"\b({number})\s*(?:/|out of)\s*(?:{number})\b|\b({number})\b",
            judge_response,
            flags=re.IGNORECASE,
        )
        if not scores:
            return 0.0
        numerator, bare = scores[-1]
        return max(0.0, min(1.0, float(numerator or bare)))

    rubric.add_reward_func(judge_score, weight=1.0)

    vf_env = vf.SingleTurnEnv(
        dataset=build_dataset,
        system_prompt="You are a helpful assistant.",
        rubric=rubric,
    )

    return vf_env
