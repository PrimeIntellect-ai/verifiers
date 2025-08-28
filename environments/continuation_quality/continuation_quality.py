import os
import random

from datasets import load_dataset
from openai import OpenAI

import verifiers as vf

_rand = random.Random(777)


def make_cut(text: str) -> dict[str, str]:
    """Makes a random cut somewhere in the paragraph"""
    n_spaces = text.count(" ")
    # mostly split near the middle
    split_space = int(_rand.normalvariate(0.5, 0.15) * n_spaces)
    # make sure there's at least ~25 words before and after the split point
    split_space = min(n_spaces - 25, max(25, split_space))
    idx = -1
    for _ in range(split_space):
        idx = text.find(" ", idx + 1)
    return {"prompt": text[:idx], "answer": text[idx:]}


def load_environment(
    dataset_name: str = "agentlans/wikipedia-paragraphs",
    dataset_split: str | None = "train",
    dataset_key: str = "text",
    judge_model: str = "gpt-4.1-mini",
    judge_base_url: str = "https://api.openai.com/v1",
    judge_api_key_var: str = "OPENAI_API_KEY",
) -> vf.Environment:
    dataset = load_dataset(dataset_name, split=dataset_split)
    # only accept examples with >~100 words or so
    dataset = dataset.filter(lambda x: x[dataset_key].count(" ") > 100)
    dataset = dataset.map(lambda x: make_cut(x[dataset_key]))
    dataset = dataset.shuffle(seed=777)

    judge_client = OpenAI(api_key=os.getenv(judge_api_key_var), base_url=judge_base_url)
    judge_prompt = """Evaluate this base model contination from a prefix, compared to the true continuation from Wikipedia.

<prefix>
{question}
</prefix>

<true_continuation>
{answer}
</true_continuation>

<model_continuation>
{response}
</model_continuation>

Provide a letter grade from A-F where:
- A: Smooth prose, facts are mostly accurate w.r.t the true continuation
- B: Smooth prose, regardless of factual accuracy
- C: Some awkward wording, spacing, or punctuation
- D: Inclusions of awkward or glitchy text along with promising prose, some coherent sentences
- F: Incoherent text

Think aloud in a <scratchpad> for a few lines, then respond with the letter grade in <grade> ... </grade> tags."""
    rubric = vf.JudgeRubric(
        judge_client=judge_client,
        judge_model=judge_model,
        judge_prompt=judge_prompt,
    )

    grade_parser = vf.XMLParser(fields=["grade"], answer_field="grade")

    def grade_reward(prompt, completion, answer, state, **kwargs) -> float:
        judge_response = rubric.judge(prompt, completion, answer, state, **kwargs)
        judge_grade = (
            (grade_parser.parse_answer(judge_response) or "F")
            .strip()
            .replace("+", "")
            .replace("-", "")
            .upper()
        )
        return {
            "A": 1.0,
            "B": 0.75,
            "C": 0.5,
            "D": 0.25,
        }.get(judge_grade, 0.0)

    rubric.add_reward_func(grade_reward, weight=1.0)

    return vf.SingleTurnEnv(
        message_type="completion",
        dataset=dataset,
        parser=vf.Parser(),
        rubric=rubric,
        sampling_args={
            "stop": ["\n"],
        },
    )
