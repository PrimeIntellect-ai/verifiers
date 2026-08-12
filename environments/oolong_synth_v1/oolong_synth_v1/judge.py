"""LLM judge for oolong-synth-v1 (used when a `judge` is configured).

A binary judge: asks whether the model response matches the reference answer. Defaults to Prime
inference (pinference).
"""

import re

import verifiers.v1 as vf

JUDGE_TEMPLATE = (
    "Question: {question}\n\nReference answer: {answer}\n\nModel response: {response}\n\n"
    "Does the model response match the reference answer? Answer 'yes' or 'no'."
)
VERDICT_RE = re.compile(r"\b(yes|no)\b")


class OolongJudgeConfig(vf.JudgeConfig):
    pass


class OolongJudge(vf.Judge[bool]):
    prompt = JUDGE_TEMPLATE

    def parse(self, response: vf.JudgeResponse[bool]) -> bool:
        match = VERDICT_RE.search(response.text.lower())
        return bool(match and match.group(1) == "yes")
