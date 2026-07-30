"""Wiki-search reference judge: same scoring as `reference`, env-specific prompt."""

from pathlib import Path

import verifiers.v1 as vf

JUDGE_PROMPT = (Path(__file__).parent / "judge_prompt.txt").read_text()


class WikiSearchJudgeConfig(vf.ReferenceJudgeConfig):
    id: vf.ID = "wiki-search-v1"
    """Local package id — `load_judge` resolves this module's `WikiSearchJudge`."""
    question_field: str = "question"


class WikiSearchJudge(vf.ReferenceJudge):
    prompt = JUDGE_PROMPT


__all__ = ["WikiSearchJudge", "WikiSearchJudgeConfig"]
