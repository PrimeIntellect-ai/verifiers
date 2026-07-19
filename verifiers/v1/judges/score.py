"""Built-in single-verdict judge: one 0-10 grade of the whole attempt."""

import re

from verifiers.v1.judge import (
    Judge,
    JudgeConfig,
    JudgeView,
    judge_question,
    judge_response,
)
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace
from verifiers.v1.types import ID

SCORE_PROMPT = """You are grading another model's answer to a task.

## Task
{question}

## Answer
{response}

Judge whether the answer actually solves the task: correctness first, then
completeness. Think briefly, then give your verdict as the LAST line of your reply,
in exactly this form:

SCORE: <integer from 0 to 10>"""


class ScoreJudgeConfig(JudgeConfig):
    id: ID = "score"
    """Pinned to the built-in, so a code-level default entry needs no explicit id."""
    question_field: str = ""
    """Task field to fill the prompt's `{question}`; empty = the task's prompt rendered
    as text (`TaskData.prompt_text`)."""
    view: JudgeView = "last_reply"
    """How much of the rollout fills `{response}` (see `JudgeView`): the final reply
    (default), or the whole transcript for judging the process."""


class ScoreJudge(Judge[float, ScoreJudgeConfig]):
    prompt = SCORE_PROMPT

    def render(self, task: TaskData, trace: Trace) -> str:
        return str(
            self.build_messages(
                question=judge_question(task, self.config.question_field),
                response=judge_response(trace, self.config.view) or "<no answer>",
            )
        )

    def verdict(self, task: TaskData, trace: Trace, reply: str) -> float:
        """The last `SCORE: <n>` in the reply (models often restate the format
        first), normalized to [0, 1]. No verdict, or one outside 0-10 (`SCORE: 95`
        must not clamp to full marks), raises — a judge failure, never a score."""
        matches = re.findall(r"SCORE:\s*(\d+(?:\.\d+)?)", reply or "")
        if not matches:
            raise ValueError(f"judge returned no 'SCORE: <0-10>' verdict: {reply!r}")
        value = float(matches[-1])
        if not 0 <= value <= 10:
            raise ValueError(f"judge scored off the 0-10 scale: {matches[-1]!r}")
        return value / 10

    async def score(self, task: TaskData, trace: Trace) -> float:
        response = await self.complete(self.render(task, trace), trace=trace)
        return self.verdict(task, trace, response.text)


__all__ = ["ScoreJudge", "ScoreJudgeConfig"]
