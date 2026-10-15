"""Decision-model judge using Prime's evaluations API by default."""

from typing import Any, cast

from pydantic import Field

from verifiers.v1.configs.judge import JudgeConfig
from verifiers.v1.judge import (
    Judge,
    JudgeResponse,
    JudgeView,
    judge_question,
    judge_response,
)
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace
from verifiers.v1.types import ID


class DecisionJudgeConfig(JudgeConfig):
    id: ID = "decision"
    model: str = "cloudflare/clef"
    base_url: str = "https://api.pinference.ai/api/v1/evaluations"
    """Full URL to POST evaluation requests to."""
    levels: list[str] = Field(
        default_factory=lambda: ["Incorrect", "Correct"], min_length=2, max_length=10
    )
    """Grading levels, worst to best. The model's probability-weighted level is
    normalized to [0, 1], so a two-level grade is the probability of correctness."""
    question_field: str = ""
    """Task field containing the question; empty uses the task's prompt text."""
    answer_field: str = ""
    """Optional task field containing a reference answer to include in the state."""
    view: JudgeView = "last_reply"


class DecisionJudge(Judge[float, DecisionJudgeConfig]):
    prompt = (
        "Grade how well the response satisfies the task in question. "
        "Use the reference answer when provided; equivalent answers are acceptable."
    )

    async def evaluate(
        self, *, trace: Trace | None = None, **fields: Any
    ) -> JudgeResponse[float]:
        def parse(answers: dict[str, Any]) -> float:
            score = answers["grade"]["score"] / (len(self.config.levels) - 1)
            if not 0 <= score <= 1:
                raise ValueError(f"decision judge score {score!r} is outside [0, 1]")
            return score

        return await self.decide(
            fields,
            {
                "grade": {
                    "type": "score",
                    "instructions": self.build_messages(**fields),
                    "criteria": self.config.levels,
                }
            },
            trace=trace,
            parse=parse,
            endpoint=self.config.base_url,
        )

    async def score(self, task: TaskData, trace: Trace) -> float:
        response = judge_response(trace, self.config.view)
        if not response.strip():
            return 0.0
        fields = {
            "question": judge_question(task, self.config.question_field),
            "response": response,
        }
        if self.config.answer_field:
            answer = getattr(task, self.config.answer_field, None)
            if answer is None:
                raise ValueError(
                    f"decision judge found no {self.config.answer_field!r} field on the task"
                )
            fields["answer"] = answer
        result = await self.evaluate(trace=trace, **fields)
        return cast(float, result.parsed)


__all__ = ["DecisionJudge", "DecisionJudgeConfig"]
