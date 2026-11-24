"""Decision-model judge using Prime's evaluations API by default."""

import json
from typing import Any, cast

from pydantic import Field

from verifiers.v1.clients.base import build_async_openai
from verifiers.v1.configs.harness import ID
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
from verifiers.v1.types import Usage


class DecisionScoreError(ValueError):
    """The evaluations response has no valid numeric grade."""


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
    images_field: str = ""
    """Optional task field containing an ordered list of image data URLs."""
    view: JudgeView = "last_reply"


class DecisionJudge(Judge[float, DecisionJudgeConfig]):
    prompt = (
        "Grade how well the response satisfies the task in question. "
        "Use the reference answer when provided; equivalent answers are acceptable."
    )

    async def evaluate(
        self,
        *,
        images: list[str] | None = None,
        trace: Trace | None = None,
        trace_state: dict[str, Any] | None = None,
        **fields: Any,
    ) -> JudgeResponse[float]:
        """Score fields with optional image evidence. Supply trace_state to log compact artifact metadata."""
        request = {
            **self.config.sampling.model_dump(exclude_none=True),
            "model": self.config.model,
            "state": fields,
            "questions": {
                "grade": {
                    "type": "score",
                    "instructions": self.build_messages(**fields),
                    "criteria": self.config.levels,
                }
            },
        }
        if images:
            request["images"] = images
        async with build_async_openai(self.config) as client:
            raw = await client.post(
                self.config.base_url, cast_to=dict[str, Any], body=request
            )
        result = raw.get("result", raw)  # Workers AI wraps its response in `result`.
        # Some providers echo the input state and image data in the response.
        compact = bool(images) or trace_state is not None
        answers = result.get("answers") if isinstance(result, dict) else None
        response = JudgeResponse[float](text=json.dumps(answers if compact else raw))
        try:
            if isinstance(result, dict) and (usage := result.get("usage")):
                response.usage = Usage(
                    prompt_tokens=usage.get(
                        "inputTokens", usage.get("input_tokens", 0)
                    ),
                    completion_tokens=usage.get(
                        "outputTokens", usage.get("output_tokens", 0)
                    ),
                    cost=usage.get("cost"),
                )
            grade = answers.get("grade") if isinstance(answers, dict) else None
            score = grade.get("score") if isinstance(grade, dict) else None
            maximum = len(self.config.levels) - 1
            if type(score) not in (int, float) or not 0 <= score <= maximum:
                raise DecisionScoreError(f"invalid decision judge score {score!r}")
            response.parsed = score / maximum
            return response
        finally:
            # The call was billed even if the score could not be parsed.
            if trace is not None:
                logged = request.copy()
                if images:
                    del logged["images"]
                    logged["image_count"] = len(images)
                if trace_state is not None:
                    logged["state"] = trace_state
                trace.record_judge_call(
                    name=self.reward_name, request=logged, response=response
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
        if self.config.images_field:
            images = getattr(task, self.config.images_field, None)
            if not isinstance(images, list) or not all(
                isinstance(url, str) for url in images
            ):
                raise ValueError(
                    f"decision judge expected a list of image URLs in {self.config.images_field!r}"
                )
            fields["images"] = images
        result = await self.evaluate(trace=trace, **fields)
        return cast(float, result.parsed)


__all__ = ["DecisionJudge", "DecisionJudgeConfig", "DecisionScoreError"]
