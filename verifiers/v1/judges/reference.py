"""Built-in reference-answer judge.

Empty model output scores zero. Judge API or parsing failures raise so the sample can
be retried instead of being scored against the model.
"""

from pathlib import Path
from typing import cast

from pydantic import model_validator

from verifiers.v1.configs.judge import JudgeConfig
from verifiers.v1.judge import (
    Judge,
    JudgeResponse,
    JudgeView,
    judge_question,
    judge_response,
    judge_verdict,
)
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace
from verifiers.v1.types import ID

REFERENCE_PROMPT = (Path(__file__).resolve().parent / "reference.txt").read_text(
    encoding="utf-8"
)


class ReferenceJudgeConfig(JudgeConfig):
    id: ID = "reference"
    """Pinned to the built-in, so a code-level default entry needs no explicit id (a TOML
    entry's `id` selects the config type before this default is ever seen)."""
    answer_field: str = "answer"
    """The task field holding the reference answer (works for extra/wire fields too). A
    list-valued field is rendered as one acceptable answer per line — the response must
    match any of them."""
    question_field: str = ""
    """Task field to fill the prompt's `{question}` (e.g. a dedicated `question` column
    without the prompt's instruction framing); empty = the task's prompt rendered as text
    (`TaskData.prompt_text`)."""
    view: JudgeView = "last_reply"
    """How much of the rollout fills `{response}` (see `JudgeView`). Defaults to the final
    reply — a reference-answer check grades the answer, not the path to it."""
    choices: tuple[str, str] = ("yes", "no")
    """The (positive, negative) verdict labels — e.g. `["A", "B"]` for graders that verdict
    with letters. Injected into the prompt as `{positive}`/`{negative}`, so the default
    template asks for whatever labels `parse` expects. The positive label scores 1.0; a
    verdict matching neither raises (a judge failure must error the rollout, not score the
    model 0)."""

    @model_validator(mode="after")
    def check_choices(self) -> "ReferenceJudgeConfig":
        # Duplicate labels would score every parsable verdict as the positive one; labels
        # are matched case-insensitively (`parse_judge_choice`), so distinctness is too.
        if not all(self.choices) or self.choices[0].upper() == self.choices[1].upper():
            raise ValueError(
                f"`choices` needs two distinct, non-empty verdict labels, got {self.choices!r}"
            )
        return self


class ReferenceJudge(Judge[float, ReferenceJudgeConfig]):
    prompt = REFERENCE_PROMPT

    def parse(self, response: JudgeResponse[float]) -> float:
        return float(
            judge_verdict(response.text, self.config.choices) == self.config.choices[0]
        )

    async def score(self, task: TaskData, trace: Trace) -> float:
        answer = getattr(task, self.config.answer_field, None)
        if answer is None:
            raise ValueError(
                f"reference judge found no {self.config.answer_field!r} field on the task; "
                "point `answer_field` at the task's reference-answer field"
            )
        if isinstance(answer, (list, tuple)):
            answer = "\n".join(str(item) for item in answer)
        response = judge_response(trace, self.config.view)
        if not response.strip():
            return 0.0  # nothing to grade — skip the (foregone) judge call
        positive, negative = self.config.choices
        result = await self.evaluate(
            trace=trace,
            question=judge_question(task, self.config.question_field),
            answer=answer,
            response=response,
            positive=positive,
            negative=negative,
        )
        return cast(float, result.parsed)


__all__ = ["ReferenceJudge", "ReferenceJudgeConfig"]
