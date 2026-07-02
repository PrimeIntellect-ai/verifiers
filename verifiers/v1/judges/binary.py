"""binary — an off-the-shelf reference-answer judge, plugged from config alone.

Asks the judge model whether the rollout's reply matches the task's reference answer and
scores 1/0. The reference answer is read off the task by field name (`answer_field`, default
`"answer"`; a list-valued field is judged as multiple acceptable answers), so it plugs into
any taskset that carries one — no taskset code:

    [[env.taskset.judges]]
    id = "binary"
    answer_field = "answer"

Grading inputs are config-selectable: `question_field` (what fills `{question}`), `view`
(final reply vs the whole transcript), `extract` (grade the last `\\boxed{...}` content),
and `choices` (the verdict labels, e.g. `["A", "B"]`).

Error attribution: a *model* failure scores 0 — an empty response short-circuits to 0.0
without a judge call, a wrong/non-committal reply gets the negative verdict. A *judge*
failure — an API error or an unparseable verdict — raises instead, erroring the rollout so
the sample is excluded/retried rather than scored against the model.
"""

from typing import Literal, cast

from verifiers.v1.judge import (
    Judge,
    JudgeConfig,
    JudgeResponse,
    JudgeView,
    judge_question,
    judge_response,
    judge_verdict,
)
from verifiers.v1.scoring import extract_boxed_answer
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace
from verifiers.v1.types import ID

BINARY_PROMPT = """Given a task, a reference answer, and a response, determine if the response \
is correct — it must match the reference answer (any one of them, when several are listed) in \
substance (exact wording may differ).

Task:
```
{question}
```

Reference answer:
```
{answer}
```

Response:
```
{response}
```

Respond either "{positive}" or "{negative}" only."""


class BinaryJudgeConfig(JudgeConfig):
    id: ID = "binary"
    """Pinned to the built-in, so a code-level default entry needs no explicit id (a TOML
    entry's `id` selects the config type before this default is ever seen)."""
    answer_field: str = "answer"
    """The task field holding the reference answer (works for extra/wire fields too). A
    list-valued field is rendered as one acceptable answer per line — the response must
    match any of them."""
    question_field: str = ""
    """Task field to fill the prompt's `{question}` (e.g. a dedicated `question` column
    without the prompt's instruction framing); empty = the task's prompt rendered as text
    (`Task.prompt_text`)."""
    view: JudgeView = "last_reply"
    """How much of the rollout fills `{response}` (see `JudgeView`). Defaults to the final
    reply — a reference-answer check grades the answer, not the path to it."""
    extract: Literal["none", "boxed"] = "none"
    """Pre-extraction applied to the response before judging: `boxed` grades the content of
    the last `\\boxed{...}` (falling back to the raw response when there is none)."""
    choices: tuple[str, str] = ("yes", "no")
    """The (positive, negative) verdict labels — e.g. `["A", "B"]` for graders that verdict
    with letters. Injected into the prompt as `{positive}`/`{negative}`, so the default
    template asks for whatever labels `parse` expects. The positive label scores 1.0; a
    verdict matching neither raises (a judge failure must error the rollout, not score the
    model 0)."""


class BinaryJudge(Judge[float, BinaryJudgeConfig]):
    """Scores the reply 1/0 against the task's reference answer."""

    prompt = BINARY_PROMPT

    def parse(self, response: JudgeResponse[float]) -> float:
        return float(
            judge_verdict(response.text, self.config.choices) == self.config.choices[0]
        )

    async def score(self, task: Task, trace: Trace) -> float:
        answer = getattr(task, self.config.answer_field, None)
        if answer is None:
            raise ValueError(
                f"binary judge found no {self.config.answer_field!r} field on the task; "
                "point `answer_field` at the task's reference-answer field"
            )
        if isinstance(answer, (list, tuple)):
            answer = "\n".join(str(item) for item in answer)
        response = judge_response(trace, self.config.view)
        if self.config.extract == "boxed":
            response = extract_boxed_answer(response, strict=True).strip() or response
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


__all__ = ["BinaryJudge", "BinaryJudgeConfig"]
