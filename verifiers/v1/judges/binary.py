"""binary — an off-the-shelf reference-answer judge, plugged from config alone.

Asks the judge model whether the rollout's final reply matches the task's reference answer and
scores 1/0. The reference answer is read off the task by field name (`answer_field`, default
`"answer"`), so it plugs into any taskset that carries one — no taskset code:

    [[env.taskset.judges]]
    id = "binary"
    answer_field = "answer"
"""

from typing import cast

from verifiers.v1.judge import (
    Judge,
    JudgeConfig,
    JudgeResponse,
    JudgeView,
    judge_question,
    judge_response,
)
from verifiers.v1.scoring import parse_judge_choice
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace
from verifiers.v1.types import ID

BINARY_PROMPT = """Given a task, a reference answer, and a response, determine if the response \
is correct — it must match the reference answer in substance (exact wording may differ).

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

Respond either "yes" or "no" only."""


class BinaryJudgeConfig(JudgeConfig):
    id: ID = "binary"
    """Pinned to the built-in, so a code-level default entry needs no explicit id (a TOML
    entry's `id` selects the config type before this default is ever seen)."""
    answer_field: str = "answer"
    """The task field holding the reference answer (works for extra/wire fields too)."""
    question_field: str = ""
    """Task field to fill the prompt's `{question}` (e.g. a dedicated `question` column
    without the prompt's instruction framing); empty = the task's prompt rendered as text
    (`Task.prompt_text`)."""
    view: JudgeView = "last_reply"
    """How much of the rollout fills `{response}` (see `JudgeView`). Defaults to the final
    reply — a reference-answer check grades the answer, not the path to it."""


class BinaryJudge(Judge[float, BinaryJudgeConfig]):
    """Scores the final reply 1/0 against the task's reference answer."""

    prompt = BINARY_PROMPT

    def parse(self, response: JudgeResponse[float]) -> float:
        return float(parse_judge_choice(response.text, ("yes", "no")) == "yes")

    async def score(self, task: Task, trace: Trace) -> float:
        answer = getattr(task, self.config.answer_field, None)
        if answer is None:
            raise ValueError(
                f"binary judge found no {self.config.answer_field!r} field on the task; "
                "point `answer_field` at the task's reference-answer field"
            )
        result = await self.evaluate(
            trace=trace,
            question=judge_question(task, self.config.question_field),
            answer=answer,
            response=judge_response(trace, self.config.view),
        )
        return cast(float, result.parsed)


__all__ = ["BinaryJudge", "BinaryJudgeConfig"]
