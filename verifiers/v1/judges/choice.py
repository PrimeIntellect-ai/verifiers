"""choice — an off-the-shelf grade-scale judge, plugged from config alone.

Asks the judge model to grade the rollout's reply against the task's reference answer with
exactly one label from a configured scale, rewarding that label's score and recording every
label as a `<name>/<label>` metric (1 for the verdict, 0 for the rest) — the shape of
official graders like SimpleQA's A/B/C (correct / incorrect / not attempted):

    [[env.taskset.judges]]
    id = "choice"
    choices = { A = 1.0, B = 0.0, C = 0.0 }
    default = "C"    # unparseable verdicts / empty replies grade as C

Shares the binary judge's input selection (`answer_field`, `question_field`, `view`); an
empty response grades as `default` without a judge call. For a plain correct/incorrect check
prefer `binary`; for per-criterion checklists prefer `rubric`.
"""

from pydantic import model_validator

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

CHOICE_PROMPT = """Given a task, a reference answer, and a response, grade the response with \
exactly one of the following labels:

{choices}

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

Respond with the label only."""


class ChoiceJudgeConfig(JudgeConfig):
    id: ID = "choice"
    """Pinned to the built-in, so a code-level default entry needs no explicit id."""
    choices: dict[str, float]
    """The grade scale: each verdict label the prompt asks for, and the reward it maps to.
    Required — e.g. `{"A": 1.0, "B": 0.0, "C": 0.0}`."""
    default: str = ""
    """Label assumed when the verdict is unparseable or the response is empty (e.g. a
    "not attempted" bucket). Empty = score 0 with no label metric set."""
    answer_field: str = "answer"
    """The task field holding the reference answer (a list-valued field is rendered as one
    acceptable answer per line)."""
    question_field: str = ""
    """Task field to fill the prompt's `{question}`; empty = the task's prompt rendered as
    text (`Task.prompt_text`)."""
    view: JudgeView = "last_reply"
    """How much of the rollout fills `{response}` (see `JudgeView`)."""

    @model_validator(mode="after")
    def check_default(self) -> "ChoiceJudgeConfig":
        if self.default and self.default not in self.choices:
            raise ValueError(
                f"`default` {self.default!r} is not one of the choices "
                f"{sorted(self.choices)}"
            )
        return self


class ChoiceJudge(Judge[str | None, ChoiceJudgeConfig]):
    """Grades the reply with one label from a configured scale; rewards its score."""

    prompt = CHOICE_PROMPT

    def parse(self, response: JudgeResponse[str | None]) -> str | None:
        return parse_judge_choice(response.text, tuple(self.config.choices))

    async def score(self, task: Task, trace: Trace) -> float:
        answer = getattr(task, self.config.answer_field, None)
        if answer is None:
            raise ValueError(
                f"choice judge found no {self.config.answer_field!r} field on the task; "
                "point `answer_field` at the task's reference-answer field"
            )
        if isinstance(answer, (list, tuple)):
            answer = "\n".join(str(item) for item in answer)
        response = judge_response(trace, self.config.view)
        verdict = None
        if response.strip():  # an empty response grades as `default`, no judge call
            result = await self.evaluate(
                trace=trace,
                question=judge_question(task, self.config.question_field),
                answer=answer,
                response=response,
                choices="\n".join(f"- {label}" for label in self.config.choices),
            )
            verdict = result.parsed
        verdict = verdict or self.config.default
        for label in self.config.choices:
            trace.record_metric(f"{self.reward_name}/{label}", float(label == verdict))
        return self.config.choices.get(verdict, 0.0)


__all__ = ["ChoiceJudge", "ChoiceJudgeConfig"]
