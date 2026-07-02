"""rubric — an off-the-shelf rubric judge, plugged from config alone.

Reads a rubric — a list of yes/no criteria — from a JSON or TOML file and asks the judge model
each criterion against the rollout's final reply, scoring each 1/0. The reward is the weighted
mean of the verdicts (`sum(w*v) / sum(w)`, so it stays in [0, 1]); each verdict is also recorded
as a `<name>/<criterion>` metric. Criterion weights come from the rubric file and are overridable
from config (`weights`); the aggregate weighs into `trace.reward` via the judge's `weight`:

    [[env.taskset.judges]]
    id = "rubric"
    path = "rubrics/quality.toml"
    weight = 0.5
    weights = { cites_sources = 2.0 }   # override the file's per-criterion weight

with `rubrics/quality.toml` listing the criteria (JSON takes `{"criteria": [...]}` or a bare list):

    [[criteria]]
    name = "cites_sources"
    text = "The response cites at least one specific source."
    weight = 1.0
"""

import asyncio
import json
import tomllib
from functools import cached_property
from pathlib import Path
from typing import cast

from verifiers.v1.judge import (
    Judge,
    JudgeConfig,
    JudgeResponse,
    JudgeView,
    judge_question,
    judge_response,
    judge_verdict,
)
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace
from verifiers.v1.types import ID, StrictBaseModel

RUBRIC_PROMPT = """Given a task, a response, and one grading criterion, determine if the \
response satisfies the criterion.

Task:
```
{question}
```

Response:
```
{response}
```

Criterion:
```
{criterion}
```

Respond either "yes" or "no" only."""


class Criterion(StrictBaseModel):
    """One rubric entry: a yes/no check the judge scores 1/0."""

    name: str
    """Key for the criterion's metric (`<judge name>/<name>`) and its `weights` override."""
    text: str
    """The check itself, phrased so "yes" means satisfied."""
    weight: float = 1.0
    """The criterion's share of the reward (overridable per name via `weights` in config)."""


class RubricJudgeConfig(JudgeConfig):
    id: ID = "rubric"
    """Pinned to the built-in, so a code-level default entry needs no explicit id."""
    path: Path
    """The rubric file (`.toml` or `.json`) listing the criteria — see the module docstring
    for the shape."""
    weights: dict[str, float] = {}
    """Per-criterion weight overrides by criterion name (config wins over the file)."""
    question_field: str = ""
    """Task field to fill the prompt's `{question}`; empty = the task's prompt rendered as
    text (`Task.prompt_text`)."""
    view: JudgeView = "full_trace"
    """How much of the rollout fills `{response}` (see `JudgeView`). Defaults to the whole
    transcript — rubric criteria typically grade the process (tool use, citations,
    intermediate steps), not just the final answer."""


class RubricJudge(Judge[float, RubricJudgeConfig]):
    """Scores the final reply 1/0 per rubric criterion; rewards their weighted mean."""

    prompt = RUBRIC_PROMPT

    def parse(self, response: JudgeResponse[float]) -> float:
        # An unparseable verdict raises (judge failure -> rollout error, not a silent 0).
        return float(judge_verdict(response.text, ("yes", "no")) == "yes")

    @cached_property
    def criteria(self) -> list[Criterion]:
        """The rubric, parsed once from `config.path` (TOML by suffix, else JSON) with the
        config's `weights` overrides applied. Fails loudly on a missing/empty file, duplicate
        criterion names, or an override naming no criterion."""
        path = self.config.path
        text = path.read_text()
        data = tomllib.loads(text) if path.suffix == ".toml" else json.loads(text)
        items = data.get("criteria", []) if isinstance(data, dict) else data
        criteria = [Criterion.model_validate(item) for item in items]
        if not criteria:
            raise ValueError(f"rubric file '{path}' lists no criteria")
        names = [criterion.name for criterion in criteria]
        if len(set(names)) != len(names):
            raise ValueError(f"rubric file '{path}' has duplicate criterion names")
        if unknown := set(self.config.weights) - set(names):
            raise ValueError(
                f"`weights` overrides name no criterion in '{path}': {sorted(unknown)}"
            )
        criteria = [
            criterion.model_copy(
                update={
                    "weight": self.config.weights.get(criterion.name, criterion.weight)
                }
            )
            for criterion in criteria
        ]
        if negative := [c.name for c in criteria if c.weight < 0]:
            # A negative weight would invert a criterion and push the reward out of [0, 1].
            raise ValueError(
                f"rubric '{path}' has negative criterion weights: {negative}"
            )
        if sum(criterion.weight for criterion in criteria) <= 0:
            raise ValueError(f"rubric '{path}' has no positive criterion weight")
        return criteria

    async def score(self, task: Task, trace: Trace) -> float:
        criteria = self.criteria
        question = judge_question(task, self.config.question_field)
        response = judge_response(trace, self.config.view)
        results = await asyncio.gather(
            *(
                self.evaluate(
                    trace=trace,
                    question=question,
                    response=response,
                    criterion=criterion.text,
                )
                for criterion in criteria
            )
        )
        verdicts = [cast(float, result.parsed) for result in results]
        for criterion, verdict in zip(criteria, verdicts):
            trace.record_metric(f"{self.reward_name}/{criterion.name}", verdict)
        total = sum(criterion.weight for criterion in criteria)
        return sum(c.weight * v for c, v in zip(criteria, verdicts)) / total


__all__ = ["Criterion", "RubricJudge", "RubricJudgeConfig"]
