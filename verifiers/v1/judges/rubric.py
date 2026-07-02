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

from pydantic import model_validator

from verifiers.v1.judge import Judge, JudgeConfig, JudgeResponse
from verifiers.v1.scoring import parse_judge_choice
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace
from verifiers.v1.types import StrictBaseModel

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
    path: str = ""
    """The rubric file (`.toml` or `.json`) listing the criteria — see the module docstring
    for the shape. Required."""
    weights: dict[str, float] = {}
    """Per-criterion weight overrides by criterion name (config wins over the file)."""

    @model_validator(mode="after")
    def require_path(self) -> "RubricJudgeConfig":
        if not self.path:
            raise ValueError("rubric judge needs a `path` to a .toml/.json rubric file")
        return self


class RubricJudge(Judge[float, RubricJudgeConfig]):
    """Scores the final reply 1/0 per rubric criterion; rewards their weighted mean."""

    prompt = RUBRIC_PROMPT

    def parse(self, response: JudgeResponse[float]) -> float:
        return float(parse_judge_choice(response.text, ("yes", "no")) == "yes")

    @cached_property
    def criteria(self) -> list[Criterion]:
        """The rubric, parsed once from `config.path` (TOML by suffix, else JSON) with the
        config's `weights` overrides applied. Fails loudly on a missing/empty file, duplicate
        criterion names, or an override naming no criterion."""
        text = Path(self.config.path).read_text()
        data = (
            tomllib.loads(text)
            if self.config.path.endswith(".toml")
            else json.loads(text)
        )
        items = data.get("criteria", []) if isinstance(data, dict) else data
        criteria = [Criterion.model_validate(item) for item in items]
        if not criteria:
            raise ValueError(f"rubric file {self.config.path!r} lists no criteria")
        names = [criterion.name for criterion in criteria]
        if len(set(names)) != len(names):
            raise ValueError(
                f"rubric file {self.config.path!r} has duplicate criterion names"
            )
        if unknown := set(self.config.weights) - set(names):
            raise ValueError(
                f"`weights` overrides name no criterion in {self.config.path!r}: {sorted(unknown)}"
            )
        return [
            criterion.model_copy(
                update={
                    "weight": self.config.weights.get(criterion.name, criterion.weight)
                }
            )
            for criterion in criteria
        ]

    async def score(self, task: Task, trace: Trace) -> float:
        question = task.prompt if isinstance(task.prompt, str) else ""
        results = await asyncio.gather(
            *(
                self.evaluate(
                    trace=trace,
                    question=question,
                    response=trace.last_reply,
                    criterion=criterion.text,
                )
                for criterion in self.criteria
            )
        )
        verdicts = [cast(float, result.parsed) for result in results]
        for criterion, verdict in zip(self.criteria, verdicts):
            trace.record_metric(f"{self.config.reward_name}/{criterion.name}", verdict)
        total = sum(criterion.weight for criterion in self.criteria)
        if total <= 0:
            raise ValueError(
                f"rubric {self.config.path!r} has no positive criterion weight"
            )
        return sum(c.weight * v for c, v in zip(self.criteria, verdicts)) / total


__all__ = ["Criterion", "RubricJudge", "RubricJudgeConfig"]
