"""rubric — an off-the-shelf rubric judge, plugged from config alone.

Reads a rubric — a list of yes/no criteria — from a JSON or TOML file and asks the judge model
to grade the criteria via structured output, scoring each 1/0 (all in one call by default, or
batched `max_criteria`-at-a-time — see `RubricJudgeConfig.max_criteria`). The reward is the weighted
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
import math
import tomllib
from functools import cached_property
from pathlib import Path
from typing import Literal, cast

from verifiers.v1.judge import (
    Judge,
    JudgeConfig,
    JudgeView,
    judge_question,
    judge_response,
)
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace
from verifiers.v1.types import ID, StrictBaseModel

# A sibling text file so it doubles as a starting point for a config `prompt_file`.
RUBRIC_PROMPT = (Path(__file__).resolve().parent / "rubric.txt").read_text(
    encoding="utf-8"
)


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
    for the shape. A relative path resolves against the eval's working directory."""
    weights: dict[str, float] = {}
    """Per-criterion weight overrides by criterion name (config wins over the file)."""
    question_field: str = ""
    """Task field to fill the prompt's `{question}`; empty = the task's prompt rendered as
    text (`Task.prompt_text`)."""
    view: JudgeView = "full_trace"
    """How much of the rollout fills `{response}` (see `JudgeView`). Defaults to the whole
    transcript — rubric criteria typically grade the process (tool use, citations,
    intermediate steps), not just the final answer."""
    max_criteria: int | None = None
    """How many criteria to grade per structured-output call. `None` (default) grades all
    criteria in one call. `1` sends one call per criterion (n independent judges); `k` batches
    them k-at-a-time. Batches are graded concurrently and merged. Smaller batches trade more
    calls for robustness where a model's structured-output reliability degrades with schema /
    prompt size. Must be >= 1."""
    max_retries: int = 0
    """Retries per batch when a call fails (empty completion, unparseable reply, or verdicts not
    matching the batch). 0 (default) keeps the strict contract — one failure errors the rollout.
    Raise it for flaky endpoints; combined with a small `max_criteria` each retried call is cheap.
    All attempts are recorded on the trace."""


class CriterionVerdict(StrictBaseModel):
    """One criterion's verdict in the judge's reply, matched back to the rubric by `name`."""

    name: str
    verdict: Literal["yes", "no"]


class RubricVerdicts(StrictBaseModel):
    """The judge's structured reply: one verdict per rubric criterion."""

    verdicts: list[CriterionVerdict]


class RubricJudge(Judge[RubricVerdicts, RubricJudgeConfig]):
    """Scores every rubric criterion 1/0 in one structured judge call; rewards their
    weighted mean."""

    prompt = RUBRIC_PROMPT
    schema = RubricVerdicts

    @cached_property
    def criteria(self) -> list[Criterion]:
        """The rubric, parsed once from `config.path` (TOML by suffix, else JSON) with the
        config's `weights` overrides applied. Fails loudly on a missing/empty file, duplicate
        criterion names, or an override naming no criterion."""
        path = self.config.path
        text = path.read_text(encoding="utf-8")
        data = (
            tomllib.loads(text) if path.suffix.lower() == ".toml" else json.loads(text)
        )
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
        if bad := [c.name for c in criteria if not 0 <= c.weight < math.inf]:
            # A negative weight would invert a criterion (pushing the reward out of [0, 1]);
            # NaN/inf (which json.loads accepts) would corrupt the weighted mean.
            raise ValueError(
                f"rubric '{path}' has negative or non-finite criterion weights: {bad}"
            )
        if sum(criterion.weight for criterion in criteria) <= 0:
            raise ValueError(f"rubric '{path}' has no positive criterion weight")
        return criteria

    def _batches(self, criteria: list[Criterion]) -> list[list[Criterion]]:
        """Split the criteria into per-call batches of at most `config.max_criteria`
        (None = one batch of all)."""
        k = self.config.max_criteria
        if k is None:
            return [criteria]
        if k < 1:
            raise ValueError(f"`max_criteria` must be >= 1 or None, got {k}")
        return [criteria[i : i + k] for i in range(0, len(criteria), k)]

    async def _grade_batch(
        self, task: Task, trace: Trace, batch: list[Criterion]
    ) -> dict[str, float]:
        """One structured-output call over `batch`; return `{name: 1.0/0.0}` matched by name."""
        result = await self.evaluate(
            trace=trace,
            question=judge_question(task, self.config.question_field),
            response=judge_response(trace, self.config.view),
            criteria="\n".join(f"- {c.name}: {c.text}" for c in batch),
        )
        verdicts = cast(RubricVerdicts, result.parsed).verdicts
        # Exactly one verdict per criterion in the batch, matched by name — anything else is a
        # judge failure and must error the rollout, not score the model (see `judge_verdict`).
        if sorted(v.name for v in verdicts) != sorted(c.name for c in batch):
            raise ValueError(
                f"judge verdicts name {sorted(v.name for v in verdicts)}, expected the "
                f"batch's {sorted(c.name for c in batch)}"
            )
        return {v.name: float(v.verdict == "yes") for v in verdicts}

    async def _score_batch(
        self, task: Task, trace: Trace, batch: list[Criterion]
    ) -> dict[str, float]:
        """Grade one batch, retrying up to `config.max_retries` times on a failed call before
        letting the last error propagate."""
        last: Exception | None = None
        for _ in range(self.config.max_retries + 1):
            try:
                return await self._grade_batch(task, trace, batch)
            except Exception as exc:  # retry transient structured-output failures
                last = exc
        raise cast(Exception, last)

    async def score(self, task: Task, trace: Trace) -> float:
        criteria = self.criteria
        # Grade the batches concurrently (one call each) and merge; default max_criteria=None is
        # a single call over all criteria.
        results = await asyncio.gather(
            *(
                self._score_batch(task, trace, batch)
                for batch in self._batches(criteria)
            )
        )
        by_name: dict[str, float] = {}
        for batch_verdicts in results:
            by_name.update(batch_verdicts)
        for criterion in criteria:
            trace.record_metric(
                f"{self.reward_name}/{criterion.name}", by_name[criterion.name]
            )
        total = sum(criterion.weight for criterion in criteria)
        return sum(c.weight * by_name[c.name] for c in criteria) / total


__all__ = [
    "Criterion",
    "CriterionVerdict",
    "RubricJudge",
    "RubricJudgeConfig",
    "RubricVerdicts",
]
