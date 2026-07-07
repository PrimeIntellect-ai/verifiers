"""rubric — an off-the-shelf rubric judge, plugged from config alone.

Reads a rubric — a list of criteria — from a JSON or TOML file and asks the judge model to grade
each by picking one of its ordered `choices` (default `["yes", "no"]`, a binary check), scored to
[0, 1] by rank — best → worst, so the first choice is 1.0 and the last 0.0 (all in one call by
default, or batched `max_criteria`-at-a-time). Grading uses plain-text JSON by default, or
structured output when `structured_output=true` (see `RubricJudgeConfig`). The reward is the
weighted mean of the per-criterion scores (`sum(w*v) / sum(w)`, so it stays in [0, 1]); each score
is also recorded as a `<name>/<criterion>` metric. Criterion weights come from the rubric file and
are overridable from config (`weights`); the aggregate weighs into `trace.reward` via the judge's
`weight`:

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
    # choices default to ["yes", "no"]; give an ordered best→worst list for a graded criterion:
    # choices = ["thorough", "partial", "none"]   # -> 1.0, 0.5, 0.0
"""

import asyncio
import json
import math
import tomllib
from functools import cached_property
from pathlib import Path
from typing import cast

from pydantic import field_validator

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

# Appended to the prompt when `structured_output` is off, so the reply is JSON we can parse. Each
# criterion carries a one-sentence `reason` written *before* the verdict (chain-of-thought), so the
# verdict follows from it and the reasoning is auditable in `trace.info["judge"]`.
JSON_SUFFIX = (
    "\n\nRespond with ONLY a JSON object and nothing else, in exactly this shape:\n"
    '{"verdicts": [{"name": "<criterion name>", "reason": "<one sentence citing specific '
    'evidence from the response>", "verdict": "<answer>"}, ...]}\n'
    "with one entry per criterion, using each criterion's exact name. For each, first write the "
    'one-sentence reason grounded in the response, then the verdict: "yes" or "no", or — for a '
    "criterion that lists allowed answers — exactly one of those answers."
)


def normalize_choice(choice: str, choices: list[str]) -> float:
    """Score a chosen option to [0, 1] by rank. `choices` are ordered best → worst, so the first
    scores 1.0, the last 0.0, and the rest evenly spaced (`choices` always has >= 2 entries)."""
    rank = choices.index(choice)
    return (len(choices) - 1 - rank) / (len(choices) - 1)


def first_verdicts_object(text: str) -> dict | None:
    """The first balanced JSON object in `text` that carries a `verdicts` key. Scanning for the
    key (not just the first `{`) skips prose and, crucially, an echoed format example — the one
    in `JSON_SUFFIX` fails to parse (its trailing `...` isn't JSON) and is passed over rather
    than mistaken for the answer. Returns `None` if no such object is found."""
    decoder = json.JSONDecoder()
    idx = text.find("{")
    while idx != -1:
        try:
            obj, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict) and "verdicts" in obj:
            return obj
        idx = text.find("{", idx + 1)
    return None


class Criterion(StrictBaseModel):
    """One rubric entry the judge grades by picking one of `choices`, scored to [0, 1] by rank.
    Defaults to a binary yes/no check (1/0)."""

    name: str
    """Key for the criterion's metric (`<judge name>/<name>`) and its `weights` override."""
    text: str
    """The check itself, phrased so the first (best) choice means satisfied."""
    weight: float = 1.0
    """The criterion's share of the reward (overridable per name via `weights` in config)."""
    choices: list[str] = ["yes", "no"]
    """Allowed answers, ordered **best → worst**: the first scores 1.0, the last 0.0, the rest
    evenly spaced by rank. Default `["yes", "no"]` is a binary check. Needs >= 2, no duplicates."""

    @field_validator("choices")
    @classmethod
    def _check_choices(cls, v: list[str]) -> list[str]:
        if len(v) < 2:
            raise ValueError(f"`choices` needs at least two options, got {v}")
        if len(set(v)) != len(v):
            raise ValueError(f"`choices` has duplicate options: {v}")
        return v


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
    """How many criteria to grade per judge call. `None` (default) grades all criteria in one
    call. `1` sends one call per criterion (n independent judges); `k` batches them k-at-a-time.
    Batches are graded concurrently and merged. Smaller batches trade more calls for focus/
    robustness. Must be >= 1."""
    structured_output: bool = False
    """Grade via JSON-schema structured output (`response_format`) when `True`; grade via plain-text
    output parsed as JSON when `False` (the default). Plain text is far more reliable on endpoints
    whose structured decoding is flaky (e.g. GLM-5.2 and other non-OpenAI models on some providers
    return empty completions for structured calls, especially over long transcripts); OpenAI models
    handle either. Transient HTTP failures are already retried by the OpenAI client."""


class CriterionVerdict(StrictBaseModel):
    """One criterion's reply, matched back to the rubric by `name`: a short `reason`
    (chain-of-thought, always recorded for auditability in `trace.info["judge"]`) written *before*
    the `verdict`, so the verdict follows from it. `verdict` is one of the criterion's `choices`
    (validated per criterion in `grade_batch`, since choices vary by criterion)."""

    name: str
    reason: str
    verdict: str


class RubricVerdicts(StrictBaseModel):
    """The judge's reply: one reasoned verdict per rubric criterion."""

    verdicts: list[CriterionVerdict]


class RubricJudge(Judge[RubricVerdicts, RubricJudgeConfig]):
    """Scores every rubric criterion 1/0 (in one judge call, or batched `max_criteria` at a
    time) and rewards their weighted mean."""

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

    async def grade_batch(
        self, task: Task, trace: Trace, batch: list[Criterion]
    ) -> dict[str, float]:
        """Grade one batch of criteria in a single judge call — structured output, or plain-text
        JSON when `structured_output` is off (which avoids flaky structured decoding). Returns
        `{name: score}`, each the chosen option normalized to [0, 1] by rank (best → worst),
        matched back to the batch by name. `score` fans this out per batch."""

        def render(
            c: Criterion,
        ) -> str:  # show non-default choices inline so the judge can pick
            line = f"- {c.name}: {c.text}"
            if c.choices != ["yes", "no"]:
                line += f" (answer one of, best to worst: {', '.join(c.choices)})"
            return line

        fields = dict(
            question=judge_question(task, self.config.question_field),
            response=judge_response(trace, self.config.view),
            criteria="\n".join(render(c) for c in batch),
        )
        if self.config.structured_output:
            result = await self.evaluate(trace=trace, **fields)
            verdicts = cast(RubricVerdicts, result.parsed).verdicts
        else:
            messages = cast(str, self.build_messages(**fields)) + JSON_SUFFIX
            result = await self.complete(
                messages, trace=trace
            )  # no schema -> plain text
            obj = first_verdicts_object(result.text)
            if obj is None:
                raise ValueError(
                    f"judge returned no verdicts JSON object: {result.text!r}"
                )
            verdicts = RubricVerdicts.model_validate(obj).verdicts
        # Exactly one verdict per criterion in the batch, matched by name — anything else is a
        # judge failure and must error the rollout, not score the model (see `judge_verdict`).
        by_criterion = {c.name: c for c in batch}
        if sorted(v.name for v in verdicts) != sorted(by_criterion):
            raise ValueError(
                f"judge verdicts name {sorted(v.name for v in verdicts)}, expected the "
                f"batch's {sorted(by_criterion)}"
            )
        scores: dict[str, float] = {}
        for v in verdicts:
            choices = by_criterion[v.name].choices
            if (
                v.verdict not in choices
            ):  # an off-menu answer is a judge failure, not a 0
                raise ValueError(
                    f"judge answered {v.verdict!r} for '{v.name}', expected one of {choices}"
                )
            scores[v.name] = normalize_choice(v.verdict, choices)
        return scores

    async def score(self, task: Task, trace: Trace) -> float:
        criteria = self.criteria
        # Split into batches of `max_criteria` (None = one batch of all), then grade the batches
        # concurrently (one judge call each) and merge.
        k = self.config.max_criteria
        if k is not None and k < 1:
            raise ValueError(f"`max_criteria` must be >= 1 or None, got {k}")
        batches = (
            [criteria]
            if k is None
            else [criteria[i : i + k] for i in range(0, len(criteria), k)]
        )
        # Fan out one call per batch; if any fails, cancel the siblings so a judge failure
        # doesn't keep billing the remaining batches.
        tasks = [
            asyncio.ensure_future(self.grade_batch(task, trace, batch))
            for batch in batches
        ]
        try:
            results = await asyncio.gather(*tasks)
        except BaseException:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
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
