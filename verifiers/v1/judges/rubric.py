"""Built-in weighted rubric judge."""

import asyncio
import json
import math
import re
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
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace
from verifiers.v1.types import ID, StrictBaseModel

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
    "one-sentence reason grounded in the response, then set verdict to exactly one of the options "
    "listed in parentheses after that criterion."
)


def normalize_choice(choice: str, choices: list[str]) -> float:
    return choices.index(choice) / (len(choices) - 1)


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
    name: str
    """Key for the criterion's metric (`<judge name>/<name>`) and its `weights` override."""
    text: str
    weight: float = 1.0
    """The criterion's share of the reward (overridable per name via `weights` in config)."""
    choices: list[str] = ["no", "yes"]
    """Allowed answers, ordered **worst → best**: the first scores 0.0, the last 1.0, the rest
    evenly spaced by rank. Default `["no", "yes"]` is a binary check. Needs >= 2, no duplicates."""

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
    """A `.toml` or `.json` file containing a `criteria` list. Relative paths resolve
    against the evaluation's working directory."""
    weights: dict[str, float] = {}
    """Per-criterion weight overrides by criterion name (config wins over the file)."""
    question_field: str = ""
    """Task field to fill the prompt's `{question}`; empty = the task's prompt rendered as
    text (`TaskData.prompt_text`)."""
    answer_field: str = ""
    """Optional task field holding a reference answer (e.g. a gold patch) to show the judge, like
    the reference judge. Empty (default) shows none — `{reference}` renders blank. A list-valued
    field is joined one item per line. Needs `{reference}` in the prompt (the built-in `rubric.txt`
    has it)."""
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
    name: str
    reason: str
    verdict: str


class RubricVerdicts(StrictBaseModel):
    verdicts: list[CriterionVerdict]


class RubricJudge(Judge[RubricVerdicts, RubricJudgeConfig]):
    prompt = RUBRIC_PROMPT
    schema = RubricVerdicts

    @cached_property
    def criteria(self) -> list[Criterion]:
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
        self, task: TaskData, trace: Trace, batch: list[Criterion]
    ) -> dict[str, float]:
        def render(c: Criterion) -> str:
            return f"- {c.name}: {c.text} (answer one of, worst to best: {', '.join(c.choices)})"

        reference = ""
        if self.config.answer_field:
            answer = getattr(task, self.config.answer_field, None)
            if answer is None:
                raise ValueError(
                    f"rubric judge found no {self.config.answer_field!r} field on the task; "
                    "point `answer_field` at the task's reference-answer field or leave it empty"
                )
            if isinstance(answer, (list, tuple)):
                answer = "\n".join(str(item) for item in answer)
            # A clearly-announced section (leading blank separates it from the Task block; the
            # template's newline before Response closes it). Absent when `answer_field` is unset.
            # Fence longer than any backtick run in the answer, so a patch/markdown containing ```
            # can't close it early and spill into the prompt as instructions.
            fence = "`" * (
                max((len(m) for m in re.findall(r"`+", answer)), default=2) + 1
            )
            reference = (
                "\nReference solution (a correct implementation of this task, for comparison):\n"
                f"{fence}\n{answer}\n{fence}\n"
            )

        fields = dict(
            question=judge_question(task, self.config.question_field),
            response=judge_response(trace, self.config.view),
            criteria="\n".join(render(c) for c in batch),
            reference=reference,
        )
        if self.config.structured_output:
            result = await self.evaluate(trace=trace, **fields)
            verdicts = cast(RubricVerdicts, result.parsed).verdicts
        else:
            messages = cast(str, self.build_messages(**fields)) + JSON_SUFFIX
            result = await self.complete(messages, trace=trace)
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
            # An off-menu answer is a judge failure, not a zero score.
            if v.verdict not in choices:
                raise ValueError(
                    f"judge answered {v.verdict!r} for '{v.name}', expected one of {choices}"
                )
            scores[v.name] = normalize_choice(v.verdict, choices)
        return scores

    async def score(self, task: TaskData, trace: Trace) -> float:
        criteria = self.criteria
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
