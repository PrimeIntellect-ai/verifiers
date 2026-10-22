"""Built-in weighted rubric judge."""

import asyncio
import json
import math
import re
import tomllib
from functools import cached_property
from pathlib import Path
from typing import Annotated, cast

from pydantic import BaseModel, Field, TypeAdapter, field_validator

from verifiers.v1.configs.judge import JudgeConfig
from verifiers.v1.judge import Judge, JudgeView, judge_question, judge_response
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace
from verifiers.v1.types import ID

CriterionWeight = Annotated[float, Field(ge=0, allow_inf_nan=False)]

RUBRIC_PROMPT = (Path(__file__).resolve().parent / "rubric.txt").read_text(
    encoding="utf-8"
)

# Appended to the prompt when `structured_output` is off, so the reply is JSON we can parse. Each
# criterion carries a one-sentence `reason` written *before* the verdict (chain-of-thought), so the
# verdict follows from it and the reasoning is auditable in `trace.info["judge_calls"]`.
JSON_SUFFIX = (
    "\n\nRespond with ONLY a JSON object and nothing else, in exactly this shape:\n"
    '{"verdicts": [{"name": "<criterion name>", "reason": "<one sentence citing specific '
    'evidence from the response>", "verdict": "<answer>"}, ...]}\n'
    "with one entry per criterion, using each criterion's exact name. For each, first write the "
    "one-sentence reason grounded in the response, then set verdict to exactly one of the options "
    "listed in parentheses after that criterion."
)


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


class Criterion(BaseModel):
    name: str
    """Key for the criterion's metric (`<judge name>/<name>`) and its `weights` override."""
    text: str
    weight: CriterionWeight = 1.0
    """The criterion's share of the reward (overridable per name via `weights` in config)."""
    choices: list[str] | dict[str, float] = Field(
        default_factory=lambda: ["no", "yes"], min_length=2
    )
    """Allowed answers. A list is ordered **worst → best**: the first scores 0.0, the last 1.0,
    the rest evenly spaced by rank. A mapping gives each answer its own finite score, in any
    order; `nan` scores nothing, which leaves the rollout unscored when the criterion
    carries weight. Default `["no", "yes"]` is a binary check. Needs >= 2, no duplicates."""

    @field_validator("choices")
    @classmethod
    def _check_choices(
        cls, v: list[str] | dict[str, float]
    ) -> list[str] | dict[str, float]:
        if isinstance(v, list) and len(set(v)) != len(v):
            raise ValueError(f"`choices` has duplicate options: {v}")
        if isinstance(v, dict) and any(math.isinf(score) for score in v.values()):
            raise ValueError(f"`choices` scores must be finite or nan: {v}")
        return v

    def score(self, choice: str) -> float:
        """The score of `choice`, one of `choices`; `nan` when it scores nothing."""
        if isinstance(self.choices, dict):
            return self.choices[choice]
        return self.choices.index(choice) / (len(self.choices) - 1)

    def render(self) -> str:
        """The criterion as a prompt line, with its allowed answers."""
        order = "" if isinstance(self.choices, dict) else ", worst to best"
        return f"- {self.name}: {self.text} (answer one of{order}: {', '.join(self.choices)})"


def weighted_score(scores: dict[str, float], criteria: list[Criterion]) -> float:
    """The criteria's weighted mean score; `nan` when a weighted criterion scored nothing.
    Zero-weight criteria never contribute, not even their `nan`."""
    weighted = [c for c in criteria if c.weight]
    total = sum(c.weight for c in weighted)
    return sum(c.weight * scores[c.name] for c in weighted) / total


CRITERIA_ADAPTER = TypeAdapter(list[Criterion])


def load_criteria(
    path: Path, weights: dict[str, CriterionWeight] | None = None
) -> list[Criterion]:
    """Load a JSON or TOML rubric and apply validated config overrides."""
    text = path.read_text(encoding="utf-8")
    data = tomllib.loads(text) if path.suffix.lower() == ".toml" else json.loads(text)
    items = data.get("criteria", []) if isinstance(data, dict) else data
    criteria = CRITERIA_ADAPTER.validate_python(items)
    if not criteria:
        raise ValueError(f"rubric file '{path}' lists no criteria")
    names = [criterion.name for criterion in criteria]
    if len(set(names)) != len(names):
        raise ValueError(f"rubric file '{path}' has duplicate criterion names")
    overrides = weights or {}
    if unknown := set(overrides) - set(names):
        raise ValueError(
            f"`weights` overrides name no criterion in '{path}': {sorted(unknown)}"
        )
    criteria = [
        criterion.model_copy(
            update={"weight": overrides.get(criterion.name, criterion.weight)}
        )
        for criterion in criteria
    ]
    total = sum(criterion.weight for criterion in criteria)
    if total == float("inf"):
        raise ValueError(f"rubric '{path}' has a non-finite total criterion weight")
    if total <= 0:
        raise ValueError(f"rubric '{path}' has no positive criterion weight")
    return criteria


class RubricJudgeConfig(JudgeConfig):
    id: ID = "rubric"
    """Pinned to the built-in, so a code-level default entry needs no explicit id."""
    path: Path
    """A `.toml` or `.json` file containing a `criteria` list. Relative paths resolve
    against the evaluation's working directory."""
    weights: dict[str, CriterionWeight] = Field(default_factory=dict)
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
    max_criteria: int | None = Field(default=None, ge=1)
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


class CriterionVerdict(BaseModel):
    name: str
    reason: str
    verdict: str


class RubricVerdicts(BaseModel):
    verdicts: list[CriterionVerdict]


def score_verdicts(
    verdicts: list[CriterionVerdict],
    criteria: list[Criterion],
    expected: str,
) -> dict[str, float]:
    """Validate a complete named verdict set and normalize its ordered choices."""
    by_criterion = {criterion.name: criterion for criterion in criteria}
    answers: dict[str, str] = {}
    for verdict in verdicts:
        if verdict.name in answers:
            raise ValueError(
                f"judge answered criterion {verdict.name!r} more than once"
            )
        answers[verdict.name] = verdict.verdict
    if sorted(answers) != sorted(by_criterion):
        raise ValueError(
            f"judge verdicts name {sorted(answers)}, expected {expected} "
            f"{sorted(by_criterion)}"
        )
    scores: dict[str, float] = {}
    for name, answer in answers.items():
        criterion = by_criterion[name]
        if answer not in criterion.choices:
            raise ValueError(
                f"judge answered {answer!r} for '{name}', expected one of "
                f"{list(criterion.choices)}"
            )
        scores[name] = criterion.score(answer)
    return scores


def criterion_metrics(
    prefix: str,
    verdicts: list[CriterionVerdict],
    criteria: list[Criterion],
    scores: dict[str, float],
) -> dict[str, float]:
    """Each criterion's score as `<prefix>/<name>`, unless it scored nothing, and for a
    criterion with scored choices (a mapping) an indicator per choice, `<prefix>/<name>/<choice>`,
    since several choices can share a score."""
    answers = {verdict.name: verdict.verdict for verdict in verdicts}
    metrics: dict[str, float] = {}
    for criterion in criteria:
        if not math.isnan(scores[criterion.name]):
            metrics[f"{prefix}/{criterion.name}"] = scores[criterion.name]
        if isinstance(criterion.choices, dict):
            for choice in criterion.choices:
                metrics[f"{prefix}/{criterion.name}/{choice}"] = float(
                    answers[criterion.name] == choice
                )
    return metrics


def drop_reason(
    prefix: str,
    verdicts: list[CriterionVerdict],
    criteria: list[Criterion],
    scores: dict[str, float],
) -> str | None:
    """`<prefix>: <name>=<answer>, ...` for each weighted criterion that scored nothing, the
    answers that leave the verdict unscored; None when the verdict scored."""
    answers = {verdict.name: verdict.verdict for verdict in verdicts}
    unscored = [
        f"{criterion.name}={answers[criterion.name]}"
        for criterion in criteria
        if criterion.weight and math.isnan(scores[criterion.name])
    ]
    return f"{prefix}: {', '.join(unscored)}" if unscored else None


class RubricJudge(Judge[RubricVerdicts, RubricJudgeConfig]):
    prompt = RUBRIC_PROMPT
    schema = RubricVerdicts

    @cached_property
    def criteria(self) -> list[Criterion]:
        return load_criteria(self.config.path, self.config.weights)

    async def grade_batch(
        self, task: TaskData, trace: Trace, batch: list[Criterion]
    ) -> dict[str, float]:
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

        fields = {
            "question": judge_question(task, self.config.question_field),
            "response": judge_response(trace, self.config.view),
            "criteria": "\n".join(c.render() for c in batch),
            "reference": reference,
        }
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
        # A malformed verdict is a judge failure and must error the rollout, not score the model.
        scores = score_verdicts(verdicts, batch, "the batch's")
        trace.record_metrics(
            criterion_metrics(self.reward_name, verdicts, batch, scores)
        )
        return scores

    async def score(self, task: TaskData, trace: Trace) -> float:
        criteria = self.criteria
        k = self.config.max_criteria
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
        return weighted_score(by_name, criteria)


__all__ = [
    "Criterion",
    "CriterionVerdict",
    "RubricJudge",
    "RubricJudgeConfig",
    "RubricVerdicts",
    "criterion_metrics",
    "weighted_score",
]
