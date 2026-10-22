"""Task-declared rubric criteria for Harbor tasks.

A task may ship `tests/criteria.toml`, a list of graded checks in the rubric `Criterion`
shape plus what the grader needs to decide each:

    pass_threshold = 1.0              # optional: the folded score a pass needs
    junit = "/logs/verifier/junit.xml"  # optional: where test.sh leaves its JUnit report

    [[criteria]]
    id = "keeps_api"
    text = "The existing public API keeps working."
    kind = "test"
    tests = ["tests/test_api.py::test_roundtrip", "api_ok"]  # JUnit tests or reward.json keys
    critical = true

    [[criteria]]
    id = "explains_tradeoff"
    text = "The report explains why the cache is invalidated on write."
    kind = "judge"
    evidence = ["/app/REPORT.md"]
    weight = 2.0
    conditional = true                # the judge may answer "not applicable"

After `test.sh` runs, the grader decides each criterion in the grading box: a `test`
criterion is met when every named test passed (a JUnit testcase without failure, error
or skip, or a reward.json value of at least 1); a `judge` criterion is answered yes, no
or (only when `conditional`) not applicable by the task's criteria judge, over the
evidence files read from the box. The reward folds them: 0 when a critical criterion is
unmet, else the weighted mean of the others that apply (1.0 when none do), and the task
passes when that score reaches `pass_threshold`.

Each criterion lands on the trace as `criteria/<id>` (1.0 met, 0.0 unmet, absent when
not applicable; conditional ones also get `criteria/<id>/<answer>` indicators) and in
`trace.info["criteria"]`, a structured report. A judge that fails raises, so the
rollout errors instead of scoring 0.
"""

import asyncio
import base64
import io
import math
import re
import tomllib
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from verifiers.v1.errors import TaskError
from verifiers.v1.judges.decision import DecisionJudge, DecisionJudgeConfig
from verifiers.v1.judges.rubric import (
    Criterion,
    CriterionVerdict,
    CriterionWeight,
    RubricJudge,
    RubricJudgeConfig,
    criterion_metrics,
)
from verifiers.v1.runtimes import Runtime
from verifiers.v1.trace import Trace

CRITERIA_FILE = "tests/criteria.toml"
"""Where a Harbor task declares its criteria, relative to the task directory."""
METRIC_PREFIX = "criteria"
YES, NO, NOT_APPLICABLE = "yes", "no", "not applicable"
CONDITIONAL_CHOICES = {NO: 0.0, YES: 1.0, NOT_APPLICABLE: math.nan}
MAX_EVIDENCE_BYTES = 20 * 1024 * 1024
"""The largest evidence file read from the grading box."""
MAX_EVIDENCE_CHARS = 100_000
"""How much of one evidence file's rendered text a judge reads."""
IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

APPLIES_PROMPT = (
    "Decide whether this grading criterion applies to the submitted work, given the "
    "task and the evidence. It is not applicable only when the situation it describes "
    "does not arise for this submission.\n\nCriterion: {criterion}"
)
MET_PROMPT = (
    "Decide whether the evidence shows the submitted work satisfies this grading "
    "criterion for the task.\n\nCriterion: {criterion}"
)
RUBRIC_PROMPT = """Given a task, the files a submission left (its evidence), and a list of grading criteria, determine for each criterion whether the submission satisfies it. Each criterion names the evidence files it is judged on.

Task:
```
{question}
```

Evidence:
{response}

Criteria:
{criteria}

For each criterion, answer under its exact name with exactly one of the options given in parentheses after it."""


class TaskCriterion(BaseModel):
    """One graded check a Harbor task declares in `tests/criteria.toml`."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    """The criterion's name: its metric (`criteria/<id>`) and its report entry."""
    text: str = Field(min_length=1)
    weight: CriterionWeight = 1.0
    critical: bool = False
    """Unmet, it zeroes the reward; met, it adds nothing to the weighted mean."""
    kind: Literal["test", "judge"]
    tests: list[str] = Field(default_factory=list)
    """`test`: the JUnit tests (`name`, `file.py::name`, `module.Class.name`; a bare
    name covers its parametrizations) or reward.json keys that must all pass."""
    conditional: bool = False
    """`judge`: whether the judge may answer not applicable."""
    evidence: list[str] = Field(default_factory=list)
    """`judge`: the files in the grading box the judge reads."""

    @model_validator(mode="after")
    def _kind_fields(self) -> "TaskCriterion":
        if self.kind == "test":
            if not self.tests:
                raise ValueError(f"test criterion {self.id!r} names no `tests`")
            if self.evidence or self.conditional:
                raise ValueError(
                    f"test criterion {self.id!r} takes no `evidence` or `conditional`"
                )
        else:
            if not self.evidence:
                raise ValueError(f"judge criterion {self.id!r} names no `evidence`")
            if self.tests:
                raise ValueError(f"judge criterion {self.id!r} takes no `tests`")
        return self

    def rubric(self) -> Criterion:
        """The criterion as the rubric judge grades it: yes/no, plus not applicable when
        conditional (which scores nothing)."""
        text = self.text
        if self.evidence:
            text += f" [evidence: {', '.join(self.evidence)}]"
        return Criterion(
            name=self.id,
            text=text,
            weight=self.weight,
            choices=dict(CONDITIONAL_CHOICES) if self.conditional else [NO, YES],
        )


class TaskCriteria(BaseModel):
    """A Harbor task's `tests/criteria.toml`."""

    model_config = ConfigDict(extra="forbid")

    pass_threshold: float = Field(1.0, ge=0, le=1)
    junit: str = "/logs/verifier/junit.xml"
    criteria: list[TaskCriterion] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> "TaskCriteria":
        ids = [criterion.id for criterion in self.criteria]
        if duplicates := sorted({i for i in ids if ids.count(i) > 1}):
            raise ValueError(f"criteria share ids {duplicates}")
        return self

    @property
    def needs_junit(self) -> bool:
        return any(c.kind == "test" for c in self.criteria)


def load_task_criteria(task_dir: Path) -> TaskCriteria | None:
    """The task's criteria file, or None when it ships none."""
    path = task_dir / CRITERIA_FILE
    if not path.is_file():
        return None
    try:
        return TaskCriteria.model_validate(tomllib.loads(path.read_text()))
    except (tomllib.TOMLDecodeError, ValueError) as exc:
        raise ValueError(f"{task_dir.name}: invalid {CRITERIA_FILE}: {exc}") from exc


class TaskRubricJudgeConfig(RubricJudgeConfig):
    """A rubric judge (a chat model) for task-declared criteria: the criteria come from
    the task, so it takes no `path` or `criteria` of its own."""

    name: str = METRIC_PREFIX

    @model_validator(mode="after")
    def _one_criteria_source(self) -> "TaskRubricJudgeConfig":
        return self


class TaskDecisionJudgeConfig(DecisionJudgeConfig):
    """A decision judge (clef on Prime's evaluations API) for task-declared criteria:
    one yes/no decision per criterion, preceded by an applicability decision for a
    conditional one. A probability of at least `threshold` decides yes."""

    name: str = METRIC_PREFIX
    threshold: float = Field(0.5, gt=0, lt=1)


CriteriaJudgeConfig = TaskDecisionJudgeConfig | TaskRubricJudgeConfig


def criteria_judge_config(value: object) -> object:
    """Resolve a raw `criteria_judge` entry by its `id`: `decision` (the default) or
    `rubric`."""
    if not isinstance(value, dict):
        return value
    kinds = {"decision": TaskDecisionJudgeConfig, "rubric": TaskRubricJudgeConfig}
    judge_id = value.get("id", "decision")
    if judge_id not in kinds:
        raise ValueError(
            f"criteria_judge id must be one of {sorted(kinds)}, not {judge_id!r}"
        )
    return kinds[judge_id].model_validate(value)


class Evidence(BaseModel):
    path: str
    text: str
    images: list[str] = Field(default_factory=list)
    """`data:` URLs, for a judge that reads images."""


class CriterionResult(BaseModel):
    """One criterion's outcome, as `trace.info["criteria"]["criteria"]` records it."""

    id: str
    kind: Literal["test", "judge"]
    critical: bool
    weight: float
    conditional: bool
    answer: Literal["yes", "no", "not applicable"]
    reason: str = ""
    tests: dict[str, str] = Field(default_factory=dict)
    """`test`: each named test's status (`passed`, `failed`, `error`, `skipped`,
    `missing`)."""
    probability: float | None = None
    """`judge` with a decision judge: the probability it gave `answer`."""

    @property
    def met(self) -> bool | None:
        return None if self.answer == NOT_APPLICABLE else self.answer == YES


class CriteriaReport(BaseModel):
    """What `trace.info["criteria"]` holds."""

    score: float
    passed: bool
    pass_threshold: float
    critical_unmet: list[str]
    criteria: list[CriterionResult]


def fold(results: list[CriterionResult], pass_threshold: float) -> CriteriaReport:
    """0 when a critical criterion is unmet, else the weighted mean of the non-critical
    criteria that apply (1.0 when none do or they weigh nothing)."""
    critical_unmet = [r.id for r in results if r.critical and r.met is False]
    rest = [r for r in results if not r.critical and r.met is not None]
    total = sum(r.weight for r in rest)
    if critical_unmet:
        score = 0.0
    elif total:
        score = sum(r.weight for r in rest if r.met) / total
    else:
        score = 1.0
    return CriteriaReport(
        score=score,
        passed=score >= pass_threshold,
        pass_threshold=pass_threshold,
        critical_unmet=critical_unmet,
        criteria=results,
    )


async def grade_criteria(
    criteria: TaskCriteria,
    runtime: Runtime,
    trace: Trace,
    *,
    question: str,
    scores: dict[str, float],
    judge: CriteriaJudgeConfig,
) -> CriteriaReport:
    """Decide every criterion in the grading box `runtime`, after `test.sh` ran and left
    `scores` (its reward.json, possibly empty); record each on `trace` and return the
    folded report."""
    tests = [c for c in criteria.criteria if c.kind == "test"]
    judged = [c for c in criteria.criteria if c.kind == "judge"]
    cases: dict[str, str] | None = None
    if any(name not in scores for c in tests for name in c.tests):
        cases = await read_junit(runtime, criteria.junit)
    results = {c.id: decide_test(c, scores, cases or {}) for c in tests}
    if judged:
        evidence = await read_evidence(
            runtime, sorted({p for c in judged for p in c.evidence})
        )
        if isinstance(judge, TaskRubricJudgeConfig):
            decided = await judge_rubric(judge, judged, evidence, question, trace)
        else:
            decided = await judge_decisions(judge, judged, evidence, question, trace)
        results.update(decided)
    ordered = [results[c.id] for c in criteria.criteria]
    report = fold(ordered, criteria.pass_threshold)
    # The rubric judge records its own criteria's metrics as it grades.
    recorded = judged if isinstance(judge, TaskRubricJudgeConfig) else []
    rest = [c for c in criteria.criteria if c not in recorded]
    trace.record_metrics(
        criterion_metrics(
            METRIC_PREFIX,
            [
                CriterionVerdict(name=c.id, reason="", verdict=results[c.id].answer)
                for c in rest
            ],
            [c.rubric() for c in rest],
            {c.id: c.rubric().score(results[c.id].answer) for c in rest},
        )
    )
    trace.info["criteria"] = report.model_dump(mode="json")
    return report


def decide_test(
    criterion: TaskCriterion, scores: dict[str, float], cases: dict[str, str]
) -> CriterionResult:
    statuses: dict[str, str] = {}
    for name in criterion.tests:
        if name in scores:
            statuses[name] = "passed" if scores[name] >= 1 else "failed"
        else:
            matched = [status for case, status in cases.items() if matches(name, case)]
            statuses[name] = (
                "missing"
                if not matched
                else next((s for s in matched if s != "passed"), "passed")
            )
    met = all(status == "passed" for status in statuses.values())
    return CriterionResult(
        id=criterion.id,
        kind="test",
        critical=criterion.critical,
        weight=criterion.weight,
        conditional=False,
        answer=YES if met else NO,
        tests=statuses,
    )


def normalize_test(name: str) -> str:
    """A test selector as a dotted path: `tests/test_a.py::TestX::test_y` and
    `tests.test_a.TestX.test_y` both become `tests.test_a.TestX.test_y`."""
    name = name.replace("::", ".")
    head, bracket, params = name.partition("[")
    head = re.sub(r"\.py(?=\.|$)", "", head).replace("/", ".")
    return head + bracket + params


def matches(selector: str, case: str) -> bool:
    """Whether a criterion's test selector names the JUnit testcase `case`
    (`classname.name`): the whole path or a trailing part of it, with or without the
    parametrization."""
    selector = normalize_test(selector)
    for candidate in (case, case.partition("[")[0]):
        if candidate == selector or candidate.endswith("." + selector):
            return True
    return False


def parse_junit(data: bytes) -> dict[str, str]:
    """Each JUnit testcase's status by `classname.name`; a name repeated (a rerun) keeps
    its worst status."""
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise TaskError(f"the verifier's JUnit report is not valid XML: {exc}") from exc
    rank = {"passed": 0, "skipped": 1, "failed": 2, "error": 3}
    cases: dict[str, str] = {}
    for case in root.iter("testcase"):
        name = case.get("name", "")
        classname = case.get("classname", "")
        key = f"{classname}.{name}" if classname else name
        status = "passed"
        for tag in ("error", "failure", "skipped"):
            if case.find(tag) is not None:
                status = {"failure": "failed"}.get(tag, tag)
                break
        if rank[status] >= rank[cases.get(key, "passed")]:
            cases[key] = status
    return cases


async def exists(runtime: Runtime, path: str) -> bool:
    probe = await runtime.run(["sh", "-c", 'test -f "$1"', "vf-criteria", path], {})
    if probe.exit_code not in (0, 1):
        raise TaskError(f"cannot inspect {path} in the grading box")
    return probe.exit_code == 0


async def read_junit(runtime: Runtime, path: str) -> dict[str, str]:
    """The JUnit report test criteria are decided from. Its absence is the verifier's
    failure, not the submission's: the rollout errors."""
    if not await exists(runtime, path):
        raise TaskError(
            f"the task's test criteria need a JUnit report at {path}, which the "
            "verifier did not write (run pytest with --junitxml)"
        )
    return parse_junit(await runtime.read(path, max_bytes=MAX_EVIDENCE_BYTES))


async def read_evidence(runtime: Runtime, paths: list[str]) -> dict[str, Evidence]:
    """Each evidence file read from the grading box and rendered for a judge. A missing
    file is evidence too: the judge reads that it is missing."""

    async def one(path: str) -> Evidence:
        if not await exists(runtime, path):
            return Evidence(path=path, text="(this file does not exist)")
        return render(path, await runtime.read(path, max_bytes=MAX_EVIDENCE_BYTES))

    return dict(zip(paths, await asyncio.gather(*(one(p) for p in paths))))


def render(path: str, data: bytes) -> Evidence:
    """A file as a judge reads it: text as is; documents, spreadsheets, slides and PDFs
    as their text; an image as itself."""
    suffix = PurePosixPath(path).suffix.lower()
    if suffix in IMAGE_TYPES:
        url = f"data:{IMAGE_TYPES[suffix]};base64,{base64.b64encode(data).decode()}"
        return Evidence(path=path, text=f"(image, {len(data)} bytes)", images=[url])
    try:
        if suffix == ".docx":
            text = docx_text(data)
        elif suffix in (".xlsx", ".xlsm"):
            text = xlsx_text(data)
        elif suffix == ".pptx":
            text = pptx_text(data)
        elif suffix == ".pdf":
            text = pdf_text(data)
        elif b"\0" in data[:8192]:
            text = f"(binary file, {len(data)} bytes)"
        else:
            text = data.decode("utf-8", errors="replace")
    except (zipfile.BadZipFile, KeyError, ET.ParseError) as exc:
        text = f"(unreadable {suffix} file, {len(data)} bytes: {exc})"
    if len(text) > MAX_EVIDENCE_CHARS:
        text = (
            text[:MAX_EVIDENCE_CHARS]
            + f"\n... ({len(text) - MAX_EVIDENCE_CHARS} more characters not shown)"
        )
    return Evidence(path=path, text=text)


W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
S = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def docx_text(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    lines = []
    for paragraph in root.iter(f"{W}p"):
        lines.append("".join(t.text or "" for t in paragraph.iter(f"{W}t")))
    return "\n".join(lines)


def xlsx_text(data: bytes) -> str:
    """Each sheet as tab-separated rows of its cells' stored values (a formula's cached
    result, not the formula)."""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = set(archive.namelist())
        shared = []
        if "xl/sharedStrings.xml" in names:
            for item in ET.fromstring(archive.read("xl/sharedStrings.xml")).iter(
                f"{S}si"
            ):
                shared.append("".join(t.text or "" for t in item.iter(f"{S}t")))
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {rel.get("Id"): rel.get("Target", "") for rel in rels}
        sheets = []
        for sheet in workbook.iter(f"{S}sheet"):
            target = targets[sheet.get(f"{R}id")].lstrip("/")
            target = target if target.startswith("xl/") else f"xl/{target}"
            rows = []
            for row in ET.fromstring(archive.read(target)).iter(f"{S}row"):
                cells = []
                for cell in row.iter(f"{S}c"):
                    value = cell.find(f"{S}v")
                    if cell.get("t") == "s" and value is not None:
                        cells.append(shared[int(value.text or 0)])
                    elif cell.get("t") == "inlineStr":
                        cells.append("".join(t.text or "" for t in cell.iter(f"{S}t")))
                    else:
                        cells.append("" if value is None else value.text or "")
                rows.append("\t".join(cells))
            sheets.append(f"## sheet {sheet.get('name')}\n" + "\n".join(rows))
    return "\n\n".join(sheets)


def pptx_text(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        slides = sorted(
            (
                n
                for n in archive.namelist()
                if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)
            ),
            key=lambda n: int(re.sub(r"\D", "", n)),
        )
        out = []
        for number, name in enumerate(slides, 1):
            root = ET.fromstring(archive.read(name))
            texts = [
                "".join(t.text or "" for t in p.iter(f"{A}t"))
                for p in root.iter(f"{A}p")
            ]
            out.append(f"## slide {number}\n" + "\n".join(t for t in texts if t))
    return "\n\n".join(out)


def pdf_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise TaskError(
            "a judge criterion's evidence is a PDF, and reading it needs pypdf "
            "installed beside verifiers"
        ) from exc
    reader = PdfReader(io.BytesIO(data))
    return "\n\n".join(
        f"## page {n}\n{page.extract_text() or ''}"
        for n, page in enumerate(reader.pages, 1)
    )


def evidence_block(paths: list[str], evidence: dict[str, Evidence]) -> str:
    blocks = []
    for path in paths:
        text = evidence[path].text
        fence = "`" * (max((len(m) for m in re.findall(r"`+", text)), default=2) + 1)
        blocks.append(f"File {path}:\n{fence}\n{text}\n{fence}")
    return "\n\n".join(blocks)


async def judge_rubric(
    config: TaskRubricJudgeConfig,
    criteria: list[TaskCriterion],
    evidence: dict[str, Evidence],
    question: str,
    trace: Trace,
) -> dict[str, CriterionResult]:
    """Grade the judge criteria with a chat-model rubric judge, `max_criteria` at a time,
    each batch over its criteria's evidence. Text only: image evidence raises."""
    if images := [p for c in criteria for p in c.evidence if evidence[p].images]:
        raise TaskError(
            f"the rubric criteria judge reads text only, and evidence {sorted(set(images))} "
            "is an image; grade it with the decision judge"
        )
    rubric = [c.rubric() for c in criteria]
    judge = RubricJudge(config.model_copy(update={"criteria": rubric}))
    if config.prompt is None:
        judge.prompt = RUBRIC_PROMPT
    k = config.max_criteria or len(criteria)
    batches = [criteria[i : i + k] for i in range(0, len(criteria), k)]

    async def grade(batch: list[TaskCriterion]) -> dict[str, float]:
        paths = list(dict.fromkeys(p for c in batch for p in c.evidence))
        return await judge.grade(
            question=question,
            response=evidence_block(paths, evidence),
            criteria=[c.rubric() for c in batch],
            trace=trace,
        )

    scores: dict[str, float] = {}
    for result in await asyncio.gather(*(grade(b) for b in batches)):
        scores.update(result)
    results = {}
    for criterion in criteria:
        score = scores[criterion.id]
        answer = NOT_APPLICABLE if math.isnan(score) else YES if score else NO
        results[criterion.id] = judged_result(criterion, answer)
    return results


async def judge_decisions(
    config: TaskDecisionJudgeConfig,
    criteria: list[TaskCriterion],
    evidence: dict[str, Evidence],
    question: str,
    trace: Trace,
) -> dict[str, CriterionResult]:
    """Decide each judge criterion with the decision judge, which sees image evidence."""

    async def decide(prompt: str, levels: list[str], criterion: TaskCriterion) -> float:
        judge = DecisionJudge(config.model_copy(update={"levels": levels}))
        if config.prompt is None:
            judge.prompt = prompt
        images = [url for p in criterion.evidence for url in evidence[p].images]
        response = await judge.evaluate(
            trace=trace,
            images=images or None,
            question=question,
            criterion=criterion.text,
            evidence=evidence_block(criterion.evidence, evidence),
        )
        assert response.parsed is not None
        return response.parsed

    async def one(criterion: TaskCriterion) -> CriterionResult:
        if criterion.conditional:
            applies = await decide(
                APPLIES_PROMPT, ["Not applicable", "Applicable"], criterion
            )
            if applies < config.threshold:
                return judged_result(criterion, NOT_APPLICABLE, 1 - applies)
        met = await decide(MET_PROMPT, ["No", "Yes"], criterion)
        return judged_result(criterion, YES if met >= config.threshold else NO, met)

    decided = await asyncio.gather(*(one(c) for c in criteria))
    return {result.id: result for result in decided}


def judged_result(
    criterion: TaskCriterion, answer: str, probability: float | None = None
) -> CriterionResult:
    return CriterionResult(
        id=criterion.id,
        kind="judge",
        critical=criterion.critical,
        weight=criterion.weight,
        conditional=criterion.conditional,
        answer=answer,
        probability=probability,
    )


__all__ = [
    "CRITERIA_FILE",
    "CriteriaJudgeConfig",
    "CriteriaReport",
    "CriterionResult",
    "TaskCriteria",
    "TaskCriterion",
    "TaskDecisionJudgeConfig",
    "TaskRubricJudgeConfig",
    "grade_criteria",
    "load_task_criteria",
]
