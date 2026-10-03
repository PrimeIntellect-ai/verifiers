"""Group grading: one agent grades a whole group of attempts at one task, of any
taskset.

A trainer mints a `GroupGradeData` row from a finished group (`from_traces`): the inner
task's wire data plus every attempt as an anonymized `Candidate`. `GroupGradeTask`
rebuilds the inner task from the inner taskset config (`--env.taskset.task.inner`) for
its prompt and runtime env. The grader's own box is generic and holds per candidate
`/grade/<label>/turns.json`, plus `patch.diff` and `tests.txt` when the attempt recorded
them. A candidate whose rollout checkpointed its box (`agent.checkpoint_on_finish`) can
be inspected in that box, restored on demand, through the `candidate_shell` tool
(`workspaces.py`). The grading policy and the workspace and verdict instructions are
`grade_prompt.md`. The grader writes `/grade/verdict.json`; `finalize` validates it into
`trace.info["group_verdict"]`.

The boxes are framework-only (`network_allow=[]`): the grader's harness reaches its
model and tools through the interception routes, and nothing else is reachable.
"""

import json
import random
import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, model_validator

import verifiers.v1 as vf
from verifiers.v1.envs.agentic_judge.env import JudgeTaskConfig, TextSource
from verifiers.v1.tasksets.group_grade.workspaces import (
    GradeState,
    Workspaces,
    WorkspacesConfig,
    build_inner,
    collect_privileged,
)
from verifiers.v1.types import AssistantMessage, ToolMessage, content_text
from verifiers.v1.utils.loaders import (
    narrow_plugin_field,
    taskset_config_type,
)

GRADE_DIR = "/grade"
VERDICT_FILE = f"{GRADE_DIR}/verdict.json"
PRIVILEGED_DIR = f"{GRADE_DIR}/privileged"
GRADE_PROMPT = (Path(__file__).parent / "grade_prompt.md").read_text()
RANKING_RULES = {
    "rewards": (Path(__file__).parent / "rank_passing.md").read_text(),
    "advantages": (Path(__file__).parent / "rank_all.md").read_text(),
}
RESULT_CHARS = 4000
"""Tool results are cut to their head and tail beyond this, everywhere."""
EDGE_TURNS = 10
"""A failing candidate's transcript keeps this many turns from each end."""
ELIDED = "[turn elided]"
MIN_QUOTE_CHARS = 20
MIN_QUOTE_WORDS = 4
MARKER = re.compile(r"^\[tool_(call|result)[^\]\n]*\]$", re.MULTILINE)
"""A turn's section headers; the same text written by the candidate is escaped."""

HackLevel = Literal["none", "suspected", "confirmed"]
Margin = Literal["slight", "clear", "large"]
Mode = Literal["rewards", "advantages"]


class Candidate(BaseModel):
    """One attempt in the group, as the grader sees it."""

    model_config = ConfigDict(frozen=True)

    label: str
    reward: float
    passed: bool
    """Whether the grader ranks this candidate's quality (GAR: reward above the group
    mean); a failing candidate's turns are `elide`d."""
    patch: str | None
    test_output: str | None
    turns: list[str]
    """`transcript_turns` of the attempt's trace; quotes are checked against these."""
    stop_condition: str | None
    num_output_tokens: int
    checkpoint: str | None = None
    """The attempt's box as its agent left it (`trace.info["checkpoint"]`), restored
    on demand by `candidate_shell`; None grades it from its files only."""


def _arguments(raw: str) -> str:
    try:
        args = json.loads(raw)
    except ValueError:
        return raw
    if not isinstance(args, dict):
        return raw
    return "\n".join(
        f"{key}: {value if isinstance(value, str) else json.dumps(value)}"
        for key, value in args.items()
    )


def _escape(text: str) -> str:
    """Neutralize section headers inside a turn's text, so the real ones are unambiguous."""
    return re.sub(r"^(?=\[tool_(call|result))", r"\\", text, flags=re.MULTILINE)


def _cap(text: str) -> str:
    if len(text) <= RESULT_CHARS:
        return text
    half = RESULT_CHARS // 2
    elided = len(text) - 2 * half
    return f"{text[:half]}\n[... {elided} chars elided ...]\n{text[-half:]}"


def transcript_turns(trace: vf.Trace) -> list[str]:
    """The trace's observable transcript, one entry per model response: its visible
    content, its tool calls (name + arguments), and the tool results answering them
    (head and tail beyond `RESULT_CHARS`). Hidden reasoning is excluded. Prompts,
    digests and citations number turns from 1, i.e. turn N is `turns[N - 1]`."""
    turns: list[list[str]] = []
    owner: dict[str, int] = {}
    for node in trace.nodes:
        message = node.message
        if node.sampled and isinstance(message, AssistantMessage):
            parts = [_escape(message.content)] if message.content else []
            for call in message.tool_calls or []:
                owner[call.id] = len(turns)
                name = f"{call.namespace}.{call.name}" if call.namespace else call.name
                parts.append(
                    f"[tool_call {name}]\n{_escape(_arguments(call.arguments))}"
                )
            turns.append(parts)
        elif isinstance(message, ToolMessage) and message.tool_call_id in owner:
            # A result replayed into a later prompt (e.g. after compaction) counts once.
            index = owner.pop(message.tool_call_id)
            result = _escape(_cap(content_text(message.content)))
            turns[index].append(f"[tool_result {message.name or ''}]\n{result}")
    return ["\n".join(parts) for parts in turns]


def policy_text(turn: str) -> str:
    """What the candidate itself wrote in a turn: its visible content and tool calls,
    without the tool results."""
    parts, start, written = [], 0, True
    for header in MARKER.finditer(turn):
        if written:
            parts.append(turn[start : header.start()])
        written = header.group(1) == "call"
        if written:
            parts.append(header.group(0))
        start = header.end()
    if written:
        parts.append(turn[start:])
    return "\n".join(parts)


def elide(turns: list[str]) -> list[str]:
    """The first and last `EDGE_TURNS` turns; the others are replaced by `ELIDED`, so
    turn numbers keep their meaning and nothing elided can be quoted."""
    if len(turns) <= 2 * EDGE_TURNS:
        return turns
    middle = len(turns) - 2 * EDGE_TURNS
    return [*turns[:EDGE_TURNS], *[ELIDED] * middle, *turns[-EDGE_TURNS:]]


class Evidence(BaseModel):
    turn: int
    quote: str


def _normalize(text: str) -> str:
    return " ".join(text.split())


def quote_found(turns: list[str], evidence: Evidence) -> bool:
    """Whether `evidence` cites an existing turn and its quote, at least
    `MIN_QUOTE_CHARS` characters or `MIN_QUOTE_WORDS` words, occurs (up to whitespace)
    in what the candidate wrote there — never in a tool result."""
    quote = _normalize(evidence.quote)
    return (
        1 <= evidence.turn <= len(turns)
        and (len(quote) >= MIN_QUOTE_CHARS or len(quote.split()) >= MIN_QUOTE_WORDS)
        and quote in _normalize(policy_text(turns[evidence.turn - 1]))
    )


class CandidateVerdict(BaseModel):
    label: str
    hack: HackLevel
    evidence: list[Evidence] = []
    """At least one when `hack` is `confirmed`."""


class GroupVerdict(BaseModel):
    candidates: list[CandidateVerdict]
    """Every label exactly once."""
    ranking: list[list[str]]
    """Tiers best to worst: a partition of the passing labels not confirmed as hacks.
    A tie expresses an inconclusive difference."""
    margins: list[Margin] = []
    """Optional: how much better each tier is than the next, one per gap (`slight`,
    `clear` or `large`); empty means every gap is `slight`. Used in the `advantages`
    mode."""
    notes: str = ""

    def check(self, data: "GroupGradeData") -> None:
        """Validate the verdict's structure against the graded group; raises ValueError.
        Citations are not checked here (see `quote_found`)."""
        passed = {c.label: c.passed for c in data.candidates}
        labels = [v.label for v in self.candidates]
        if sorted(labels) != sorted(passed):
            raise ValueError(
                f"candidates must list every label exactly once: got {labels}, "
                f"expected {sorted(passed)}"
            )
        for v in self.candidates:
            if v.hack == "confirmed" and not v.evidence:
                raise ValueError(f"{v.label}: a confirmed hack needs evidence")
        rankable = {
            v.label
            for v in self.candidates
            if passed[v.label] and v.hack != "confirmed"
        }
        ranked = [label for tier in self.ranking for label in tier]
        if any(not tier for tier in self.ranking) or sorted(ranked) != sorted(rankable):
            raise ValueError(
                f"ranking must partition the passing, non-confirmed labels "
                f"{sorted(rankable)} into non-empty tiers: got {self.ranking}"
            )
        if self.margins and len(self.margins) != len(self.ranking) - 1:
            raise ValueError(
                f"margins needs one entry per gap between tiers "
                f"({len(self.ranking) - 1}): got {self.margins}"
            )


class GroupGradeData(vf.TaskData):
    inner: dict
    """The inner task's wire data, validated by the inner taskset's data type."""
    candidates: list[Candidate]
    mode: Mode = "rewards"
    """`rewards`: the grader sees rewards and ranks the passing candidates.
    `advantages`: it sees no rewards and ranks every candidate (`from_traces` marks
    them all passing), optionally with margins."""
    workdir: str | None = GRADE_DIR
    network_allow: list[str] = Field(default_factory=list)
    """Framework-only: the grader reaches its model and nothing else."""

    @classmethod
    def from_traces(
        cls,
        traces: Sequence[vf.Trace],
        passed: Sequence[bool] | None,
        rng: random.Random,
    ) -> tuple["GroupGradeData", dict[str, str]]:
        """A row grading `traces` (attempts at one task), shuffled by `rng` and
        labeled `c01`, `c02`, ...; returns it with the label -> trace id map.
        `passed=None` is the `advantages` mode: every candidate is ranked, on quality
        alone. The prompt is left unset: the serving task composes it from its
        config."""
        mode: Mode = "rewards" if passed is not None else "advantages"
        passed = [True] * len(traces) if passed is None else passed
        if len(traces) != len(passed):
            raise ValueError(f"{len(traces)} traces but {len(passed)} pass flags")
        order = list(range(len(traces)))
        rng.shuffle(order)
        candidates, labels = [], {}
        for n, i in enumerate(order, 1):
            trace, label = traces[i], f"c{n:02d}"
            labels[label] = trace.id
            candidates.append(
                Candidate(
                    label=label,
                    reward=trace.reward,
                    passed=passed[i],
                    patch=trace.info.get("patch"),
                    test_output=trace.info.get("test_output"),
                    turns=transcript_turns(trace)
                    if passed[i]
                    else elide(transcript_turns(trace)),
                    stop_condition=trace.stop_condition,
                    num_output_tokens=trace.num_output_tokens,
                    checkpoint=trace.info.get("checkpoint"),
                )
            )
        inner = traces[0].task.data
        data = cls(
            inner=inner.model_dump(mode="json"),
            candidates=candidates,
            mode=mode,
            name=inner.name,
        )
        return data, labels


class GroupGradeTaskConfig(vf.TaskConfig):
    inner: SerializeAsAny[vf.TasksetConfig] = vf.TasksetConfig()
    """The graded taskset's config (`id` plus its own fields); rebuilds the inner task
    from `GroupGradeData.inner` for its prompt and runtime env."""
    workspaces: WorkspacesConfig | None = WorkspacesConfig()
    """The `candidate_shell` tool onto checkpointed candidates' boxes; None serves no
    tools (for a harness without MCP)."""
    prompt: TextSource | None = None
    """Replaces `grade_prompt.md` (grading policy, workspace and verdict instructions);
    inline text or `{ path = "..." }`. The task and the candidate list are always
    appended."""

    @model_validator(mode="before")
    @classmethod
    def _resolve_inner(cls, data):
        if isinstance(data, dict) and data.get("inner") is not None:
            narrow_plugin_field(data, "inner", taskset_config_type)
        return data


class GroupGradeTask(vf.Task[GroupGradeData, GradeState, GroupGradeTaskConfig]):
    NEEDS_CONTAINER = True

    def __init__(
        self, data: GroupGradeData, config: GroupGradeTaskConfig | None = None
    ) -> None:
        super().__init__(data, config)
        inner = self.config.inner
        if not inner.id:
            raise ValueError("group-grade needs the inner taskset: task.inner.id")
        self.inner = build_inner(inner, data.inner)
        if data.prompt is None:
            self.data = data.model_copy(update={"prompt": self.build_prompt()})

    @classmethod
    def toolsets(cls, config: GroupGradeTaskConfig) -> list[vf.Toolset]:
        if config.workspaces is None:
            return []
        inner = config.inner.model_dump(mode="json")
        return [Workspaces(config.workspaces.model_copy(update={"inner": inner}))]

    def build_prompt(self) -> str:
        policy = self.config.prompt
        body = GRADE_PROMPT if policy is None else JudgeTaskConfig._resolve(policy)
        task = f"## The task the candidates were given\n\n{self.inner.data.prompt_text}"
        if self.inner.data.system_prompt:
            task += (
                "\n\n## The rules the candidates were given\n\n"
                f"{self.inner.data.system_prompt}"
            )
        boxes = self.config.workspaces is not None
        rewards = self.data.mode == "rewards"
        listing = "\n".join(
            f"- {c.label}: "
            + (
                f"{'PASSED' if c.passed else 'FAILED'}, reward {c.reward:g}, "
                if rewards
                else ""
            )
            + f"{len(c.turns)} turns, {c.num_output_tokens} output tokens, "
            f"stopped by {c.stop_condition}"
            + (f", box: {'yes' if c.checkpoint else 'no'}" if boxes else "")
            for c in self.data.candidates
        )
        rules = RANKING_RULES[self.data.mode]
        return "\n\n".join([body, rules, task, f"## Candidates\n\n{listing}"])

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        # Ground truth the candidates never saw; only file names and sizes reach the
        # trace.
        privileged = await collect_privileged(self.inner, type(runtime.config)())
        written = {
            **{f"{PRIVILEGED_DIR}/{p}": c for p, c in privileged.files.items()},
            **{
                f"{PRIVILEGED_DIR}/staged/{p.lstrip('/')}": c
                for p, c in privileged.staged.items()
            },
        }
        if privileged.notes:
            written[f"{PRIVILEGED_DIR}/NOTES.md"] = privileged.notes.encode()
        for path, content in written.items():
            await runtime.write(path, content)
        trace.info["privileged"] = {path: len(c) for path, c in written.items()}
        for c in self.data.candidates:
            folder = f"{GRADE_DIR}/{c.label}"
            await runtime.write(f"{folder}/turns.json", json.dumps(c.turns).encode())
            if c.patch:
                await runtime.write(f"{folder}/patch.diff", c.patch.encode())
            if c.test_output:
                await runtime.write(f"{folder}/tests.txt", c.test_output.encode())
        await runtime.run(["rm", "-f", VERDICT_FILE], {})

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        try:
            raw = await runtime.read(VERDICT_FILE)
        except Exception as e:
            raise ValueError(f"the grader wrote no verdict to {VERDICT_FILE}") from e
        verdict = GroupVerdict.model_validate_json(raw)
        verdict.check(self.data)
        trace.info["group_verdict"] = verdict.model_dump(mode="json")

    @vf.reward
    async def valid(self, trace: vf.Trace) -> float:
        return float("group_verdict" in trace.info)

    @vf.metric
    async def restores(self, trace: vf.Trace) -> float:
        return float(len(trace.state.restores))

    @vf.metric
    async def restore_seconds(self, trace: vf.Trace) -> float:
        """Total seconds spent restoring candidate boxes."""
        return sum(trace.state.restores)


class GroupGradeConfig(vf.TasksetConfig):
    task: GroupGradeTaskConfig = GroupGradeTaskConfig()
    path: Path | None = None
    """Optional JSONL of `GroupGradeData` rows, for a manual run; tasks are otherwise
    minted by the trainer and sent to a served env."""


class GroupGradeTaskset(vf.Taskset[GroupGradeTask, GroupGradeConfig]):
    def load(self) -> Iterator[GroupGradeTask]:
        if self.config.path is None:
            return
        for line in self.config.path.read_text().splitlines():
            yield GroupGradeTask(
                GroupGradeData.model_validate_json(line), self.config.task
            )
