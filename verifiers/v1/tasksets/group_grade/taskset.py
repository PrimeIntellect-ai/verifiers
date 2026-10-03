"""Group grading: one agent grades a whole group of attempts at one task.

A trainer mints a `GroupGradeData` row from a finished group (`from_traces`): the inner
task's wire data plus every attempt as an anonymized `Candidate`. `GroupGradeTask`
rebuilds the inner task from the inner taskset config (`--env.taskset.task.inner`) and
sets up one box: the inner task's `setup`, the candidates' workspace (`git_workspace`),
the inner task's `stage_verifier` (hidden tests), and per candidate
`/grade/<label>/{patch.diff,turns.json,tests.txt}`. The grading policy and the
workspace and verdict instructions are `grade_prompt.md`. The grader writes
`/grade/verdict.json`; `finalize` validates it into `trace.info["group_verdict"]`.

The box is framework-only (`network_allow=[]`): setup runs with egress, the grader's
harness reaches its model through the interception routes, and nothing else is reachable.
"""

import json
import random
import re
import shlex
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, model_validator

import verifiers.v1 as vf
from verifiers.v1.envs.agentic_judge.env import JudgeTaskConfig, TextSource
from verifiers.v1.types import AssistantMessage, ToolMessage, content_text
from verifiers.v1.utils.decorators import invoke
from verifiers.v1.utils.git import resolve_head, snapshot_untracked
from verifiers.v1.utils.loaders import (
    narrow_plugin_field,
    taskset_class,
    taskset_config_type,
)

GRADE_DIR = "/grade"
VERDICT_FILE = f"{GRADE_DIR}/verdict.json"
SWITCH = "/usr/local/bin/gar-switch"
GRADE_PROMPT = (Path(__file__).parent / "grade_prompt.md").read_text()
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


class GroupGradeData(vf.TaskData):
    inner: dict
    """The inner task's wire data, validated by the inner taskset's data type."""
    candidates: list[Candidate]
    network_allow: list[str] = Field(default_factory=list)
    """Framework-only: the grader reaches its model and nothing else."""

    @classmethod
    def from_traces(
        cls, traces: Sequence[vf.Trace], passed: Sequence[bool], rng: random.Random
    ) -> tuple["GroupGradeData", dict[str, str]]:
        """A row grading `traces` (attempts at one task), shuffled by `rng` and
        labeled `c01`, `c02`, ...; returns it with the label -> trace id map. The
        prompt is left unset: the serving task composes it from its config."""
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
                )
            )
        inner = traces[0].task.data
        data = cls(
            inner=inner.model_dump(mode="json"),
            candidates=candidates,
            name=inner.name,
            image=inner.image,
            workdir=inner.workdir,
            resources=inner.resources,
        )
        return data, labels


class GroupGradeTaskConfig(vf.TaskConfig):
    inner: SerializeAsAny[vf.TasksetConfig] = vf.TasksetConfig()
    """The graded taskset's config (`id` plus its own fields); rebuilds the inner task
    from `GroupGradeData.inner` for `setup` and `stage_verifier`."""
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


class GroupGradeTask(vf.Task[GroupGradeData, vf.State, GroupGradeTaskConfig]):
    NEEDS_CONTAINER = True

    def __init__(
        self, data: GroupGradeData, config: GroupGradeTaskConfig | None = None
    ) -> None:
        super().__init__(data, config)
        inner = self.config.inner
        if not inner.id:
            raise ValueError("group-grade needs the inner taskset: task.inner.id")
        task_cls = taskset_class(inner.id).task_type()
        self.inner = task_cls(
            task_cls.data_type().model_validate(data.inner), inner.task
        )
        self.compose_prompt = data.prompt is None
        if self.compose_prompt:
            self.data = data.model_copy(update={"prompt": self.build_prompt({})})

    def build_prompt(self, patch_notes: dict[str, str]) -> str:
        policy = self.config.prompt
        body = GRADE_PROMPT if policy is None else JudgeTaskConfig._resolve(policy)
        task = f"## The task the candidates were given\n\n{self.inner.data.prompt_text}"
        if self.inner.data.system_prompt:
            task += (
                "\n\n## The rules the candidates were given\n\n"
                f"{self.inner.data.system_prompt}"
            )
        listing = "\n".join(
            f"- {c.label}: {'PASSED' if c.passed else 'FAILED'}, reward {c.reward:g}, "
            f"{len(c.turns)} turns, {c.num_output_tokens} output tokens, "
            f"stopped by {c.stop_condition}"
            + (f", {patch_notes[c.label]}" if c.label in patch_notes else "")
            for c in self.data.candidates
        )
        return "\n\n".join([body, task, f"## Candidates\n\n{listing}"])

    def runtime_env(self) -> dict[str, str]:
        return self.inner.runtime_env()

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        await invoke(self.inner.setup, {"trace": trace, "runtime": runtime})
        for c in self.data.candidates:
            folder = f"{GRADE_DIR}/{c.label}"
            await runtime.write(f"{folder}/patch.diff", (c.patch or "").encode())
            await runtime.write(f"{folder}/turns.json", json.dumps(c.turns).encode())
            await runtime.write(f"{folder}/tests.txt", (c.test_output or "").encode())
        patch_notes = await self.git_workspace(runtime)
        await invoke(self.inner.stage_verifier, {"trace": trace, "runtime": runtime})
        await self.git_switch(runtime)
        await self._sh(runtime, f"rm -f {VERDICT_FILE}")
        if self.compose_prompt:
            # Whether each patch applies is known only now; the harness reads the
            # prompt after setup.
            data = trace.task.data.model_copy(
                update={"prompt": self.build_prompt(patch_notes)}
            )
            trace.task = trace.task.model_copy(update={"data": data})

    async def git_workspace(self, runtime: vf.Runtime) -> dict[str, str]:
        """Commit each candidate's patch on branch `gar/<label>` off the base commit and
        leave the base checked out; returns each candidate's patch note."""
        base = await resolve_head(runtime)
        if not base:
            raise RuntimeError("group-grade needs a git repository at the task workdir")
        # Files the image ships untracked; patches captured without `ignore` carry them.
        excludes = [f"--exclude={path}" for path in await snapshot_untracked(runtime)]
        notes = {}
        for c in self.data.candidates:
            branch = shlex.quote(f"gar/{c.label}")
            checkout = f"git checkout -q -f -B {branch} {base}"
            if not (c.patch or "").strip():
                notes[c.label] = "no patch"
                await self._sh(runtime, checkout)
                continue
            apply = shlex.join(
                [
                    "git",
                    "apply",
                    "--index",
                    "--binary",
                    *excludes,
                    f"{GRADE_DIR}/{c.label}/patch.diff",
                ]
            )
            commit = (
                "git -c user.name=gar -c user.email=gar@localhost commit -q "
                f"--no-verify --allow-empty -m {shlex.quote(c.label)}"
            )
            result = await runtime.run(
                ["sh", "-c", f"{checkout} && {apply} && {commit}"], {}
            )
            if result.exit_code:
                notes[c.label] = "patch does NOT apply (branch at base commit)"
                await self._sh(runtime, f"git reset -q --hard && {checkout}")
            else:
                notes[c.label] = "patch applied"
        await self._sh(runtime, f"git checkout -q -f --detach {base}")
        return notes

    async def git_switch(self, runtime: vf.Runtime) -> None:
        """Install `gar-switch`, keeping what is untracked now (the staged tests, image
        files) safe from its `git clean`."""
        repo = (await runtime.run(["git", "rev-parse", "--show-toplevel"], {})).stdout
        await self._sh(
            runtime,
            "git ls-files --others --exclude-standard --directory | sed 's|^|/|' "
            '>> "$(git rev-parse --git-path info/exclude)"',
        )
        switch = (
            '#!/bin/sh\nset -e\n[ -n "$1" ] || { echo "usage: gar-switch <label>" >&2; exit 2; }\n'
            f'cd {shlex.quote(repo.strip())}\ngit checkout -q -f "gar/$1"\ngit clean -fdq\n'
            'echo "on gar/$1"\n'
        )
        await runtime.write(SWITCH, switch.encode())
        await self._sh(runtime, f"chmod +x {SWITCH}")

    @staticmethod
    async def _sh(runtime: vf.Runtime, command: str) -> None:
        result = await runtime.run(["sh", "-c", command], {})
        if result.exit_code:
            raise RuntimeError(
                f"group-grade setup failed: {command!r}: {(result.stderr or '').strip()[-300:]}"
            )

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
