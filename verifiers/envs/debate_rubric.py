"""Debate rubric: ±1 zero-sum reward from judge outcome + diagnostic metrics.

Reward (single source of training signal):
    winner member → +1, loser member → −1, judge → 0, no decision → 0.

Diagnostics (weight-0 telemetry; never feed reward):
    per-member: turns, parse_error_count, num_commits, num_unique_commits,
                accuracy, extraction_failed, initial_correct, final_correct
    episode:    agreement, truth_member_correct (judge-less packs only),
                episode_scalar (1 iff judge picked truth_member), winner
"""

from __future__ import annotations

from typing import Any

import verifiers as vf
from verifiers.clients import Client
from verifiers.envs.debate.fields import EnumScoring, FieldSpec, classify_enum
from verifiers.envs.debate.prompts import (
    DebatePrompts,
    JudgeTemplate,
    _normalize_verdict_token,
)
from verifiers.parsers.parser import Parser
from verifiers.rubrics.judge_rubric import JudgeRubric
from verifiers.rubrics.multi_agent_rubric import MultiAgentRubric
from verifiers.types import (
    AssistantMessage,
    MARScore,
    MemberScore,
    State,
    TrajectoryStep,
    UserMessage,
)

JUDGE_SAMPLING_ARGS = {
    "temperature": 0.0,
    "max_completion_tokens": 256,
    "reasoning_effort": "medium",
}


# -- Module-local helpers (pure, testable without a rubric instance) ------


def question_from_state(state: State) -> str:
    """Return the first user message's content from ``state['prompt']``,
    tolerating both dict and pydantic message shapes."""
    for msg in state.get("prompt") or []:
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "")
        content = (
            msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
        )
        if role == "user" and content:
            return str(content)
    return ""


def winning_member(trajectory: list[TrajectoryStep]) -> str | None:
    """Last judge step's decision, or None if absent / malformed.

    Reverse-walk and break on the first judge step — whether or not it
    carries a 'decision'. Falling back to an earlier judge would
    silently install a stale verdict.
    """
    for step in reversed(trajectory):
        extras = step.get("extras", {})
        if extras.get("member_id") != "judge":
            continue
        return (extras.get("fields") or {}).get("decision")
    return None


def zero_sum_reward(member_id: str, winner: str | None) -> float:
    """+1 winner, −1 loser, 0 judge, 0 when no decision, 0 on tie.

    ``winner == "tie"`` is a first-class judge outcome in packs where
    the judge can declare a draw (selfplay.yaml does). Both debaters
    receive 0 — zero-sum holds (0 + 0 = 0) and the RAE baseline update
    is neutral.
    """
    if winner is None or winner == "tie" or member_id == "judge":
        return 0.0
    return 1.0 if member_id == winner else -1.0


def member_snapshot(
    member_id: str,
    steps: list[TrajectoryStep],
    prompts: DebatePrompts,
) -> dict[str, Any]:
    """One forward pass. Returns plain dict (no new type — the keys are
    the schema): commits, latest_spec, latest_had_answer, turns,
    parse_errors.

    Semantics:
      - commits:           chronological answer commits.
      - latest_spec:       spec at the phase of the most recent commit.
      - latest_had_answer: whether the member's LAST step had
                           ``extras['fields']['answer']`` — distinguishes
                           "wrong answer" from "latest step unparseable".
      - turns:             total step count for this member.
      - parse_errors:      count of steps with ``extras['parse_error']``.
    """
    commits: list[str] = []
    latest_spec: FieldSpec | None = None
    latest_had_answer = False
    parse_errors = 0
    for step in steps:
        extras = step.get("extras", {})
        if extras.get("parse_error"):
            parse_errors += 1
        fields = extras.get("fields") or {}
        if "answer" in fields:
            commits.append(str(fields["answer"]))
            specs = prompts.get_field_specs(member_id, extras.get("phase", "")) or {}
            latest_spec = specs.get("answer")
            latest_had_answer = True
        else:
            latest_had_answer = False
    return {
        "commits": commits,
        "latest_spec": latest_spec,
        "latest_had_answer": latest_had_answer,
        "turns": len(steps),
        "parse_errors": parse_errors,
    }


def maybe_judge(
    prompts: DebatePrompts,
    kind: str,
    client: Client | None,
    model: str,
) -> JudgeRubric | None:
    """Build a JudgeRubric iff the pack declares the template AND a client
    is available. Otherwise return None — score-time callers raise
    ``vf.Error`` if they actually need the missing piece."""
    tmpl = prompts.judges.get(kind)
    if tmpl is None or client is None:
        return None
    return JudgeRubric(
        parser=Parser(),
        judge_client=client,
        judge_model=model,
        judge_prompt=tmpl.user,
        judge_sampling_args=JUDGE_SAMPLING_ARGS,
    )


def resolve_verdict(
    verdict: str, tmpl: JudgeTemplate, *, kind: str, source_ref: str
) -> bool:
    """Exact-token verdict match. Substring matching ('correct' in
    'incorrect') silently inverts labels — fail loud instead."""
    token = _normalize_verdict_token(verdict)
    if token == tmpl.positive:
        return True
    if token == tmpl.negative:
        return False
    raise vf.Error(
        f"{kind} verdict {token!r} not in "
        f"{{{tmpl.positive!r}, {tmpl.negative!r}}} "
        f"(raw={verdict!r}); pack={source_ref}"
    )


# -- Rubric ---------------------------------------------------------------


class DebateRubric(MultiAgentRubric):
    """Debate scoring. Reward from judge outcome only; grader/matcher are
    diagnostic signals that never feed training reward."""

    def __init__(
        self,
        truth_member: str,
        members: list[str],
        prompts: DebatePrompts,
        judge_client: Client | None = None,
        judge_model: str = "gpt-4.1-nano",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if judge_client is not None and not isinstance(judge_client, Client):
            raise TypeError(
                f"judge_client must be a vf.Client subclass, "
                f"got {type(judge_client).__name__}"
            )
        self.truth_member = truth_member
        self.members = members
        self.prompts = prompts
        self.grader = maybe_judge(prompts, "grader", judge_client, judge_model)
        self.matcher = maybe_judge(prompts, "matcher", judge_client, judge_model)
        # Precomputed: does the pack declare an 'answer' field anywhere for
        # this member? Controls whether we attempt accuracy/extraction_failed.
        self.member_declares_answer: dict[str, bool] = {
            mid: any(
                "answer" in (phase_fields or {}) for phase_fields in phases.values()
            )
            for mid, phases in prompts.fields.items()
        }

    async def build_marscore(self, state: State) -> MARScore:
        if "answer" not in state:
            raise KeyError(
                "state missing 'answer' (ground truth) — dataset schema violation"
            )
        target = str(state["answer"])
        question = question_from_state(state)
        steps_by_mid = self.split_by_member(state)
        winner = winning_member(state.get("trajectory", []))
        if winner is None and "judge" in self.prompts.fields:
            raise vf.Error(
                f"Pack {self.prompts.source_ref!r} declares a judge but no "
                "decision was produced (early termination, malformed verdict, "
                "or missing judge step). Refusing to fall back to answer "
                "grading — that would silently generate fake training signal."
            )
        # Validate the judge's decision is an allowed member (minus the
        # judge itself — judges are neutral by convention) or "tie".
        # Without this check a typo / out-of-schema verdict would drop
        # through ``zero_sum_reward`` and both debaters would get −1 each
        # (member_id == winner fails for all), producing fake training
        # reward instead of an errored rollout.
        if winner is not None:
            valid_winners = (set(self.members) - {"judge"}) | {"tie"}
            if winner not in valid_winners:
                raise vf.Error(
                    f"Judge decision {winner!r} not in allowed set "
                    f"{sorted(valid_winners)}; pack={self.prompts.source_ref!r}"
                )

        snaps = {
            mid: member_snapshot(mid, steps_by_mid.get(mid, []), self.prompts)
            for mid in self.members
        }

        members: list[MemberScore] = []
        episode_metrics: dict[str, float] = {}
        for mid in self.members:
            m = snaps[mid]
            metrics: dict[str, float] = {"turns": float(m["turns"])}
            await self.emit_member_diagnostics(mid, m, target, question, state, metrics)
            members.append(
                MemberScore(
                    member_id=mid,
                    reward=zero_sum_reward(mid, winner),
                    parse_error_count=m["parse_errors"],
                    metrics=metrics,
                )
            )
        await self.emit_agreement(snaps, question, state, episode_metrics)
        await self.emit_truth_member_correct(
            snaps, target, question, state, winner, episode_metrics
        )

        return MARScore(
            members=members,
            episode_scalar=1.0 if winner == self.truth_member else 0.0,
            episode_metrics=episode_metrics,
            episode_categorical={"winner": winner},
        )

    def build_errored_marscore(
        self, state: State, *, error_type: str, error_phase: str
    ) -> MARScore:
        """Zero-reward MARScore with parse_error_count preserved for debug."""
        try:
            steps_by_mid = self.split_by_member(state)
        except ValueError:
            steps_by_mid = {}
        snaps = {
            mid: member_snapshot(mid, steps_by_mid.get(mid, []), self.prompts)
            for mid in self.members
        }
        return MARScore(
            members=[
                MemberScore(
                    member_id=mid,
                    reward=0.0,
                    parse_error_count=snaps[mid]["parse_errors"],
                )
                for mid in self.members
            ],
            episode_scalar=0.0,
            episode_metrics={"errored_rollout": 1.0},
            episode_error={"error_type": error_type, "error_phase": error_phase},
        )

    # -- diagnostics (weight-0; do not feed reward) ----------------------

    async def emit_member_diagnostics(
        self,
        member_id: str,
        m: dict[str, Any],
        target: str,
        question: str,
        state: State,
        dst: dict[str, float],
    ) -> None:
        """num_commits, num_unique_commits, accuracy, extraction_failed,
        initial_correct, final_correct. Gated on member≠judge; accuracy
        further gated on target + pack declaring 'answer' for this member."""
        if member_id == "judge":
            return
        dst["num_commits"] = float(len(m["commits"]))
        dst["num_unique_commits"] = float(len(set(m["commits"])))
        if not target or not self.member_declares_answer.get(member_id, False):
            return
        if not m["latest_had_answer"]:
            if m["turns"]:
                dst["extraction_failed"] = 1.0
            return
        spec, seq = m["latest_spec"], m["commits"]
        final = float(
            await self.verdict(seq[-1], target, question, spec, "grader", state)
        )
        dst["accuracy"] = final
        dst["final_correct"] = final
        dst["extraction_failed"] = 0.0
        dst["initial_correct"] = (
            final
            if seq[0] == seq[-1]
            else float(
                await self.verdict(seq[0], target, question, spec, "grader", state)
            )
        )

    async def emit_agreement(
        self,
        snaps: dict[str, dict[str, Any]],
        question: str,
        state: State,
        dst: dict[str, float],
    ) -> None:
        """agreement: do two debaters commit to the same final answer?"""
        debaters = [
            (mid, m)
            for mid, m in snaps.items()
            if mid != "judge" and m["latest_had_answer"]
        ]
        if len(debaters) < 2:
            return
        (_, a), (_, b) = debaters[0], debaters[1]
        spec = a["latest_spec"] or b["latest_spec"]
        dst["agreement"] = float(
            await self.verdict(
                b["commits"][-1], a["commits"][-1], question, spec, "matcher", state
            )
        )

    async def emit_truth_member_correct(
        self,
        snaps: dict[str, dict[str, Any]],
        target: str,
        question: str,
        state: State,
        winner: str | None,
        dst: dict[str, float],
    ) -> None:
        """truth_member_correct: judge-less test packs only.

        Grades the truth-side debater's final answer vs ground truth when
        the pack doesn't declare a judge. Telemetry only.
        """
        if winner is not None or "judge" in self.prompts.fields or not target:
            return
        truth = snaps.get(self.truth_member)
        if truth is None or not truth["latest_had_answer"]:
            return
        dst["truth_member_correct"] = float(
            await self.verdict(
                truth["commits"][-1],
                target,
                question,
                truth["latest_spec"],
                "grader",
                state,
            )
        )

    async def verdict(
        self,
        answer: str,
        target: str,
        question: str,
        spec: FieldSpec | None,
        kind: str,
        state: State,
    ) -> bool:
        """MCQ short-circuit when spec is EnumScoring; else LLM (grader or
        matcher). Matcher is just an asymmetric grader — A is target, B is
        response — so both routes share this primitive."""
        if spec and isinstance(spec.scoring, EnumScoring):
            c = classify_enum(str(answer), spec.scoring.values)
            t = classify_enum(str(target), spec.scoring.values)
            return c.is_valid and t.is_valid and c.canonical == t.canonical
        judge = self.grader if kind == "grader" else self.matcher
        if judge is None:
            raise vf.Error(
                f"Open-ended {kind} needs a judge_client; "
                f"pack={self.prompts.source_ref!r}"
            )
        raw = await judge.judge(
            prompt=[UserMessage(content=question)],
            completion=[AssistantMessage(content=str(answer))],
            answer=str(target),
            state=state,
        )
        return resolve_verdict(
            raw,
            self.prompts.judges[kind],
            kind=kind,
            source_ref=self.prompts.source_ref,
        )
