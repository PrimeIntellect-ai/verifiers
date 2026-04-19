"""Debate rubric: W (winner) + G (grading) + M (matching) scoring."""

from __future__ import annotations

from typing import Any, Callable

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
from verifiers.types import AssistantMessage, MARScore, MemberScore, State, UserMessage

OutcomeFn = Callable[[State], str | None]
"""(state) -> winning role_id, or None if no winner."""


def _extract_question(state: State) -> str:
    """Extract the question text from ``state['prompt']`` (dataset messages).

    Handles both plain-dict messages and Pydantic ``CustomBaseModel`` messages
    (``UserMessage`` etc.) — env layer may pass either depending on whether
    ``maybe_normalize_messages`` ran.
    """
    for msg in state.get("prompt") or []:
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "")
        content = (
            msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
        )
        if role == "user" and content:
            return content
    return ""


def _count_parse_errors(state: State, members: list[str]) -> dict[str, int]:
    """Count quarantined utterances per member by walking the trajectory once.

    A step's quarantine flag is set by the kernel when ``parse_channels`` rejected
    the raw output; downstream training masks those completion tokens (P0-2).
    """
    counts: dict[str, int] = {m: 0 for m in members}
    for step in state.get("trajectory", []):
        extras = step.get("extras", {})
        mid = extras.get("member_id")
        if mid is None or mid not in counts:
            continue
        if extras.get("parse_error"):
            counts[mid] += 1
    return counts


def any_field_needs_open_ended(
    fields: dict[str, dict[str, dict[str, FieldSpec]]],
) -> bool:
    """Walk the 3-level (role → phase → field) tree and return True iff any
    ``answer`` field routes through the LLM grader/matcher path.

    The grader and matcher are only consulted when ``spec.scoring`` is not
    an :class:`EnumScoring` instance (including when the spec is missing or
    ``spec.scoring is None``). Everything else — Binary, Numeric, or an
    absent spec — is open-ended.

    Used by the eager judge_client gate: we must only demand a judge client
    when a scored path actually needs one. A pack that declares dead-code
    ``_grader`` / ``_matcher`` fallbacks (e.g. selfplay's MCQ-only pack)
    must still construct cleanly without a judge client.
    """
    for role_tree in fields.values():
        for phase_tree in role_tree.values():
            spec = phase_tree.get("answer")
            if spec is None:
                continue
            if spec.scoring is None:
                return True
            if not isinstance(spec.scoring, EnumScoring):
                return True
    return False


class DebateRubric(MultiAgentRubric):
    """Scores a debate via three concerns:

    - **W (Winner)**: judge decision -> per-member + episode reward
    - **G (Grading)**: each debater's answer vs ground truth -> accuracy metrics
    - **M (Matching)**: debater A vs B answer -> agreement metric
    """

    def __init__(
        self,
        truth_role: str,
        members: list[str],
        prompts: DebatePrompts,
        judge_client: Client | None = None,
        judge_model: str = "gpt-4.1-nano",
        outcome_fn: OutcomeFn | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # DebateRubric's two judge-client preconditions, both validated here
        # because this is the component whose scoring path would crash if
        # either is violated. The factory (load_environment) is a pure
        # dispatcher and does not duplicate these checks.
        #
        # 1. Judge client must be a vf.Client subclass (raw provider SDK
        #    clients are rejected — they bypass uniform exception handling
        #    and tool/message conversion).
        if judge_client is not None and not isinstance(judge_client, Client):
            raise TypeError(
                f"DebateRubric.judge_client must be a vf.Client subclass, "
                f"got {type(judge_client).__name__}. Wrap raw provider SDK "
                "clients via OpenAIChatCompletionsClient(AsyncOpenAI(...)) "
                "or the equivalent for your provider."
            )
        # 2. If the pack declares LLM-judge templates AND at least one
        #    'answer' field routes to the LLM grader/matcher path (non-enum
        #    scoring), a judge_client is required. Packs that declare
        #    _grader/_matcher as dead-code fallback but score every answer
        #    field via classify_enum (e.g. selfplay MCQ) must still
        #    construct cleanly without a judge client.
        needs_open_ended = any_field_needs_open_ended(prompts.fields)
        if prompts.judges and needs_open_ended and judge_client is None:
            declared = sorted(prompts.judges.keys())
            raise ValueError(
                f"DebateRubric: pack {prompts.source_ref!r} declares "
                f"{len(declared)} judge template(s) ({declared}) and has "
                "at least one 'answer' field with non-enum scoring routing "
                "to the LLM grader, but judge_client was not provided. "
                "Open-ended grading requires a vf.Client-wrapped judge "
                "client (e.g. OpenAIChatCompletionsClient(AsyncOpenAI("
                "api_key=..., base_url=...))). Construction without one "
                "would silently burn rollout budget before failing at "
                "score time. If your env-server uses load_environment, "
                "pass judge_api_key + judge_base_url to auto-construct."
            )
        self.truth_role = truth_role
        self.members = members
        self.prompts = prompts
        self.judge_client = judge_client
        self.judge_model = judge_model
        self.outcome_fn = outcome_fn or self._default_outcome

        # Compose two JudgeRubric instances — HybridMathRubric pattern:
        # ship the component whenever the pack declares it AND a client
        # is available, regardless of any_field_needs_open_ended() above.
        # The helper is a load-time "demand a client at all" gate, not a
        # per-rubric gate: mixed-mode packs (enum answer in one phase,
        # open-ended in another) still route through the grader at score
        # time, and tests that pass spec=None need the grader exposed
        # against the default pack. JudgeRubric handles the vf.Client
        # call, exception propagation, and state["judge_response"]
        # caching.
        judge_sampling_args = {
            "temperature": 0.0,
            "max_completion_tokens": 256,
            "reasoning_effort": "medium",
        }
        parser = Parser()
        grader_template = prompts.judges.get("grader")
        if grader_template is not None and judge_client is not None:
            self.grader_rubric: JudgeRubric | None = JudgeRubric(
                parser=parser,
                judge_client=judge_client,
                judge_model=judge_model,
                judge_prompt=grader_template.user,
                judge_sampling_args=judge_sampling_args,
            )
        else:
            self.grader_rubric = None
        matcher_template = prompts.judges.get("matcher")
        if matcher_template is not None and judge_client is not None:
            self.matcher_rubric: JudgeRubric | None = JudgeRubric(
                parser=parser,
                judge_client=judge_client,
                judge_model=judge_model,
                judge_prompt=matcher_template.user,
                judge_sampling_args=judge_sampling_args,
            )
        else:
            self.matcher_rubric = None

    # -- Data extraction (one reverse pass over trajectory) -----------------

    def _extract_debate_state(
        self, state: State
    ) -> tuple[
        dict[str, str],
        dict[str, str],
        dict[str, FieldSpec],
        set[str],
    ]:
        """Reverse-walk trajectory once and build:
          - roles: {member_id: role_id}
          - answers: {member_id: last_answer}  (only LATEST step counts)
          - answer_specs: {member_id: FieldSpec for 'answer' at that phase}
          - failed_members: set of member_ids whose LATEST step had no
            'fields.answer' (extraction failed at the authoritative step)

        Invariant: a member's FIRST step encountered in reverse order is
        their LATEST step chronologically. Only that step is authoritative
        for answer/spec. We do NOT fall back to older steps — a stale
        commit must not be graded as the final answer.
        """
        roles: dict[str, str] = {}
        answers: dict[str, str] = {}
        answer_specs: dict[str, FieldSpec] = {}
        failed_members: set[str] = set()
        seen_members: set[str] = set()
        for step in reversed(state.get("trajectory", [])):
            extras = step.get("extras", {})
            mid = extras.get("member_id")
            if mid is None:
                continue
            if mid in seen_members:
                continue
            if "role_id" not in extras:
                raise KeyError(
                    f"trajectory step for member {mid!r} missing 'role_id' "
                    "in extras — framework invariant violation"
                )
            rid = extras["role_id"]
            roles[mid] = rid
            seen_members.add(mid)
            fields = extras.get("fields")
            if fields and "answer" in fields:
                answers[mid] = fields["answer"]
                phase = extras.get("phase", "")
                specs = self.prompts.get_field_specs(rid, phase)
                if specs and "answer" in specs:
                    answer_specs[mid] = specs["answer"]
            else:
                failed_members.add(mid)
        return roles, answers, answer_specs, failed_members

    def _extract_commit_sequences(self, state: State) -> dict[str, list[str]]:
        """Walk trajectory forward, extract every answer commit per member
        in chronological order. Used for flip diagnostics (initial_correct,
        final_correct, num_unique_commits). Returns {member_id: [a0, a1, ...]}.
        """
        sequences: dict[str, list[str]] = {}
        for step in state.get("trajectory", []):
            extras = step.get("extras", {})
            mid = extras.get("member_id")
            if mid is None:
                continue
            fields = extras.get("fields") or {}
            if "answer" in fields:
                sequences.setdefault(mid, []).append(str(fields["answer"]))
        return sequences

    # -- W: Winner determination --------------------------------------------

    def _default_outcome(self, state: State) -> str | None:
        """Return the LAST judge step's decision, or None if absent/malformed.

        We break on the first judge step encountered in reverse order —
        whether or not it carries a 'decision' field. Falling back to an
        earlier judge would silently install a stale verdict.
        """
        for step in reversed(state.get("trajectory", [])):
            extras = step.get("extras", {})
            if extras.get("role_id") != "judge":
                continue
            fields = extras.get("fields")
            if fields and "decision" in fields:
                return fields["decision"]
            return None
        return None

    # -- G: Grading ---------------------------------------------------------

    async def _grade(
        self,
        answer: str,
        target: str,
        question: str,
        spec: FieldSpec | None,
        state: State,
    ) -> bool:
        """Is answer correct? MCQ: classify_enum. Open-ended: LLM _grader."""
        if spec and isinstance(spec.scoring, EnumScoring):
            c = classify_enum(str(answer), spec.scoring.values)
            t = classify_enum(str(target), spec.scoring.values)
            return c.is_valid and t.is_valid and c.canonical == t.canonical

        if self.grader_rubric is None:
            if "grader" not in self.prompts.judges:
                raise RuntimeError(
                    "Open-ended grading requires a _grader block in the YAML "
                    "prompt pack."
                )
            raise RuntimeError(
                "Open-ended grading requires a judge_client at DebateRubric "
                "construction time. Pack declares a _grader template but no "
                "vf.Client-wrapped judge_client was provided."
            )
        grader_template = self.prompts.judges["grader"]
        verdict = await self.grader_rubric.judge(
            prompt=[UserMessage(content=question)],
            completion=[AssistantMessage(content=str(answer))],
            answer=str(target),
            state=state,
        )
        return self._resolve_verdict(verdict, grader_template, kind="grader")

    # -- M: Matching --------------------------------------------------------

    async def _match(
        self,
        a: str,
        b: str,
        question: str,
        spec: FieldSpec | None,
        state: State,
    ) -> bool:
        """Do answers agree? MCQ: string match. Open-ended: LLM _matcher.

        The matcher is reformulated as an asymmetric grader: A is passed as
        the reference/target, B as the model response. This lets us reuse
        JudgeRubric's {question, answer, response} format contract.
        """
        if spec and isinstance(spec.scoring, EnumScoring):
            ca = classify_enum(str(a), spec.scoring.values)
            cb = classify_enum(str(b), spec.scoring.values)
            return ca.is_valid and cb.is_valid and ca.canonical == cb.canonical

        if self.matcher_rubric is None:
            if "matcher" not in self.prompts.judges:
                raise RuntimeError(
                    "Open-ended matching requires a _matcher block in the "
                    "YAML prompt pack."
                )
            raise RuntimeError(
                "Open-ended matching requires a judge_client at DebateRubric "
                "construction time. Pack declares a _matcher template but no "
                "vf.Client-wrapped judge_client was provided."
            )
        matcher_template = self.prompts.judges["matcher"]
        verdict = await self.matcher_rubric.judge(
            prompt=[UserMessage(content=question)],
            completion=[AssistantMessage(content=str(b))],
            answer=str(a),
            state=state,
        )
        return self._resolve_verdict(verdict, matcher_template, kind="matcher")

    def _resolve_verdict(
        self, verdict: str, template: JudgeTemplate, *, kind: str
    ) -> bool:
        """Exact-token verdict classification. Raises on anything else.

        `template.positive` / `.negative` are already normalized (uppercased,
        punct-stripped first word) at pack-load time — so we normalize the
        raw verdict the same way and compare for equality. Substring matching
        ("correct" in "incorrect") silently inverts labels; fail loud instead.
        """
        token = _normalize_verdict_token(verdict)
        if token == template.positive:
            return True
        if token == template.negative:
            return False
        # vf.Error (not RuntimeError) so the rollout quarantines via
        # MultiAgentRubric.score_rollout's catch instead of killing the
        # whole batch through asyncio.gather propagation.
        raise vf.Error(
            f"{kind} returned unrecognized verdict token {token!r} "
            f"(raw response: {verdict!r}). "
            f"Expected {template.positive!r} or {template.negative!r}. "
            f"Pack source: {self.prompts.source_ref}"
        )

    # -- Main scoring -------------------------------------------------------

    def build_errored_marscore(
        self, state: State, *, error_type: str, error_phase: str
    ) -> MARScore:
        """Build the zero-reward MARScore used on rollout/scoring failures."""
        parse_errors = _count_parse_errors(state, self.members)
        return MARScore(
            members=[
                MemberScore(
                    member_id=mid,
                    role_id=mid,  # role unknown on error path; fall back to mid
                    reward=0.0,
                    parse_error_count=parse_errors.get(mid, 0),
                )
                for mid in self.members
            ],
            episode_scalar=0.0,
            episode_metrics={"errored_rollout": 1.0},
            episode_error={"error_type": error_type, "error_phase": error_phase},
        )

    async def build_marscore(self, state: State) -> MARScore:
        """Main W/G/M scoring body.

        The base ``MultiAgentRubric`` catches only ``vf.Error`` around this
        method. Dataset schema violations (for example missing
        ``state["answer"]``) and programming bugs still propagate loud.
        """
        try:
            target = state["answer"]
        except KeyError as exc:
            raise KeyError(
                "state is missing 'answer' (ground truth) — dataset schema "
                "violation. Rubric will not silently coerce to empty."
            ) from exc

        roles, answers, answer_specs, failed_members = self._extract_debate_state(state)
        question = _extract_question(state)
        parse_errors = _count_parse_errors(state, self.members)

        # Per-member working dict — projects to MemberScore.metrics later.
        # Float-only by design (no bool/int/str): the aggregator averages these
        # without a type check; mixing in categorical sentinels is the same
        # ZIP-code mistake the episode-level split solves.
        per_member: dict[str, dict[str, float]] = {m: {} for m in self.members}
        member_rewards: dict[str, float] = {}
        episode_metrics: dict[str, float] = {}
        episode_categorical: dict[str, str | None] = {}

        # W: winner
        winning_role = self.outcome_fn(state)
        truth_member = next(
            (m for m in self.members if roles.get(m) == self.truth_role), None
        )
        # "Does the YAML contract declare a judge?" — static signal.
        # Using trajectory-observed roles is unsafe: early termination
        # (parser raise, empty response, overlong prompt, network error)
        # can produce a rollout with no judge step and bypass the check.
        judge_expected = "judge" in self.prompts.fields

        # episode_scalar is single-purpose: "did the truth side win the
        # judge's vote." It is NEVER populated from the grader's correctness
        # signal. The judge-less fallback (test packs only) records the
        # grader's verdict in episode_metrics["truth_member_correct"] for
        # telemetry, but does NOT feed member_rewards — that would silently
        # mix two different upstream signals (judge verdict vs grader call)
        # into the same trainer-facing field.
        truth_member_correct: float | None = None
        if winning_role is not None:
            episode_scalar = 1.0 if winning_role == self.truth_role else 0.0
        elif judge_expected:
            # vf.Error so the rollout quarantines (errored MARScore, batch
            # survives asyncio.gather) rather than killing scoring outright.
            raise vf.Error(
                f"Prompt pack {self.prompts.source_ref!r} declares a judge but "
                "no decision was produced (winning_role is None). This happens "
                "when the judge never ran (early termination: parser raise, "
                "empty response, overlong prompt, network error) or produced "
                "a malformed verdict. Refusing to fall back to answer-grading — "
                "that would silently generate fake training signal."
            )
        elif truth_member and answers.get(truth_member) and target:
            correct = await self._grade(
                answers[truth_member],
                target,
                question,
                answer_specs.get(truth_member),
                state,
            )
            # Telemetry only — this signal does NOT enter member_rewards.
            truth_member_correct = 1.0 if correct else 0.0
            episode_scalar = 0.0
        else:
            episode_scalar = 0.0
        if truth_member_correct is not None:
            episode_metrics["truth_member_correct"] = truth_member_correct

        # Per-member rewards + turn counts
        # Single source: judge verdict. Win-based, no truth_role coupling.
        # Fallback (no verdict): all zero — the grader path produces only
        # telemetry, never reward.
        for mid in self.members:
            role = roles.get(mid, mid)
            if winning_role is not None:
                r = 1.0 if role == winning_role else 0.0
            else:
                r = 0.0
            member_rewards[mid] = r
            turns = sum(
                1
                for s in state.get("trajectory", [])
                if s.get("extras", {}).get("member_id") == mid
            )
            per_member[mid]["turns"] = float(turns)

        # G: accuracy (per-member; gated on YAML declaring answer fields for
        # that member's role). extraction_failed/{mid} disambiguates
        # "wrong answer" from "couldn't parse one" — both used to collapse
        # into accuracy=0.0.
        if target:
            for mid in self.members:
                role = roles.get(mid)
                if role is None or role == "judge":
                    continue
                role_fields = self.prompts.fields.get(role) or {}
                declares_answer = any(
                    "answer" in (trigger_fields or {})
                    for trigger_fields in role_fields.values()
                )
                if not declares_answer:
                    continue
                if mid in answers:
                    correct = await self._grade(
                        answers[mid], target, question, answer_specs.get(mid), state
                    )
                    acc = 1.0 if correct else 0.0
                    per_member[mid]["accuracy"] = acc
                    per_member[mid]["extraction_failed"] = 0.0
                elif mid in failed_members:
                    per_member[mid]["extraction_failed"] = 1.0

        # Flip diagnostics (eval-only; no training reward path touches these)
        # Per member, walk all commits, grade first and last against GT, count
        # uniqueness. Commits are recomputable from the trajectory; we don't
        # persist them on state.
        commits = self._extract_commit_sequences(state)
        for mid in self.members:
            if roles.get(mid) == "judge":
                continue
            seq = commits.get(mid, [])
            num_commits = len(seq)
            num_unique = len(set(seq))
            per_member[mid]["num_commits"] = float(num_commits)
            per_member[mid]["num_unique_commits"] = float(num_unique)

            if not seq or not target:
                continue
            # Skip flip-correctness diagnostics when the authoritative latest
            # step was unparseable — grading a stale commit would be
            # misleading, and accuracy is absent by design.
            final_correct = per_member[mid].get("accuracy")
            if final_correct is None:
                continue
            spec = answer_specs.get(mid)
            per_member[mid]["final_correct"] = final_correct
            # initial_correct needs a separate grade only if seq[0] differs
            # from the canonical (latest) answer.
            canonical = answers.get(mid)
            if seq[0] == canonical:
                init_val = final_correct
            else:
                initial_correct = await self._grade(
                    seq[0], target, question, spec, state
                )
                init_val = 1.0 if initial_correct else 0.0
            per_member[mid]["initial_correct"] = init_val

        # M: agreement (conditional on 2+ debater answers)
        debater_answers = [
            (mid, answers[mid])
            for mid in self.members
            if roles.get(mid) != "judge" and mid in answers
        ]
        if len(debater_answers) >= 2:
            a_mid, a_ans = debater_answers[0]
            b_mid, b_ans = debater_answers[1]
            spec = answer_specs.get(a_mid) or answer_specs.get(b_mid)
            same = await self._match(a_ans, b_ans, question, spec, state)
            episode_metrics["agreement"] = 1.0 if same else 0.0

        # Winner — categorical, not an averageable scalar.
        # Per-role winrates are computable at aggregation time from
        # MemberScore.reward (mean over rollouts grouped by role_id).
        episode_categorical["winner"] = winning_role  # str | None

        return MARScore(
            members=[
                MemberScore(
                    member_id=mid,
                    role_id=roles.get(mid, mid),
                    reward=member_rewards[mid],
                    parse_error_count=parse_errors.get(mid, 0),
                    metrics=per_member[mid],
                )
                for mid in self.members
            ],
            episode_scalar=episode_scalar,
            episode_metrics=episode_metrics,
            episode_categorical=episode_categorical,
        )
