"""Debate rubric: W (winner) + G (grading) + M (matching) scoring."""

from __future__ import annotations

import asyncio
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
from verifiers.types import State

OutcomeFn = Callable[[State], str | None]
"""(state) -> winning role_id, or None if no winner."""


def _extract_question(state: State) -> str:
    """Pull the first user message from the prompt as the question text."""
    for msg in state.get("prompt", []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "")
    return ""


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
        # Fail loud at construction time if the pack declares LLM-judge
        # templates *and* at least one 'answer' field routes to the LLM
        # grader/matcher path (i.e. is not covered by an EnumScoring fast
        # path). Packs that declare _grader/_matcher as dead-code fallback
        # but score every answer field via classify_enum (e.g. selfplay MCQ)
        # must still construct cleanly without a judge client.
        needs_open_ended = any_field_needs_open_ended(prompts.fields)
        if prompts.judges and needs_open_ended and judge_client is None:
            declared = sorted(prompts.judges.keys())
            raise ValueError(
                f"Prompt pack {prompts.source_ref!r} declares "
                f"{len(declared)} judge template(s) ({declared}) and has "
                "at least one 'answer' field with non-enum scoring that "
                "routes to the LLM grader, but no judge_client was "
                "provided to DebateRubric. Open-ended grading requires a "
                "vf.Client-wrapped judge client (e.g. "
                "OpenAIChatCompletionsClient(AsyncOpenAI(...))). Passing "
                "None here would silently burn rollout budget before "
                "failing at score time."
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
                judge_prompt=grader_template.user_format_string,
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
                judge_prompt=matcher_template.user_format_string,
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

    def _extract_commit_sequences(
        self, state: State
    ) -> dict[str, list[str]]:
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
            prompt=[{"role": "user", "content": question}],
            completion=[{"role": "assistant", "content": str(answer)}],
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
            prompt=[{"role": "user", "content": question}],
            completion=[{"role": "assistant", "content": str(b)}],
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
        raise RuntimeError(
            f"{kind} returned unrecognized verdict token {token!r} "
            f"(raw response: {verdict!r}). "
            f"Expected {template.positive!r} or {template.negative!r}. "
            f"Pack source: {self.prompts.source_ref}"
        )

    # -- Main scoring -------------------------------------------------------

    async def score_rollout(self, state: State) -> None:
        # Short-circuit on already-errored rollouts. The rollout loop in
        # DebateEnv.rollout sets state['error'] on vf.Error and
        # state['prompt_too_long'] on vf.OverlongPromptError. Running the
        # W/G/M pipeline on an incomplete trajectory would either trip the
        # G7.3 "missing judge" invariant and raise (killing score_group's
        # asyncio.gather and propagating to the whole batch) or invent a
        # score from a trajectory the rollout already flagged as broken.
        # Instead: emit errored_rollout=1.0 + error_type, reward=0.0, and
        # return. This is a record-the-failure path, not a swallow-error
        # path — downstream consumers can filter on errored_rollout.
        errored = state.get("error") is not None
        overlong = state.get("prompt_too_long", False)
        if errored or overlong:
            if overlong:
                error_type = "prompt_too_long"
            else:
                error_type = type(state["error"]).__name__
            state["reward"] = 0.0
            # metrics is dict[str, float] by framework contract; error
            # taxonomy goes on state["error_info"] so RubricGroup
            # aggregation + TUI display don't have to coerce strings.
            state["metrics"] = {"errored_rollout": 1.0}
            state["error_info"] = {
                "error_type": error_type,
                "error_phase": "rollout",
            }
            state["commits"] = {}
            state["member_rewards"] = {m: 0.0 for m in self.members}
            state["member_metrics"] = {m: {} for m in self.members}
            state["episode_metrics"] = {"errored_rollout": 1.0}
            return

        try:
            target = state["answer"]
        except KeyError as exc:
            raise KeyError(
                "state is missing 'answer' (ground truth) — dataset schema "
                "violation. Rubric will not silently coerce to empty."
            ) from exc
        try:
            await self._score_rollout_body(state, target)
        except vf.Error as exc:
            # Capture on state so maybe_retry.reraise_error_from_state in
            # verifiers/utils/async_utils.py can discover it and trigger
            # tenacity retry on retryable subclasses (vf.InfraError,
            # vf.InvalidModelResponseError). error_info carries the
            # downstream-filter taxonomy; metrics stays dict[str, float]
            # so RubricGroup aggregation + TUI display don't choke on
            # string values.
            state["error"] = exc
            state["reward"] = 0.0
            state["metrics"] = {"errored_rollout": 1.0}
            state["error_info"] = {
                "error_type": type(exc).__name__,
                "error_phase": "scoring",
            }
            state["commits"] = {}
            state["member_rewards"] = {m: 0.0 for m in self.members}
            state["member_metrics"] = {m: {} for m in self.members}
            state["episode_metrics"] = {"errored_rollout": 1.0}
            return

    async def _score_rollout_body(self, state: State, target: Any) -> None:
        """Main W/G/M scoring body. Kept as a separate method so
        ``score_rollout`` can wrap it in a single vf.Error boundary without
        interfering with the F2 pre-scoring short-circuit or the ground-
        truth KeyError propagation."""
        roles, answers, answer_specs, failed_members = self._extract_debate_state(state)
        question = _extract_question(state)
        metrics: dict[str, float] = {}
        # Structured dual-write (MultiAgentRubric contract). During migration
        # we populate both the flat `metrics` dict (legacy string-keyed) and
        # these three structured dicts. Bridge prefers structured, falls back
        # to flat. TODO(phase-4-followup): drop flat writes once Phase 5 lands
        # and the bridge's legacy fallback path is removed.
        member_rewards: dict[str, float] = {}
        member_metrics: dict[str, dict[str, float]] = {m: {} for m in self.members}
        episode_metrics: dict[str, float] = {}

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

        if winning_role is not None:
            state["reward"] = 1.0 if winning_role == self.truth_role else 0.0
        elif judge_expected:
            raise RuntimeError(
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
            state["reward"] = 1.0 if correct else 0.0
        else:
            state["reward"] = 0.0

        # Per-member rewards + turn counts
        for mid in self.members:
            role = roles.get(mid, mid)
            if winning_role is not None:
                r = 1.0 if role == winning_role else 0.0
            else:
                r = state["reward"] if mid == truth_member else 0.0
            metrics[f"reward/{mid}"] = r
            member_rewards[mid] = r
            turns = sum(
                1
                for s in state.get("trajectory", [])
                if s.get("extras", {}).get("member_id") == mid
            )
            metrics[f"turns/{mid}"] = float(turns)
            member_metrics[mid]["turns"] = float(turns)

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
                    metrics[f"accuracy/{mid}"] = acc
                    metrics[f"extraction_failed/{mid}"] = 0.0
                    member_metrics[mid]["accuracy"] = acc
                    member_metrics[mid]["extraction_failed"] = 0.0
                elif mid in failed_members:
                    metrics[f"extraction_failed/{mid}"] = 1.0
                    member_metrics[mid]["extraction_failed"] = 1.0

        # Flip diagnostics (eval-only; no training reward path touches these)
        # Per member, walk all commits, grade first and last against GT, count
        # uniqueness. Store the raw ordered commit sequence on state["commits"].
        commits = self._extract_commit_sequences(state)
        state["commits"] = commits
        for mid in self.members:
            if roles.get(mid) == "judge":
                continue
            seq = commits.get(mid, [])
            num_commits = len(seq)
            num_unique = len(set(seq))
            metrics[f"num_commits/{mid}"] = float(num_commits)
            metrics[f"num_unique_commits/{mid}"] = float(num_unique)
            member_metrics[mid]["commits"] = float(num_commits)
            member_metrics[mid]["num_unique_commits"] = float(num_unique)

            if not seq or not target:
                continue
            # Skip flip-correctness diagnostics when the authoritative latest
            # step was unparseable — grading a stale commit would be
            # misleading, and accuracy/{mid} is absent by design.
            final_correct = metrics.get(f"accuracy/{mid}")
            if final_correct is None:
                continue
            spec = answer_specs.get(mid)
            metrics[f"final_correct/{mid}"] = final_correct
            member_metrics[mid]["final_correct"] = final_correct
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
            metrics[f"initial_correct/{mid}"] = init_val
            member_metrics[mid]["initial_correct"] = init_val

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
            agreement = 1.0 if same else 0.0
            metrics["agreement"] = agreement
            episode_metrics["agreement"] = agreement

        # Winner index
        if winning_role is not None:
            winner_member = next(
                (m for m in self.members if roles.get(m) == winning_role), None
            )
            winner_val = (
                float(self.members.index(winner_member))
                if winner_member
                else -1.0
            )
        else:
            winner_val = -1.0
        metrics["winner"] = winner_val
        episode_metrics["winner"] = winner_val

        state["metrics"] = metrics
        state["member_rewards"] = member_rewards
        state["member_metrics"] = member_metrics
        state["episode_metrics"] = episode_metrics

    async def score_group(self, states: list[State]) -> None:
        # score_rollout captures vf.Error onto state["error"] itself (never
        # escapes), so batch-isolation for retryable judge outages is
        # satisfied at a lower layer and maybe_retry can pick the error up
        # via reraise_error_from_state. Anything that DOES escape here is
        # either a dataset schema violation (KeyError from missing
        # state["answer"]) or a programming bug (AttributeError, etc.) —
        # both MUST propagate loud. Bare gather, no return_exceptions.
        await asyncio.gather(*(self.score_rollout(s) for s in states))
