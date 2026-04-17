"""DebateEnv: thin MultiAgentEnv subclass for debate protocols.

Keeps only debate-specific concerns:
  * DebatePrompts prompt pack
  * XML field extraction from the public channel
  * Opponent visibility derived from ``think_visibility``
  * Role-ID mapping (member_id → role) and opponent attribution wrapping
  * Final debate transcript rendered into ``state['completion']``

Generic machinery (rollout loop, atomic simultaneous commit, stop
conditions, kernel threading, lineage cache, trajectory step append) is
inherited from :class:`MultiAgentEnv`.
"""

from __future__ import annotations

import logging
from typing import Any

from verifiers.clients import Client
from verifiers.envs.debate.fields import EnumScoring
from verifiers.envs.debate.parsing import extract_fields
from verifiers.envs.debate.prompts import (
    DebatePrompts,
    build_context,
    resolve_prompts,
)
from verifiers.envs.multi_actor_kernel import (
    KernelState,
    SlotProgram,
    StaticSchedule,
    TurnSlot,
    Utterance,
)
from verifiers.envs.multi_agent_env import MultiAgentEnv, VisibilityMode
from verifiers.types import Messages, State

_log = logging.getLogger(__name__)


class DebateEnv(MultiAgentEnv):
    """Debate-specific MultiAgentEnv.

    Subclasses :class:`MultiAgentEnv` and specialises four things:
      1. :meth:`build_prompt`  -- render via DebatePrompts
      2. :meth:`extract_fields` -- XML field parsing from public channel
      3. :meth:`visibility_policy` -- derived from ``think_visibility``
      4. :meth:`role_for_member` -- via ``role_for_actor`` map
      5. :meth:`render_completion` -- flatten trajectory into messages
    """

    def __init__(
        self,
        schedule: SlotProgram,
        prompts: DebatePrompts,
        members: list[str],
        *,
        role_for_actor: dict[str, str] | None = None,
        actor_overrides: dict[str, tuple[Client | None, str | None]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            schedule=schedule,
            members=members,
            actor_overrides=actor_overrides,
            think_tag=prompts.think_tag,
            **kwargs,
        )

        # Cross-check 1: env.members must equal rubric.members exactly.
        # Silent drift desyncs round_index (env) from per-member reward
        # attribution (rubric) and yields plausible-but-wrong training
        # signal. Only enforced when the rubric exposes a members attribute.
        rubric_members = getattr(self.rubric, "members", None)
        if rubric_members is not None and list(rubric_members) != list(members):
            raise ValueError(
                f"DebateEnv.members != rubric.members\n"
                f"  env    : {list(members)}\n"
                f"  rubric : {list(rubric_members)}\n"
                f"Both must be identical (same ids, same order) -- "
                f"round_index and reward attribution key off them."
            )

        # Cross-check 2: for StaticSchedule, unique slot actors must equal
        # the declared members set. Dynamic SlotProgram implementations
        # are exempt (actor set may be data-dependent).
        if isinstance(schedule, StaticSchedule):
            slot_actors: set[str] = set()
            for slot in schedule._slots:
                slot_actors.update(slot.actors)
            member_set = set(members)
            if slot_actors != member_set:
                raise ValueError(
                    f"DebateEnv.members != unique actors in StaticSchedule\n"
                    f"  members          : {sorted(member_set)}\n"
                    f"  schedule actors  : {sorted(slot_actors)}\n"
                    f"  in members only  : {sorted(member_set - slot_actors)}\n"
                    f"  in schedule only : {sorted(slot_actors - member_set)}"
                )

        self.prompts = prompts
        self._role_for_actor: dict[str, str] = dict(role_for_actor or {})

    # -- role resolution -----------------------------------------------------

    def role_for_member(self, member_id: str) -> str:
        return self._role_for_actor.get(member_id, member_id)

    def _member_round_count(self, member_id: str) -> int:
        """Count schedule slots where ``member_id`` participates as actor.

        Correct under:
          - sequential schedules (one actor per slot) — gives slots/N.
          - simultaneous schedules (N actors per slot) — gives per-member
            slot count, not the full slot count.
          - judge-inclusive schedules where a member (e.g. judge) appears
            in fewer slots than debaters.
        Falls back to 1 for dynamic (non-iterable) schedules; construction-
        time cross-check rejects mismatch between members and schedule
        actor set for the static case.
        """
        slots = getattr(self.schedule, "_slots", None)
        if slots is None:
            return 1
        count = sum(1 for slot in slots if member_id in slot.actors)
        return max(count, 1)

    # -- visibility policy ---------------------------------------------------

    def _render_opponent_message(
        self, utt: Utterance, viewer_id: str, viewer_role: str
    ) -> dict[str, str]:
        """Render one opponent utterance as a user message for ``viewer_id``.

        Selects raw vs. public channel per visibility_policy and wraps with
        speaker attribution. Shared by ``build_prompt`` and ``_format_history``.
        """
        vis = self.visibility_policy(utt, viewer_id)
        content = utt.raw_content if vis == "full" else utt.public_channel
        speaker_role = self.role_for_member(utt.member_id)
        content = self.prompts.wrap_opponent(
            utt.phase,
            content,
            member_id=utt.member_id,
            role_id=speaker_role,
            viewer_role=viewer_role,
        )
        return {"role": "user", "content": content}

    def visibility_policy(self, utt: Utterance, viewer_id: str) -> VisibilityMode:
        if utt.member_id == viewer_id:
            return "full"
        speaker_role = self.role_for_member(utt.member_id)
        viewer_role = self.role_for_member(viewer_id)
        vis = self.prompts.think_visibility.get(speaker_role, "disabled")
        if vis == "open":
            return "full"
        if vis == "visible_to_judge" and viewer_role == "judge":
            return "full"
        return "public_only"

    # -- prompt construction -------------------------------------------------

    async def build_prompt(
        self, state: State, member_id: str, slot: TurnSlot
    ) -> Messages:
        """Render the prompt for ``member_id`` at ``slot``.

        Monotonic-extension invariant (base-class contract): for a fixed
        member, the slot-N+1 prompt is equal byte-for-byte to the slot-N
        prompt on its leading messages and only adds a suffix. To achieve
        this, each own-turn in the transcript is rendered as the pair
        ``[instruction_that_preceded_it, assistant=raw_content]`` -- i.e.
        we re-render the instruction for that turn's phase/round_index.
        Opponent turns are rendered as wrapped user messages. The
        current turn's instruction + optional assistant-prefill sit at
        the tail. No contiguous-user-message consolidation (that would
        split one message into two across boundaries that shift when
        history grows).
        """
        kernel_state: KernelState = state["_kernel"]
        role = self.role_for_member(member_id)
        question = _extract_question(state)

        # Per-member num_rounds: count slots where this member participates
        # as an actor. Independent of global slot count so simultaneous
        # schedules (N actors in 1 slot) and judge-inclusive schedules (one
        # member participates in fewer slots) both give correct
        # is_first_round / is_last_round flags. For schedules that don't
        # expose slot iteration (dynamic), fall back to 1 — the static
        # check above already rejects that case at construction for non-
        # dynamic schedules.
        num_rounds = self._member_round_count(member_id)

        def _ctx_for(round_idx: int, phase: str) -> dict[str, Any]:
            return build_context(
                task_prompt=question,
                viewer_role=role,
                phase=phase,
                round_index=round_idx,
                num_rounds=num_rounds,
                answer=state.get("answer", ""),
            )

        # Positional round_index: count own commits already in the
        # transcript. Independent of slot_id values (which may be
        # sparse/semantic for custom schedules) — the N-th own turn is
        # always round N.
        own_commits_so_far = sum(
            1 for u in kernel_state.transcript if u.member_id == member_id
        )
        current_round = own_commits_so_far
        ctx_current = _ctx_for(current_round, slot.phase)

        msgs: list[dict[str, str]] = [
            {"role": "system", "content": self.prompts.render_system(role, ctx_current)},
        ]

        q_text = self.prompts.render_question(role, ctx_current)
        if q_text:
            msgs.append({"role": "user", "content": q_text})

        viewer_role = role
        # Positional round counter for own turns. Independent of slot_id
        # arithmetic so sparse / semantic slot_id schemes don't produce
        # nonsensical round labels in re-rendered past instructions.
        own_round_so_far = 0
        for utt in kernel_state.transcript:
            if utt.member_id == member_id:
                past_ctx = _ctx_for(own_round_so_far, utt.phase)
                past_instr = self.prompts.render_instruction(
                    role, utt.phase, past_ctx
                )
                if past_instr:
                    msgs.append({"role": "user", "content": past_instr})
                msgs.append({"role": "assistant", "content": utt.raw_content})
                own_round_so_far += 1
            else:
                msgs.append(self._render_opponent_message(utt, member_id, viewer_role))

        instruction = self.prompts.render_instruction(role, slot.phase, ctx_current)
        if instruction:
            msgs.append({"role": "user", "content": instruction})

        prefill = self.prompts.render_prefill(role, slot.phase, ctx_current)
        if prefill:
            msgs.append({"role": "assistant", "content": prefill})

        return msgs

    def _format_history(
        self, kernel_state: KernelState, viewer_id: str
    ) -> list[dict[str, str]]:
        """Format transcript entries for ``viewer_id``.

        Own utterances → assistant role, ``raw_content`` verbatim (KV cache
        coherence). Others → user role, content selected by
        :meth:`visibility_policy` and wrapped with speaker attribution.
        """
        viewer_role = self.role_for_member(viewer_id)
        msgs: list[dict[str, str]] = []
        for utt in kernel_state.transcript:
            if utt.member_id == viewer_id:
                msgs.append({"role": "assistant", "content": utt.raw_content})
            else:
                msgs.append(self._render_opponent_message(utt, viewer_id, viewer_role))
        return msgs

    # -- field extraction ----------------------------------------------------

    async def extract_fields(
        self, public_channel: str, member_id: str, slot: TurnSlot
    ) -> dict[str, Any] | None:
        """Extract XML fields from the public channel, per role+phase specs.

        Parse ambiguity (duplicate schema tags, raised as ValueError) is a
        per-step signal: we log and return None so the step is still
        appended and the rubric's failed_members path fires
        ``extraction_failed/{mid}=1.0``. Terminating would discard other
        members' valid commits.
        """
        role = self.role_for_member(member_id)
        specs = self.prompts.get_field_specs(role, slot.phase)
        if not specs:
            return None
        try:
            return extract_fields(public_channel, specs)
        except ValueError as exc:
            _log.warning(
                "field extraction failed for member=%s phase=%s: %s",
                member_id, slot.phase, exc,
            )
            return None

    # -- completion rendering ------------------------------------------------

    async def render_completion(self, state: State) -> None:
        state["completion"] = [
            msg for step in state["trajectory"] for msg in step["completion"]
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_question(state: State) -> str:
    """Extract the question text from ``state['prompt']`` (dataset messages)."""
    for msg in state.get("prompt") or []:
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "")
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
        if role == "user" and content:
            return content
    return ""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _validate_judge_templates(prompts: DebatePrompts) -> None:
    """Fail loud on two judge-template footguns:

    1. ``user_format_string`` empty -> JudgeRubric would render a 0-byte
       prompt at score time. Reject.
    2. Verdict token collides with an ``EnumScoring`` value of any
       ``answer`` field -> transcript greps misattribute judge output
       to a debater commit. Reject.
    """
    for kind, template in prompts.judges.items():
        if not template.user_format_string.strip():
            raise ValueError(
                f"Judge template {kind!r} has empty user_format_string. "
                "Production YAML packs populate this from the _grader/"
                "_matcher 'user' block at pack-load time; an empty value "
                "means the loader bypassed _compile_judge_blocks or the "
                "block was hand-constructed without it. JudgeRubric.judge "
                "would render a 0-byte prompt at score time."
            )
        pos = template.positive
        neg = template.negative
        for role_fields in prompts.fields.values():
            for phase_fields in role_fields.values():
                answer_spec = phase_fields.get("answer")
                if answer_spec is None:
                    continue
                if not isinstance(answer_spec.scoring, EnumScoring):
                    continue
                for enum_val in answer_spec.scoring.values:
                    upper = enum_val.upper()
                    if upper == pos or upper == neg:
                        raise ValueError(
                            f"Judge template {kind!r} verdict token "
                            f"{pos!r}/{neg!r} collides with answer enum "
                            f"value {enum_val!r}. Pick distinct verdict "
                            "tokens so transcript greps can't misattribute "
                            "judge output to a debater commit."
                        )


def load_environment(**kwargs: Any) -> DebateEnv:
    """Construct a DebateEnv from a prompt pack and a static schedule.

    Usage::

        from verifiers.envs.debate_env import load_environment

        env = load_environment(
            schedule_slots=[
                {"slot_id": 0, "actors": ["A"], "phase": "propose"},
                {"slot_id": 1, "actors": ["B"], "phase": "propose"},
                {"slot_id": 2, "actors": ["J"], "phase": "final"},
            ],
            members=["A", "B", "J"],
            truth_role="debater_a",
            prompts_ref="selfplay",
            role_for_actor={"A": "debater_a", "B": "debater_b", "J": "judge"},
            eval_dataset=my_dataset,
        )

    Required:
        schedule_slots, members, truth_role.
    Prompt source (exactly one): ``prompts_ref`` or ``prompts``.
    Optional: role_for_actor, actor_overrides, judge_client, judge_model,
    judge_api_key, judge_base_url, judge_max_retries, dataset/eval_dataset.
    """
    from verifiers.envs.debate_rubric import (
        DebateRubric,
        any_field_needs_open_ended,
    )

    raw_slots = kwargs.pop("schedule_slots")
    slots = tuple(
        TurnSlot(
            slot_id=s["slot_id"],
            actors=tuple(s["actors"]),
            phase=s.get("phase", ""),
        )
        for s in raw_slots
    )
    schedule = StaticSchedule(slots)

    prompts_ref = kwargs.pop("prompts_ref", None)
    prompts = kwargs.pop("prompts", None)
    if prompts is not None:
        if not isinstance(prompts, DebatePrompts):
            raise TypeError(f"Expected DebatePrompts, got {type(prompts).__name__}")
    elif prompts_ref is not None:
        prompts = resolve_prompts(prompts_ref)
    else:
        raise ValueError("Must provide either 'prompts_ref' or 'prompts'")

    _validate_judge_templates(prompts)

    members = kwargs.pop("members")
    truth_role = kwargs.pop("truth_role")
    judge_client = kwargs.pop("judge_client", None)
    judge_model = kwargs.pop("judge_model", "gpt-4.1-nano")
    judge_api_key = kwargs.pop("judge_api_key", None)
    judge_base_url = kwargs.pop("judge_base_url", None)
    judge_max_retries = kwargs.pop("judge_max_retries", 10)

    if judge_client is None and (judge_api_key or judge_base_url):
        from openai import AsyncOpenAI
        from verifiers.clients.openai_chat_completions_client import (
            OpenAIChatCompletionsClient,
        )
        judge_client = OpenAIChatCompletionsClient(
            AsyncOpenAI(
                api_key=judge_api_key,
                base_url=judge_base_url,
                max_retries=judge_max_retries,
            )
        )

    if (
        prompts.judges
        and any_field_needs_open_ended(prompts.fields)
        and judge_client is None
    ):
        declared = sorted(prompts.judges.keys())
        raise ValueError(
            f"load_environment: prompt pack {prompts.source_ref!r} declares "
            f"{len(declared)} judge template(s) ({declared}) and has at "
            "least one 'answer' field with non-enum scoring that routes to "
            "the LLM grader, but judge_client was not provided. Open-ended "
            "grading requires a vf.Client-wrapped judge client (e.g. "
            "OpenAIChatCompletionsClient(AsyncOpenAI(...))). Raw provider "
            "SDK clients are no longer accepted -- wrap them first."
        )
    if judge_client is not None and not isinstance(judge_client, Client):
        raise TypeError(
            f"load_environment: judge_client must be a vf.Client subclass "
            f"(got {type(judge_client).__name__}). Wrap raw provider SDK "
            "clients via OpenAIChatCompletionsClient(AsyncOpenAI(...)) or "
            "the equivalent for your provider."
        )
    rubric = DebateRubric(
        truth_role=truth_role,
        members=members,
        prompts=prompts,
        judge_client=judge_client,
        judge_model=judge_model,
    )

    role_for_actor = kwargs.pop("role_for_actor", None)
    actor_overrides = kwargs.pop("actor_overrides", None)

    return DebateEnv(
        schedule=schedule,
        prompts=prompts,
        members=members,
        role_for_actor=role_for_actor,
        actor_overrides=actor_overrides,
        rubric=rubric,
        **kwargs,
    )
