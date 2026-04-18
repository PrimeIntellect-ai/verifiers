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
from verifiers.envs.debate.parsing import extract_fields
from verifiers.envs.debate.prompts import (
    DebatePrompts,
    build_context,
    resolve_prompts,
)
from verifiers.envs.multi_agent_kernel import (
    KernelState,
    SlotProgram,
    StaticSchedule,
    TurnSlot,
    Utterance,
)
from verifiers.envs.debate_rubric import _extract_question
from verifiers.envs.multi_agent_env import MultiAgentEnv, VisibilityMode
from verifiers.types import Messages, State

_log = logging.getLogger(__name__)


class DebateEnv(MultiAgentEnv):
    """Debate-specific MultiAgentEnv.

    Subclasses :class:`MultiAgentEnv` and specialises four things:
      1. :meth:`build_prompt`  -- render via DebatePrompts
      2. :meth:`extract_fields` -- XML field parsing from public channel
      3. :meth:`visibility_policy` -- derived from ``think_visibility``
      4. :meth:`role_for_member` -- via ``role_for_agent`` map
      5. :meth:`render_completion` -- flatten trajectory into messages
    """

    def __init__(
        self,
        schedule: SlotProgram,
        prompts: DebatePrompts,
        members: list[str],
        *,
        role_for_agent: dict[str, str] | None = None,
        agent_overrides: dict[str, tuple[Client | None, str | None]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            schedule=schedule,
            members=members,
            agent_overrides=agent_overrides,
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

        # Cross-check 2: for StaticSchedule, unique slot agents must equal
        # the declared members set. Dynamic SlotProgram implementations
        # are exempt (agent set may be data-dependent).
        if isinstance(schedule, StaticSchedule):
            slot_agents: set[str] = set()
            for slot in schedule._slots:
                slot_agents.update(slot.agents)
            member_set = set(members)
            if slot_agents != member_set:
                raise ValueError(
                    f"DebateEnv.members != unique agents in StaticSchedule\n"
                    f"  members          : {sorted(member_set)}\n"
                    f"  schedule agents  : {sorted(slot_agents)}\n"
                    f"  in members only  : {sorted(member_set - slot_agents)}\n"
                    f"  in schedule only : {sorted(slot_agents - member_set)}"
                )

        self.prompts = prompts
        self._role_for_agent: dict[str, str] = dict(role_for_agent or {})

    # -- role resolution -----------------------------------------------------

    def role_for_member(self, member_id: str) -> str:
        return self._role_for_agent.get(member_id, member_id)

    def _member_round_count(self, member_id: str) -> int:
        """Count schedule slots where ``member_id`` participates as agent.

        Correct under:
          - sequential schedules (one agent per slot) — gives slots/N.
          - simultaneous schedules (N agents per slot) — gives per-member
            slot count, not the full slot count.
          - judge-inclusive schedules where a member (e.g. judge) appears
            in fewer slots than debaters.
        Falls back to 1 for dynamic (non-iterable) schedules; construction-
        time cross-check rejects mismatch between members and schedule
        agent set for the static case.
        """
        slots = getattr(self.schedule, "_slots", None)
        if slots is None:
            return 1
        count = sum(1 for slot in slots if member_id in slot.agents)
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
        num_rounds = self._member_round_count(member_id)
        current_round = sum(
            1 for u in kernel_state.transcript if u.member_id == member_id
        )
        ctx_current = self._build_prompt_context(
            state, role, slot.phase, current_round, num_rounds, question
        )

        msgs: list[dict[str, str]] = [
            {"role": "system", "content": self.prompts.render_system(role, ctx_current)},
        ]
        q_text = self.prompts.render_question(role, ctx_current)
        if q_text:
            msgs.append({"role": "user", "content": q_text})

        own_round_so_far = 0
        for utt in kernel_state.transcript:
            if utt.member_id == member_id:
                msgs.extend(
                    self._render_own_turn(
                        utt, role, own_round_so_far, num_rounds, question, state
                    )
                )
                own_round_so_far += 1
            else:
                msgs.append(self._render_opponent_message(utt, member_id, role))

        msgs.extend(self._render_current_suffix(role, slot, ctx_current))
        return msgs

    # -- build_prompt helpers ------------------------------------------------

    def _build_prompt_context(
        self,
        state: State,
        role: str,
        phase: str,
        round_index: int,
        num_rounds: int,
        question: str,
    ) -> dict[str, Any]:
        """Assemble the Jinja context dict for one (role, phase, round).

        ``num_rounds`` is per-member (see ``_member_round_count``) so
        is_first_round / is_last_round flags stay correct under
        simultaneous and judge-inclusive schedules.
        """
        return build_context(
            task_prompt=question,
            viewer_role=role,
            phase=phase,
            round_index=round_index,
            num_rounds=num_rounds,
            answer=state.get("answer", ""),
        )

    def _render_own_turn(
        self,
        utt: Utterance,
        role: str,
        round_index: int,
        num_rounds: int,
        question: str,
        state: State,
    ) -> list[dict[str, str]]:
        """Render one own-turn utterance as ``[instruction, assistant=raw]``.

        Re-renders the instruction that preceded ``utt`` using the past
        turn's own (phase, round_index), then emits the verbatim
        ``raw_content`` as the assistant message. ``round_index`` is the
        positional own-turn counter (N-th own commit = round N),
        independent of slot_id so sparse / semantic slot_ids don't
        produce nonsensical round labels.
        """
        past_ctx = self._build_prompt_context(
            state, role, utt.phase, round_index, num_rounds, question
        )
        msgs: list[dict[str, str]] = []
        past_instr = self.prompts.render_instruction(role, utt.phase, past_ctx)
        if past_instr:
            msgs.append({"role": "user", "content": past_instr})
        msgs.append({"role": "assistant", "content": utt.raw_content})
        return msgs

    def _render_current_suffix(
        self, role: str, slot: TurnSlot, ctx_current: dict[str, Any]
    ) -> list[dict[str, str]]:
        """Render the tail of the prompt: current instruction + optional prefill."""
        msgs: list[dict[str, str]] = []
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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def load_environment(**kwargs: Any) -> DebateEnv:
    """Construct a DebateEnv from a prompt pack and a static schedule.

    The factory is a pure dispatcher: it converts stringly-typed kwargs
    into typed component objects (schedule, prompts, judge client) and
    wires them into a rubric + env. It performs NO validation — each
    component validates its own preconditions at construction:

      * ``DebatePrompts.__post_init__``  — pack invariants (verdict-token
        collisions, non-empty judge templates)
      * ``DebateRubric.__init__``        — judge-client gate, client type
      * ``DebateEnv.__init__``           — cross-component coherence
        (env.members == rubric.members, schedule agents ⊆ members)

    Usage::

        env = load_environment(
            schedule_slots=[
                {"slot_id": 0, "agents": ["A"], "phase": "propose"},
                {"slot_id": 1, "agents": ["B"], "phase": "propose"},
                {"slot_id": 2, "agents": ["J"], "phase": "final"},
            ],
            members=["A", "B", "J"],
            truth_role="debater_a",
            prompts_ref="selfplay",
            role_for_agent={"A": "debater_a", "B": "debater_b", "J": "judge"},
            eval_dataset=my_dataset,
        )

    Required: schedule_slots, members, truth_role.
    Prompt source (exactly one): ``prompts_ref`` (str, registry lookup)
    or ``prompts`` (already-built DebatePrompts).
    Optional: role_for_agent, agent_overrides, judge_client OR
    (judge_api_key + judge_base_url + judge_max_retries), judge_model,
    dataset/eval_dataset.
    """
    from verifiers.envs.debate_rubric import DebateRubric

    schedule = StaticSchedule(
        tuple(
            TurnSlot(
                slot_id=s["slot_id"],
                agents=tuple(s["agents"]),
                phase=s.get("phase", ""),
            )
            for s in kwargs.pop("schedule_slots")
        )
    )
    prompts = _resolve_prompts_arg(
        kwargs.pop("prompts_ref", None),
        kwargs.pop("prompts", None),
    )
    judge_client = _build_judge_client(
        explicit=kwargs.pop("judge_client", None),
        api_key=kwargs.pop("judge_api_key", None),
        base_url=kwargs.pop("judge_base_url", None),
        max_retries=kwargs.pop("judge_max_retries", 10),
    )
    rubric = DebateRubric(
        truth_role=kwargs.pop("truth_role"),
        members=kwargs["members"],            # don't pop — env needs it too
        prompts=prompts,
        judge_client=judge_client,
        judge_model=kwargs.pop("judge_model", "gpt-4.1-nano"),
    )
    return DebateEnv(
        schedule=schedule,
        prompts=prompts,
        members=kwargs.pop("members"),
        role_for_agent=kwargs.pop("role_for_agent", None),
        agent_overrides=kwargs.pop("agent_overrides", None),
        rubric=rubric,
        **kwargs,
    )


def _resolve_prompts_arg(
    prompts_ref: str | None, prompts: DebatePrompts | None
) -> DebatePrompts:
    """Return a DebatePrompts from whichever source the caller provided.

    Exactly one of ``prompts_ref`` (registry lookup) or ``prompts`` (typed
    object) must be given. Pure conversion; ``DebatePrompts.__post_init__``
    runs the intrinsic pack validation.
    """
    if prompts is not None and prompts_ref is not None:
        raise ValueError("Provide exactly one of 'prompts_ref' or 'prompts'")
    if prompts is not None:
        if not isinstance(prompts, DebatePrompts):
            raise TypeError(
                f"Expected DebatePrompts, got {type(prompts).__name__}"
            )
        return prompts
    if prompts_ref is not None:
        return resolve_prompts(prompts_ref)
    raise ValueError("Must provide either 'prompts_ref' or 'prompts'")


def _build_judge_client(
    *,
    explicit: Any | None,
    api_key: str | None,
    base_url: str | None,
    max_retries: int,
) -> Any | None:
    """Return a judge client from an explicit object OR connection kwargs.

    Pure construction — no validation. The rubric's ``__init__`` is the
    single owner of the "client type" and "open-ended grading requires a
    client" invariants. Returns ``None`` when neither source is provided.
    """
    if explicit is not None:
        return explicit
    if api_key is None and base_url is None:
        return None
    from openai import AsyncOpenAI

    from verifiers.clients.openai_chat_completions_client import (
        OpenAIChatCompletionsClient,
    )

    return OpenAIChatCompletionsClient(
        AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
        )
    )
