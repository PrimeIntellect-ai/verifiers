"""DebateEnv: kernel-driven multi-actor debate on the Environment contract."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

_log = logging.getLogger(__name__)

import verifiers as vf
from verifiers.clients import Client
from verifiers.envs.debate.fields import EnumScoring
from verifiers.envs.debate.parsing import extract_fields
from verifiers.envs.debate.prompts import (
    DebatePrompts,
    build_context,
    resolve_prompts,
)
from verifiers.envs.debate.think import redact_think, strip_think
from verifiers.envs.multi_actor_kernel import (
    KernelState,
    SlotProgram,
    StaticSchedule,
    TurnSlot,
    apply_action,
)
from verifiers.types import (
    Messages,
    Response,
    RolloutInput,
    SamplingArgs,
    State,
    TrajectoryStep,
)
from verifiers.utils.message_utils import maybe_normalize_messages
from verifiers.utils.response_utils import (
    parse_response_message,
    parse_response_tokens,
)


# ---------------------------------------------------------------------------
# DebateEnv
# ---------------------------------------------------------------------------


class DebateEnv(vf.Environment):
    """Kernel-driven multi-actor debate environment.

    Subclasses ``vf.Environment`` directly (NOT MultiTurnEnv). The kernel
    (``multi_actor_kernel``) manages turn order; both actors call
    ``get_model_response`` as first-class participants.

    Actor resolution is mode-free:
      - self-play: all actors use defaults (client=None, model=None)
      - adapters: per-actor model strings (e.g. ``"base:lora_A"``)
      - fixed opponent: one actor gets a separate client+model
    """

    def __init__(
        self,
        schedule: SlotProgram,
        prompts: DebatePrompts,
        *,
        role_for_actor: dict[str, str] | None = None,
        actor_overrides: dict[str, tuple[Client | None, str | None]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.schedule = schedule
        self.prompts = prompts
        self._role_for_actor: dict[str, str] = role_for_actor or {}
        self._actor_overrides: dict[str, tuple[Client | None, str | None]] = (
            actor_overrides or {}
        )
        self._num_actors = _count_actors(schedule)

    def _role(self, member_id: str) -> str:
        return self._role_for_actor.get(member_id, member_id)

    # -- actor resolution (mode-free) ----------------------------------------

    def _resolve_actor(
        self, actor: str
    ) -> tuple[Client | None, str | None]:
        return self._actor_overrides.get(actor, (None, None))

    # -- prompt construction (3-section) -------------------------------------

    def _build_prompt(
        self, state: State, actor: str, phase: str
    ) -> Messages:
        kernel_state: KernelState = state["_kernel"]
        role = self._role(actor)
        question = _extract_question(state)

        num_slots = len(self.schedule) if hasattr(self.schedule, "__len__") else 0
        num_rounds = num_slots // self._num_actors if self._num_actors else 0
        round_index = kernel_state.slot_index // self._num_actors if self._num_actors else 0

        ctx = build_context(
            task_prompt=question,
            viewer_role=role,
            phase=phase,
            round_index=round_index,
            num_rounds=num_rounds,
            answer=state.get("answer", ""),
        )

        # 1. System: who you are (phase-independent — prefix-cache stable)
        msgs: list[dict[str, str]] = [
            {"role": "system", "content": self.prompts.render_system(role, ctx)},
        ]

        # Question (first user message)
        q_text = self.prompts.render_question(role, ctx)
        if q_text:
            msgs.append({"role": "user", "content": q_text})

        # 2. History: what was said, think-stripped for opponents
        msgs.extend(self._format_history(kernel_state, actor))

        # 3. Instruction: what to do + output format
        instruction = self.prompts.render_instruction(role, phase, ctx)
        if instruction:
            msgs.append({"role": "user", "content": instruction})

        # Prefill (trailing assistant message)
        prefill = self.prompts.render_prefill(role, phase, ctx)
        if prefill:
            msgs.append({"role": "assistant", "content": prefill})

        return _consolidate_messages(msgs)

    def _format_history(
        self, kernel_state: KernelState, actor: str
    ) -> list[dict[str, str]]:
        """Format transcript entries, stripping think blocks for opponents."""
        role = self._role(actor)
        msgs: list[dict[str, str]] = []
        for utt in kernel_state.transcript:
            if utt.member_id == actor:
                msgs.append({"role": "assistant", "content": utt.content})
            else:
                content = utt.content
                speaker_role = self._role(utt.member_id)
                vis = self.prompts.think_visibility.get(speaker_role, "disabled")
                if vis != "open" and not (vis == "visible_to_judge" and role == "judge"):
                    # Privacy-first redaction: strip closed tags AND any unclosed
                    # <think...> opener through EOF, so a malformed emitter cannot
                    # leak private reasoning to opponent/judge views.
                    content = redact_think(content, tag=self.prompts.think_tag)[0]
                content = self.prompts.wrap_opponent(
                    utt.phase,
                    content,
                    member_id=utt.member_id,
                    role_id=speaker_role,
                    viewer_role=role,
                )
                msgs.append({"role": "user", "content": content})
        return msgs

    # -- trajectory step creation --------------------------------------------

    async def _add_debate_step(
        self,
        state: State,
        prompt_messages: Messages,
        response: Response,
        member_id: str,
        phase: str,
        fields: dict[str, Any] | None = None,
    ) -> None:
        completion_messages = await parse_response_message(response)
        tokens = await parse_response_tokens(response, self.max_seq_len)
        response_is_truncated = response.message.is_truncated or False
        is_truncated = response_is_truncated or (
            tokens is not None and bool(tokens.get("is_truncated"))
        )
        role_id = self._role(member_id)
        extras: dict[str, Any] = {
            "member_id": member_id,
            "role_id": role_id,
            "phase": phase,
        }
        if fields is not None:
            extras["fields"] = fields
        step = TrajectoryStep(
            prompt=prompt_messages,
            completion=completion_messages,
            response=response,
            tokens=tokens,
            reward=None,
            advantage=None,
            is_truncated=is_truncated,
            trajectory_id=state["trajectory_id"],
            extras=extras,
        )
        state["trajectory"].append(step)

    # -- field extraction ----------------------------------------------------

    def _extract_fields(
        self, content: str, actor: str, phase: str
    ) -> dict[str, Any] | None:
        """Extract structured fields from response, if specs are defined.

        Parse ambiguity (duplicate schema tags, raised as ValueError from
        parsing.parse) is a per-step, per-member signal — not a terminal
        rollout error. We log loudly and return None so the step still gets
        appended to the trajectory with fields=None; the rubric's
        failed_members path then emits extraction_failed/{mid}=1.0 and
        suppresses accuracy/{mid}. Terminating the rollout would discard
        other members' valid commits and hide the failure as state['error']
        while the rubric silently graded stale earlier steps.
        """
        role = self._role(actor)
        specs = self.prompts.get_field_specs(role, phase)
        if not specs:
            return None
        cleaned = strip_think(content, tag=self.prompts.think_tag)[0]
        try:
            return extract_fields(cleaned, specs)
        except ValueError as exc:
            _log.warning(
                "field extraction failed for actor=%s phase=%s: %s",
                actor, phase, exc,
            )
            return None

    # -- stop conditions -----------------------------------------------------

    @vf.stop(priority=100)
    async def has_error(self, state: State) -> bool:
        return state.get("error") is not None

    @vf.stop
    async def debate_complete(self, state: State) -> bool:
        kernel_state = state.get("_kernel")
        return kernel_state is not None and self.schedule.current_slot(kernel_state) is None

    @vf.stop
    async def prompt_too_long(self, state: State) -> bool:
        return state.get("prompt_too_long", False)

    # -- completion rendering ------------------------------------------------

    async def render_completion(self, state: State) -> None:
        completion_msgs: list[dict[str, Any]] = []
        for step in state["trajectory"]:
            completion_msgs.extend(step["completion"])
        state["completion"] = completion_msgs

    # -- main rollout loop ---------------------------------------------------

    async def rollout(
        self,
        input: RolloutInput,
        client: Client,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        state = await self.init_state(input, client, model, sampling_args)
        try:
            kernel_state = KernelState(slot_index=0)
            state["_kernel"] = kernel_state

            while not await self.is_completed(state):
                slot = self.schedule.current_slot(state["_kernel"])
                if slot is None:
                    break

                try:
                    if len(slot.actors) > 1:
                        await self._run_simultaneous_slot(state, slot)
                    else:
                        await self._run_sequential_slot(state, slot)
                except vf.OverlongPromptError:
                    state["prompt_too_long"] = True
                    state["is_truncated"] = True
                except vf.Error as e:
                    state["error"] = e

            await self.render_completion(state)
            return state
        except asyncio.CancelledError:
            await self._cleanup(state)
            raise

    async def _run_sequential_slot(self, state: State, slot: TurnSlot) -> None:
        actor = slot.actors[0]
        prompt = self._build_prompt(state, actor, slot.phase)
        prompt = maybe_normalize_messages(prompt, field_name="debate_prompt")
        actor_client, actor_model = self._resolve_actor(actor)
        response = await self.get_model_response(
            state, prompt, client=actor_client, model=actor_model
        )
        content = response.message.content or ""
        token_count = _completion_token_count(response)

        fields = self._extract_fields(content, actor, slot.phase)

        result = apply_action(
            state["_kernel"], self.schedule, actor, content, token_count
        )
        state["_kernel"] = result.new_state
        await self._add_debate_step(
            state, prompt, response, actor, slot.phase, fields=fields
        )

    async def _run_simultaneous_slot(self, state: State, slot: TurnSlot) -> None:
        prompts = [
            maybe_normalize_messages(
                self._build_prompt(state, a, slot.phase),
                field_name="debate_prompt",
            )
            for a in slot.actors
        ]
        overrides = [self._resolve_actor(a) for a in slot.actors]

        responses = await asyncio.gather(*[
            self.get_model_response(state, p, client=o[0], model=o[1])
            for p, o in zip(prompts, overrides)
        ])

        for actor, prompt, response in zip(slot.actors, prompts, responses):
            content = response.message.content or ""
            token_count = _completion_token_count(response)

            fields = self._extract_fields(content, actor, slot.phase)

            result = apply_action(
                state["_kernel"], self.schedule, actor, content, token_count
            )
            state["_kernel"] = result.new_state
            await self._add_debate_step(
                state, prompt, response, actor, slot.phase, fields=fields
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _completion_token_count(response: Response) -> int:
    """Extract completion token count from a Response."""
    if response.message.tokens and response.message.tokens.completion_ids:
        return len(response.message.tokens.completion_ids)
    return 0


def _count_actors(schedule: SlotProgram) -> int:
    """Count unique actors in a schedule."""
    if isinstance(schedule, StaticSchedule):
        actors: set[str] = set()
        for slot in schedule._slots:
            actors.update(slot.actors)
        return len(actors) or 1
    return 1


def _extract_question(state: State) -> str:
    """Extract the question text from state['prompt'] (the dataset messages).

    The prompt field is a list of messages from the dataset row. The first
    user-role message contains the question text.
    """
    prompt = state.get("prompt")
    if prompt:
        for msg in prompt:
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
            if role == "user" and content:
                return content
    return ""


def _consolidate_messages(msgs: list[dict[str, str]]) -> Messages:
    """Merge contiguous same-role non-system messages."""
    if not msgs:
        return msgs
    out = [msgs[0]]
    for msg in msgs[1:]:
        if (
            msg["role"] == out[-1]["role"]
            and msg["role"] != "system"
            and isinstance(msg["content"], str)
            and isinstance(out[-1]["content"], str)
        ):
            out[-1] = {
                "role": msg["role"],
                "content": out[-1]["content"] + "\n\n" + msg["content"],
            }
        else:
            out.append(msg)
    return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _validate_judge_templates(prompts: DebatePrompts) -> None:
    """Fail loud at load time on two judge-template footguns:

    1. ``template.user_format_string`` is empty. The 1B→1C transition
       added a raw-format-string carrier on ``JudgeTemplate`` so
       ``JudgeRubric.judge`` can run ``.format(question=..., answer=...,
       response=...)`` instead of Jinja. Production loader populates it
       from ``block["user"]``, but a hand-constructed ``JudgeTemplate``
       (or a YAML pack with an empty ``user`` block) would silently
       leave it as the ``""`` default and the grader would render a
       0-byte prompt at score time. Reject empty.

    2. A verdict token exactly matches an ``EnumScoring`` value of any
       ``answer`` field in the pack. The rubric classifies verdicts by
       exact-token compare against ``template.positive`` /
       ``template.negative`` (see ``_normalize_verdict_token``), so
       overlap between verdict tokens themselves is not a bug —
       ``CORRECT``/``INCORRECT`` is the canonical grader pair and both
       must be accepted even though one is a substring of the other.
       The collision we *do* reject is when a verdict token coincides
       with a canonical answer option (e.g. ``positive="A"`` while an
       MCQ answer field ranges over ``{A,B,C,D}``). In that case the
       grader's verdict and a debater commit are indistinguishable in
       transcript greps and downstream tooling that greps for the
       verdict would silently misattribute it.

    ``template.positive``/``template.negative`` are already normalized
    (uppercased, first-word, punct-stripped) at pack-load time, so
    comparisons here upper-case the enum side only.
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

    Usage (direct import — the only supported path)::

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

    Required kwargs:
        schedule_slots: list of dicts with keys ``slot_id``, ``actors``, ``phase``
        members: list of member_id strings (e.g. ["A", "B"])
        truth_role: role_id whose victory = episode success

    Prompt configuration (exactly one of):
        prompts_ref: str — YAML ref for :func:`resolve_prompts` (e.g. "default",
            "selfplay", or a filesystem path to a YAML pack)
        prompts: DebatePrompts — pre-loaded DebatePrompts instance

    Optional kwargs:
        role_for_actor, actor_overrides, judge_client, judge_model,
        dataset / eval_dataset, and any other Environment kwargs.

        For the judge, callers can either pass a pre-built
        ``judge_client`` (a ``vf.Client`` instance such as
        ``OpenAIChatCompletionsClient(AsyncOpenAI(...))``), or pass
        credentials via ``judge_api_key`` / ``judge_base_url`` and let
        the factory construct the client with
        ``max_retries=judge_max_retries`` (default 10). Prefer the
        factory construction path so retry budget stays consistent
        across callers.

    .. note::
       This factory is **not** reachable via ``vf.load_environment("debate_env")``.
       The framework loader at :mod:`verifiers.utils.env_utils` resolves
       ``env_id`` to a **top-level** module name (``importlib.import_module(env_id)``),
       but this module lives at :mod:`verifiers.envs.debate_env`. Calling
       ``vf.load_environment("debate_env")`` raises
       ``ValueError: Could not import 'debate_env' environment`` (wrapping
       ``ModuleNotFoundError: No module named 'debate_env'``). Use the direct
       import shown above — that is the contract every production caller relies on.
    """
    from verifiers.envs.debate_rubric import (
        DebateRubric,
        any_field_needs_open_ended,
    )

    # Build schedule
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

    # Resolve prompts
    prompts_ref = kwargs.pop("prompts_ref", None)
    prompts = kwargs.pop("prompts", None)
    if prompts is not None:
        if not isinstance(prompts, DebatePrompts):
            raise TypeError(f"Expected DebatePrompts, got {type(prompts).__name__}")
    elif prompts_ref is not None:
        prompts = resolve_prompts(prompts_ref)
    else:
        raise ValueError("Must provide either 'prompts_ref' or 'prompts'")

    # Fail loud on judge-template footguns (empty format string, verdict
    # token collisions) before we burn any rollout budget. See
    # _validate_judge_templates for the two failure modes.
    _validate_judge_templates(prompts)

    # Build rubric
    members = kwargs.pop("members")
    truth_role = kwargs.pop("truth_role")
    judge_client = kwargs.pop("judge_client", None)
    judge_model = kwargs.pop("judge_model", "gpt-4.1-nano")
    judge_api_key = kwargs.pop("judge_api_key", None)
    judge_base_url = kwargs.pop("judge_base_url", None)
    judge_max_retries = kwargs.pop("judge_max_retries", 10)

    # Construction path for callers that pass credentials instead of a
    # pre-built client. max_retries defaults to 10 (matching the
    # framework ClientConfig default) rather than the OpenAI SDK's
    # default of 2, because transient judge outages retry through
    # maybe_retry at the rollout layer plus the SDK's own retry — the
    # two compound, and an under-provisioned SDK retry budget burns
    # rollouts on recoverable flakes.
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

    # Eager judge_client validation: fail at load time, not score time.
    # We only demand a judge_client when the pack declares LLM-judge
    # templates AND at least one 'answer' field routes through the LLM
    # grader path (non-EnumScoring). Packs that ship _grader/_matcher as
    # dead-code fallback but score every answer field via classify_enum
    # (e.g. selfplay MCQ) must still load without a judge client.
    # DebateRubric.__init__ mirrors this gate as a belt-and-braces safety
    # net for callers who bypass this factory.
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
            "SDK clients are no longer accepted — wrap them first."
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

    # Actor overrides and role mapping
    role_for_actor = kwargs.pop("role_for_actor", None)
    actor_overrides = kwargs.pop("actor_overrides", None)

    return DebateEnv(
        schedule=schedule,
        prompts=prompts,
        role_for_actor=role_for_actor,
        actor_overrides=actor_overrides,
        rubric=rubric,
        **kwargs,
    )
