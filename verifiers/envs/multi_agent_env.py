"""MultiAgentEnv: generic N-actor rollout loop on the Environment contract.

Abstracts out the actor-agnostic machinery shared by multi-actor
environments (debate, RPS, PD, proposer-solver, ...):

- Slot-scheduled rollout loop (sequential and simultaneous barriers).
- Stop conditions with priority ordering (error > schedule_exhausted >
  prompt_too_long).
- Per-member actor resolution for self-play / adapters / fixed-opponent.
- Atomic simultaneous-slot commit (all commits land or none do).

Subclasses implement only the domain-specific bits: ``build_prompt``,
``render_completion``, optional ``extract_fields`` / ``visibility_policy``.

Design note — NOT a MultiTurnEnv subclass: ``MultiTurnEnv.rollout`` is
``@final`` and shaped for a single (env → actor → env) conversation.
Multi-actor rollouts are N speakers sharing a transcript — a different
shape that warrants a sibling of MultiTurnEnv, not a subclass.
"""

from __future__ import annotations

import asyncio
import logging
from abc import abstractmethod
from typing import Any, Literal, final

import verifiers as vf
from verifiers.clients import Client
from verifiers.envs.multi_actor_kernel import (
    KernelState,
    SlotProgram,
    TurnSlot,
    Utterance,
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
from verifiers.utils.message_utils import (
    fold_consecutive_user_messages,
    maybe_normalize_messages,
)
from verifiers.utils.response_utils import (
    parse_response_message,
    parse_response_tokens,
)

_log = logging.getLogger(__name__)

VisibilityMode = Literal["full", "public_only", "hidden"]


class MultiAgentEnv(vf.Environment):
    """Generic N-actor rollout environment.

    Rollout contract::

        init_state → KernelState(slot_index=0)
        while not is_completed:
            slot = schedule.current_slot(kernel)
            if slot is None: break
            if len(slot.actors) > 1: run_simultaneous_slot
            else:                   run_sequential_slot
        render_completion

    Monotonic prompt invariant (subclass CONTRACT):
        ``build_prompt(state, A, slot_N+1)`` MUST structurally extend
        ``build_prompt(state, A, slot_N)`` by appending at most a
        [user, assistant-prefill] pair at the tail. No prior messages
        modified or removed. Violating this defeats the vLLM prefix-cache
        reuse path in ``OpenAIChatCompletionsTokenClient.get_prompt_ids``
        and forces O(T²) tokenization over a T-turn episode.
    """

    def __init__(
        self,
        *,
        schedule: SlotProgram,
        members: list[str],
        actor_overrides: dict[str, tuple[Client | None, str | None]] | None = None,
        think_tag: str = "thinking",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not members:
            raise ValueError("MultiAgentEnv requires a non-empty members list")
        if len(members) != len(set(members)):
            raise ValueError(
                f"MultiAgentEnv.members contains duplicates: {members}"
            )
        overrides = actor_overrides or {}
        stray = set(overrides) - set(members)
        if stray:
            raise ValueError(
                f"actor_overrides keys not in members: {sorted(stray)} "
                f"(members={members})"
            )

        self.schedule: SlotProgram = schedule
        self.members: list[str] = list(members)
        self.actor_overrides: dict[str, tuple[Client | None, str | None]] = dict(
            overrides
        )
        self.think_tag = think_tag

    # -- abstract: subclass must implement -----------------------------------

    @abstractmethod
    async def build_prompt(
        self, state: State, member_id: str, slot: TurnSlot
    ) -> Messages:
        """Build the prompt for ``member_id`` at ``slot``.

        MUST satisfy the monotonic extension invariant (see class docstring).
        """

    @abstractmethod
    async def render_completion(self, state: State) -> None:
        """Populate ``state['completion']`` from the trajectory."""

    # -- optional hooks (sensible defaults) ----------------------------------

    async def extract_fields(
        self, public_channel: str, member_id: str, slot: TurnSlot
    ) -> dict[str, Any] | None:
        """Extract structured fields from a committed public channel.

        Default: no structured fields.
        """
        return None

    def resolve_actor(
        self, member_id: str
    ) -> tuple[Client | None, str | None]:
        """Return (client, model) override for ``member_id`` or (None, None)."""
        return self.actor_overrides.get(member_id, (None, None))

    def visibility_policy(
        self, utt: Utterance, viewer_id: str
    ) -> VisibilityMode:
        """Control what ``viewer_id`` sees of ``utt``.

        Default: opponents see public only; authors see full.
        """
        if utt.member_id == viewer_id:
            return "full"
        return "public_only"

    async def on_step_committed(
        self,
        state: State,
        utt: Utterance,
        fields: dict[str, Any] | None,
    ) -> None:
        """Hook fired after a commit + trajectory step append. Default: no-op."""
        return None

    def role_for_member(self, member_id: str) -> str:
        """Map member_id → role_id. Default: identity."""
        return member_id

    # -- prompt preparation --------------------------------------------------

    async def _prepare_prompt(
        self, state: State, actor: str, slot: TurnSlot
    ) -> Messages:
        """Subclass build_prompt → fold consecutive users → normalize.

        Fold before normalization: collapses consecutive user runs into
        single messages so chat-template rendering stays in-distribution
        AND so the token-stitch tail reduces to a _is_valid_env_tail-
        accepted shape (single trailing user, no (user,user) pairs).
        """
        prompt = await self.build_prompt(state, actor, slot)
        prompt = fold_consecutive_user_messages(prompt)
        return maybe_normalize_messages(prompt, field_name="multi_agent_prompt")

    # -- stop conditions (priority-ordered) ----------------------------------

    @vf.stop(priority=100)
    async def has_error(self, state: State) -> bool:
        return state.get("error") is not None

    @vf.stop(priority=50)
    async def schedule_exhausted(self, state: State) -> bool:
        kernel = state.get("_kernel")
        if kernel is None:
            return False
        return self.schedule.current_slot(kernel) is None

    @vf.stop(priority=10)
    async def prompt_too_long(self, state: State) -> bool:
        return state.get("prompt_too_long", False)

    # -- final: rollout loop -------------------------------------------------

    @final
    async def rollout(
        self,
        input: RolloutInput,
        client: Client,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        state = await self.init_state(input, client, model, sampling_args)
        try:
            state["_kernel"] = KernelState(slot_index=0)

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

    @final
    async def _run_sequential_slot(
        self, state: State, slot: TurnSlot
    ) -> None:
        actor = slot.actors[0]
        prompt = await self._prepare_prompt(state, actor, slot)
        actor_client, actor_model = self.resolve_actor(actor)
        state["_lineage_key"] = actor
        response = await self.get_model_response(
            state, prompt, client=actor_client, model=actor_model
        )
        content = response.message.content or ""
        token_count = _completion_token_count(response)

        result = apply_action(
            state["_kernel"], self.schedule, actor, content, token_count,
            think_tag=self.think_tag,
        )
        state["_kernel"] = result.new_state
        utt = result.committed[0]
        fields = await self.extract_fields(utt.public_channel, actor, slot)
        step = await self._build_step(state, prompt, response, utt, fields)
        state["trajectory"].append(step)
        await self.on_step_committed(state, utt, fields)

    @final
    async def _run_simultaneous_slot(
        self, state: State, slot: TurnSlot
    ) -> None:
        """Fully-staged atomic commit.

        Four-phase protocol — every phase runs to completion before the
        next begins, and nothing in state is mutated until the final
        publish step:

          1. Fan out model calls under TaskGroup. On first raise,
             TaskGroup cancels every sibling coroutine → no wasted
             tokens, no late completions leaking into the shared usage
             tracker after the slot is doomed.
          2. Stage: fold responses into a local kernel (the real
             state["_kernel"] is NOT touched), build per-actor
             (utt, fields, TrajectoryStep) triples, run extract_fields.
             Any raise here discards the local buffers entirely.
          3. on_step_committed for each actor. A raise still discards
             everything — the public state is unchanged.
          4. Publish: append every TrajectoryStep then assign
             state["_kernel"]. Dict append + assignment are the only
             writes and they're both non-await: if we reach phase 4, the
             slot succeeds atomically.
        """
        prompts = [await self._prepare_prompt(state, a, slot) for a in slot.actors]
        overrides = [self.resolve_actor(a) for a in slot.actors]

        # Per-actor state views isolate ``_lineage_key`` across concurrent
        # branches. State is a dict subclass; the shallow copy shares all
        # mutable values (trajectory, kernel, usage tracker) by reference
        # but each branch gets its own lineage-key read. Only applies to
        # the read path inside get_prompt_ids — no branch MUTATES a per-
        # actor key, so no cross-branch interference.
        per_actor_states = [type(state)(state) for _ in slot.actors]
        for s, a in zip(per_actor_states, slot.actors):
            s["_lineage_key"] = a

        # Phase 1: fan out under TaskGroup. First raise cancels siblings,
        # so no doomed-slot tokens leak into the shared usage tracker.
        # TaskGroup wraps failures in ExceptionGroup; unwrap to the first
        # OverlongPromptError / vf.Error so the rollout loop's normal
        # except clauses continue to work.
        responses: list[Response] = [None] * len(slot.actors)  # type: ignore[list-item]

        async def _run_one(idx: int) -> None:
            s = per_actor_states[idx]
            p = prompts[idx]
            o = overrides[idx]
            responses[idx] = await self.get_model_response(
                s, p, client=o[0], model=o[1]
            )

        # OverlongPromptError listed first so it's handled before the
        # broader vf.Error branch (it's a vf.Error subclass); both branches
        # re-raise the first exception after logging any suppressed peers.
        try:
            async with asyncio.TaskGroup() as tg:
                for i in range(len(slot.actors)):
                    tg.create_task(_run_one(i))
        except* vf.OverlongPromptError as eg:
            _log_suppressed_peers(slot.slot_id, eg.exceptions)
            raise eg.exceptions[0]
        except* vf.Error as eg:
            _log_suppressed_peers(slot.slot_id, eg.exceptions)
            raise eg.exceptions[0]

        # Phase 2: stage the fold into a LOCAL kernel + build trajectory
        # steps + extract fields. state["_kernel"] untouched.
        staged_kernel = state["_kernel"]
        staged_triples: list[tuple[Utterance, dict[str, Any] | None, TrajectoryStep]] = []
        committed_utts: list[Utterance] = []
        for actor, response in zip(slot.actors, responses):
            content = response.message.content or ""
            token_count = _completion_token_count(response)
            result = apply_action(
                staged_kernel, self.schedule, actor, content, token_count,
                think_tag=self.think_tag,
            )
            staged_kernel = result.new_state
            if result.committed:
                committed_utts.extend(result.committed)

        if len(committed_utts) != len(slot.actors):
            raise vf.Error(
                f"simultaneous slot {slot.slot_id}: expected "
                f"{len(slot.actors)} commits, got {len(committed_utts)}"
            )

        for actor, prompt, response, utt in zip(
            slot.actors, prompts, responses, committed_utts
        ):
            fields = await self.extract_fields(utt.public_channel, actor, slot)
            step = await self._build_step(state, prompt, response, utt, fields)
            staged_triples.append((utt, fields, step))

        # Phase 3: run commit hooks. Raising here still leaves state
        # untouched — nothing has been published yet.
        for utt, fields, _step in staged_triples:
            await self.on_step_committed(state, utt, fields)

        # Phase 4: PUBLISH. Synchronous tail: trajectory appends + kernel
        # assignment. No awaits, no raises after this point.
        for _utt, _fields, step in staged_triples:
            state["trajectory"].append(step)
        state["_kernel"] = staged_kernel

    # -- trajectory step build / append -------------------------------------

    async def _build_step(
        self,
        state: State,
        prompt: Messages,
        response: Response,
        utt: Utterance,
        fields: dict[str, Any] | None,
    ) -> TrajectoryStep:
        """Build a TrajectoryStep without mutating state.

        State is only mutated by the caller's ``trajectory.append``; the
        simultaneous-slot atomic staging phase constructs every step first
        (async: token parsing) and bails cleanly if any one raises.
        """
        completion_messages = await parse_response_message(response)
        tokens = await parse_response_tokens(response, self.max_seq_len)
        response_is_truncated = response.message.is_truncated or False
        is_truncated = response_is_truncated or (
            tokens is not None and bool(tokens.get("is_truncated"))
        )
        extras: dict[str, Any] = {
            "member_id": utt.member_id,
            "role_id": self.role_for_member(utt.member_id),
            "phase": utt.phase,
        }
        if fields is not None:
            extras["fields"] = fields
        return TrajectoryStep(
            prompt=prompt,
            completion=completion_messages,
            response=response,
            tokens=tokens,
            reward=None,
            advantage=None,
            is_truncated=is_truncated,
            trajectory_id=state["trajectory_id"],
            extras=extras,
        )

def _completion_token_count(response: Response) -> int:
    if response.message.tokens and response.message.tokens.completion_ids:
        return len(response.message.tokens.completion_ids)
    return 0


def _log_suppressed_peers(slot_id: int, exceptions: tuple[BaseException, ...]) -> None:
    if len(exceptions) > 1:
        _log.warning(
            "simultaneous-slot %d suppressed peer exceptions: %r",
            slot_id, exceptions[1:],
        )
