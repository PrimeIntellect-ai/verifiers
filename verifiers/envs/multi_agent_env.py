"""MultiAgentEnv: generic N-agent rollout loop on the Environment contract.

Abstracts out the agent-agnostic machinery shared by multi-agent
environments (debate, RPS, PD, proposer-solver, ...):

- Slot-scheduled rollout loop (sequential and simultaneous barriers).
- Stop conditions with priority ordering (error > schedule_exhausted >
  prompt_too_long).
- Per-member agent resolution for self-play / adapters / fixed-opponent.
- Atomic simultaneous-slot commit (all commits land or none do).

Subclasses implement only the domain-specific bits: ``build_prompt``,
``render_completion``, optional ``extract_fields`` / ``visibility_policy``.

Design note — NOT a MultiTurnEnv subclass: ``MultiTurnEnv.rollout`` is
``@final`` and shaped for a single (env → agent → env) conversation.
Multi-agent rollouts are N speakers sharing a transcript — a different
shape that warrants a sibling of MultiTurnEnv, not a subclass.
"""

from __future__ import annotations

import asyncio
import logging
import re
from abc import abstractmethod
from typing import Any, Literal, final

import verifiers as vf
from verifiers.clients import Client
from verifiers.envs.multi_agent_kernel import (
    KernelState,
    SlotProgram,
    TurnSlot,
    Utterance,
    apply_action,
)
from verifiers.envs.request_context import ModelRequestContext
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
from verifiers.utils.usage_utils import StateUsageTracker

_log = logging.getLogger(__name__)

VisibilityMode = Literal["full", "public_only", "hidden"]


class MultiAgentEnv(vf.Environment):
    """Generic N-agent rollout environment.

    Rollout contract::

        init_state → KernelState(slot_index=0)
        while not is_completed:
            slot = schedule.current_slot(kernel)
            if slot is None: break
            if len(slot.agents) > 1: run_simultaneous_slot
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
        agent_overrides: dict[str, tuple[Client | None, str | None]] | None = None,
        think_tag: str = "thinking",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not members:
            raise ValueError("MultiAgentEnv requires a non-empty members list")
        if len(members) != len(set(members)):
            raise ValueError(f"MultiAgentEnv.members contains duplicates: {members}")
        overrides = agent_overrides or {}
        stray = set(overrides) - set(members)
        if stray:
            raise ValueError(
                f"agent_overrides keys not in members: {sorted(stray)} "
                f"(members={members})"
            )

        self.schedule: SlotProgram = schedule
        self.members: list[str] = list(members)
        self.agent_overrides: dict[str, tuple[Client | None, str | None]] = dict(
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

    def resolve_agent(self, member_id: str) -> tuple[Client | None, str | None]:
        """Return (client, model) override for ``member_id`` or (None, None)."""
        return self.agent_overrides.get(member_id, (None, None))

    def visibility_policy(self, utt: Utterance, viewer_id: str) -> VisibilityMode:
        """Control what ``viewer_id`` sees of ``utt``.

        Default: opponents see public only; authors see full.
        """
        if utt.member_id == viewer_id:
            return "full"
        return "public_only"

    # -- prompt preparation --------------------------------------------------

    async def _prepare_prompt(
        self, state: State, agent: str, slot: TurnSlot
    ) -> Messages:
        """Subclass ``build_prompt`` → normalize at boundary → fold.

        Pipeline order matters. ``maybe_normalize_messages`` guards the
        subclass boundary: if a subclass returns raw dicts it warns once
        and promotes to typed. ``fold_consecutive_user_messages`` then
        runs typed-in → typed-out, preserving types through to the
        model-request and the stitcher. Folding collapses adjacent user
        runs so chat-template rendering stays in-distribution and the
        token-stitch tail passes ``_is_valid_env_tail``.
        """
        prompt = await self.build_prompt(state, agent, slot)
        prompt = maybe_normalize_messages(prompt, field_name="multi_agent_prompt")
        return fold_consecutive_user_messages(prompt)

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
                    if len(slot.agents) > 1:
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
    async def _run_sequential_slot(self, state: State, slot: TurnSlot) -> None:
        agent = slot.agents[0]
        prompt = await self._prepare_prompt(state, agent, slot)
        agent_client, agent_model = self.resolve_agent(agent)
        parent_tracker = self._get_usage_tracker(state, create_if_missing=True)
        response = await self.get_model_response(
            state,
            prompt,
            client=agent_client,
            model=agent_model,
            request_context=ModelRequestContext(
                lineage_key=agent,
                usage_tracker=parent_tracker,
            ),
        )
        content = _coerce_text_content(response.message.content)
        content = _enrich_with_provider_reasoning(content, response, self.think_tag)
        token_count = _completion_token_count(response)

        result = apply_action(
            state["_kernel"],
            self.schedule,
            agent,
            content,
            token_count,
            think_tag=self.think_tag,
        )
        state["_kernel"] = result.new_state
        utt = result.committed[0]
        fields = await self.extract_fields(utt.public_channel, agent, slot)
        step = await self._build_step(state, prompt, response, utt, fields)
        state["trajectory"].append(step)

    @final
    async def _run_simultaneous_slot(self, state: State, slot: TurnSlot) -> None:
        """Fully-staged atomic commit.

        Three-phase protocol — every phase runs to completion before the
        next begins, and nothing in state is mutated until the final
        publish step:

          1. Fan out model calls under TaskGroup. On first raise,
             TaskGroup cancels every sibling coroutine → no wasted
             tokens, no late completions leaking into the shared usage
             tracker after the slot is doomed.
          2. Stage: fold responses into a local kernel (the real
             state["_kernel"] is NOT touched), build per-agent
             TrajectorySteps, run extract_fields. Any raise here discards
             the local buffers entirely.
          3. Publish: append every TrajectoryStep, assign
             state["_kernel"] and merge per-agent usage trackers. These
             writes are all non-await, so if we reach phase 3 the slot
             succeeds atomically.
        """
        prompts = [await self._prepare_prompt(state, a, slot) for a in slot.agents]
        overrides = [self.resolve_agent(a) for a in slot.agents]

        # Per-agent request contexts isolate the prefix-cache partition key
        # (``lineage_key``) and usage accounting across concurrent branches
        # without cloning the shared rollout state. Every branch charges a
        # child tracker; only the publish phase merges them back into the
        # parent, so a doomed slot never leaks token usage.
        parent_tracker = self._get_usage_tracker(state, create_if_missing=True)
        per_agent_trackers: list[StateUsageTracker | None] = [
            parent_tracker.fork() if parent_tracker is not None else None
            for _ in slot.agents
        ]

        # Phase 1: fan out concurrent model calls. On first raise, cancel
        # still-pending siblings — no wasted tokens, no late completions
        # leaking into the shared usage tracker after the slot is doomed.
        # Implemented with asyncio.wait(FIRST_EXCEPTION) to preserve the
        # cancel-on-first-raise semantics TaskGroup would give us while
        # staying compatible with Python 3.10 (TaskGroup is 3.11+ and
        # ``requires-python`` declares ``>=3.10``).
        responses: list[Response] = [None] * len(slot.agents)  # type: ignore[list-item]

        async def _run_one(idx: int) -> None:
            p = prompts[idx]
            o = overrides[idx]
            responses[idx] = await self.get_model_response(
                state,
                p,
                client=o[0],
                model=o[1],
                request_context=ModelRequestContext(
                    lineage_key=slot.agents[idx],
                    usage_tracker=per_agent_trackers[idx],
                ),
            )

        tasks = [asyncio.create_task(_run_one(i)) for i in range(len(slot.agents))]

        # ``asyncio.wait`` itself can be cancelled by the parent (rollout
        # timeout, Ctrl-C, encompassing wait_for) — if that happens while
        # tasks are inflight they keep running, leak tokens into the
        # shared usage tracker, and can late-commit into the wrong slot.
        # TaskGroup would have drained them on ``__aexit__``; we replicate
        # that behaviour explicitly with try/finally.
        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_EXCEPTION
            )
        except BaseException:
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        # First-exception return — cancel the stragglers AND capture any
        # results they already produced. ``asyncio.gather(...,
        # return_exceptions=True)`` returns each task's result or
        # exception; pending tasks that had already raised before our
        # cancel took effect would otherwise be silently dropped, losing
        # their error for the priority picker below.
        pending_results: list[Any] = []
        if pending:
            for t in pending:
                t.cancel()
            pending_results = list(
                await asyncio.gather(*pending, return_exceptions=True)
            )

        # Collect real exceptions from ``done`` AND from any pending task
        # whose raise beat our cancel. Filter out the ``CancelledError``s
        # we ourselves triggered — those are housekeeping, not signal.
        real_errors: list[BaseException] = [
            exc
            for t in done
            if (exc := t.exception()) is not None
            and not isinstance(exc, asyncio.CancelledError)
        ]
        for r in pending_results:
            if isinstance(r, BaseException) and not isinstance(
                r, asyncio.CancelledError
            ):
                real_errors.append(r)

        if real_errors:
            # Priority: OverlongPromptError first (most-specific recovery
            # in the rollout loop), then any other vf.Error, then bubble
            # the first remaining.
            chosen = (
                next(
                    (e for e in real_errors if isinstance(e, vf.OverlongPromptError)),
                    None,
                )
                or next(
                    (e for e in real_errors if isinstance(e, vf.Error)),
                    None,
                )
                or real_errors[0]
            )
            others = tuple(e for e in real_errors if e is not chosen)
            if others:
                _log_suppressed_peers(slot.slot_id, others)
            raise chosen

        # Phase 2: stage the fold into a LOCAL kernel + build trajectory
        # steps + extract fields. state["_kernel"] untouched.
        staged_kernel = state["_kernel"]
        staged_steps: list[TrajectoryStep] = []
        committed_utts: list[Utterance] = []
        for agent, response in zip(slot.agents, responses):
            content = _coerce_text_content(response.message.content)
            content = _enrich_with_provider_reasoning(content, response, self.think_tag)
            token_count = _completion_token_count(response)
            result = apply_action(
                staged_kernel,
                self.schedule,
                agent,
                content,
                token_count,
                think_tag=self.think_tag,
            )
            staged_kernel = result.new_state
            if result.committed:
                committed_utts.extend(result.committed)

        if len(committed_utts) != len(slot.agents):
            raise vf.Error(
                f"simultaneous slot {slot.slot_id}: expected "
                f"{len(slot.agents)} commits, got {len(committed_utts)}"
            )

        for agent, prompt, response, utt in zip(
            slot.agents, prompts, responses, committed_utts
        ):
            fields = await self.extract_fields(utt.public_channel, agent, slot)
            step = await self._build_step(state, prompt, response, utt, fields)
            staged_steps.append(step)

        # Phase 3: PUBLISH. Synchronous tail: trajectory appends + kernel
        # assignment + usage-tracker merge. No awaits, no raises after this
        # point. Merging is the accounting-side counterpart of the kernel
        # assignment — both happen iff the slot succeeds.
        for step in staged_steps:
            state["trajectory"].append(step)
        state["_kernel"] = staged_kernel
        if isinstance(parent_tracker, StateUsageTracker):
            for child in per_agent_trackers:
                if child is not None:
                    parent_tracker.merge(child)

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
            "phase": utt.phase,
        }
        if fields is not None:
            extras["fields"] = fields
        # Per-step quarantine flag — kernel set this on the Utterance when
        # parse_channels rejected the raw output. Propagating it here lets
        # the trainer mask the malformed completion tokens (otherwise the
        # raw garbage is trainable, which is the exact P0-2 the kernel
        # quarantine was meant to prevent).
        if utt.parse_error is not None:
            extras["parse_error"] = utt.parse_error
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


def _coerce_text_content(content: Any) -> str:
    """Coerce ``response.message.content`` to a string for the kernel.

    Provider returns either ``str`` or ``list[ContentPart]`` (multimodal).
    The MA pipeline (parse_channels, raw_content storage, downstream
    rendering) is text-only — joins TextContentPart text fields and
    drops non-text parts. Most multi-agent debate tasks are text-only;
    this helper exists so a stray multimodal response doesn't crash the
    kernel without surfacing.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                txt = part.get("text")
                if isinstance(txt, str):
                    text_parts.append(txt)
            else:
                txt = getattr(part, "text", None)
                if isinstance(txt, str):
                    text_parts.append(txt)
        return "".join(text_parts)
    return ""


def _extract_provider_reasoning(response: Response) -> str | None:
    """Flatten provider-emitted reasoning to a single string, or None.

    OpenAI o-series surfaces ``reasoning_content`` (str). Anthropic
    surfaces ``thinking_blocks`` (list of typed blocks; ``redacted``
    blocks have no readable content). Concatenated with paragraph
    breaks. Returns None when nothing reasoning-shaped is present.
    """
    msg = response.message
    parts: list[str] = []

    rc = getattr(msg, "reasoning_content", None)
    if isinstance(rc, str) and rc.strip():
        parts.append(rc.strip())

    blocks = getattr(msg, "thinking_blocks", None) or []
    for blk in blocks:
        text = getattr(blk, "thinking", None) or (
            blk.get("thinking") if isinstance(blk, dict) else None
        )
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())

    return "\n\n".join(parts) if parts else None


_NATIVE_THINK_PROBE = re.compile(r"<\s*(?:think|thinking)\s*>", re.IGNORECASE)


def _enrich_with_provider_reasoning(
    content: str, response: Response, think_tag: str
) -> str:
    """Wrap provider reasoning in ``<{think_tag}>...</{think_tag}>`` and
    prepend to ``content`` so ``parse_channels`` can split it normally.

    Single source of truth: kernel sees one ``raw_content`` string;
    ``public_channel`` is what survives stripping; ``private_channel``
    is what got stripped; "full" opponent view is the verbatim string.
    No special channels-merge inside the kernel.

    Skip enrichment when the model already emitted any think block
    inline — adding a second would trip ``parse_channels``'s
    one-block-per-whitelist contract and quarantine the utterance. If
    a model surfaces reasoning BOTH inline AND via the structured
    field we trust the inline channel as author-shaped intent.
    """
    pr = _extract_provider_reasoning(response)
    if not pr:
        return content
    if (
        _NATIVE_THINK_PROBE.search(content)
        or f"<{think_tag}".lower() in content.lower()
    ):
        return content
    return f"<{think_tag}>\n{pr}\n</{think_tag}>\n\n{content}"


def _log_suppressed_peers(slot_id: int, exceptions: tuple[BaseException, ...]) -> None:
    if exceptions:
        _log.warning(
            "simultaneous-slot %d suppressed peer exceptions: %r",
            slot_id,
            exceptions,
        )
