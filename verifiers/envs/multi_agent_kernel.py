"""Pure-functional multi-agent episode kernel."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from verifiers.errors import ContentParseError, KernelProtocolError


@dataclass(frozen=True)
class TurnSlot:
    """One step in the schedule.

    len(agents) == 1 → sequential (commit immediately).
    len(agents) > 1  → simultaneous (barrier until all submit).
    """

    slot_id: int
    agents: tuple[str, ...]
    phase: str = ""

    def __post_init__(self) -> None:
        if not self.agents:
            raise ValueError("TurnSlot.agents must be non-empty")
        if len(self.agents) != len(set(self.agents)):
            raise ValueError(f"TurnSlot.agents contains duplicates: {self.agents}")


@dataclass(frozen=True)
class Utterance:
    """Structured agent output committed to the transcript.

    Three channels, populated once at commit time by ``parse_channels``:

    - ``raw_content``: verbatim model output, never mutated. Author's view
      renders this directly. Training bridge reads this for loss.
    - ``public_channel``: content with ``<{think_tag}>...</{think_tag}>``
      removed. Opponent/judge view uses this; field extractors read this.
    - ``private_channel``: the stripped think-block contents (or None if
      absent). May be revealed to select viewers by visibility policy.
    """

    member_id: str
    slot_id: int
    phase: str
    raw_content: str
    public_channel: str
    private_channel: str | None
    token_count: int
    # Per-utterance quarantine flag: populated when parse_channels rejected
    # the raw output (unclosed/nested/multiple/stray tags). Non-None means
    # the rollout continues but downstream consumers (rubric, bridge) know
    # this member's channel split is unreliable — public_channel is forced
    # to "" and private_channel to None so no malformed markup leaks.
    parse_error: str | None = None


@dataclass(frozen=True)
class KernelState:
    """Immutable episode state.

    ``_active_slot`` caches the slot for the current simultaneous barrier,
    guarding against non-deterministic SlotProgram implementations.
    """

    slot_index: int
    transcript: tuple[Utterance, ...] = ()
    pending: MappingProxyType[str, Utterance] = field(
        default_factory=lambda: MappingProxyType({})
    )
    _active_slot: TurnSlot | None = None


@dataclass(frozen=True)
class ActionResult:
    new_state: KernelState
    committed: tuple[Utterance, ...]


@runtime_checkable
class SlotProgram(Protocol):
    """Returns the current slot, or None when the episode is finished."""

    def current_slot(self, state: KernelState) -> TurnSlot | None: ...


class StaticSchedule:
    """SlotProgram backed by a fixed tuple of TurnSlots."""

    def __init__(self, slots: tuple[TurnSlot, ...]) -> None:
        self._slots = slots

    def current_slot(self, state: KernelState) -> TurnSlot | None:
        if state.slot_index >= len(self._slots):
            return None
        return self._slots[state.slot_index]

    def __len__(self) -> int:
        return len(self._slots)


# ---------------------------------------------------------------------------
# Channel parsing
# ---------------------------------------------------------------------------

# Native reasoning-model tags. Always stripped from public output, even when
# the pack configures a different private-channel tag — otherwise a model that
# emits ``<think>secret</think>`` under a pack with ``think_tag="reason"``
# would leak the raw native block into the opponent view.
_NATIVE_THINK_ALT = r"think(?:ing)?"


def _compile_tag(alt: str) -> tuple[re.Pattern, re.Pattern]:
    return (
        re.compile(rf"<(?:{alt})\b[^>]*>", re.IGNORECASE),
        re.compile(rf"</(?:{alt})\s*>", re.IGNORECASE),
    )


def _extract_one_block(raw: str, alt: str, label: str) -> tuple[str, str | None]:
    """Locate at most one ``<alt>…</alt>`` block.

    Returns ``(residual, inner)``: ``residual`` has the block excised
    (no whitespace normalization), ``inner`` is the raw body between
    opener/closer, or ``None`` if no block was present.

    Raises ``ContentParseError`` on unbalanced, multiple, nested, or
    stray-closer markup — these are model-output formatting slips,
    quarantined per-utterance by ``apply_action``.
    """
    opener_re, closer_re = _compile_tag(alt)
    openers = list(opener_re.finditer(raw))
    closers = list(closer_re.finditer(raw))

    if not openers and not closers:
        return raw, None

    if len(openers) != len(closers):
        raise ContentParseError(
            f"parse_channels: unbalanced {label} markup "
            f"({len(openers)} opener(s), {len(closers)} closer(s))"
        )

    if len(openers) > 1:
        # Nested vs. sequential disambiguation: second opener begins
        # before first closer ends → nesting, else distinct blocks.
        if openers[1].start() < closers[0].end():
            raise ContentParseError(
                f"parse_channels: nested {label} tags are not allowed"
            )
        raise ContentParseError(
            f"parse_channels: multiple {label} blocks found "
            f"({len(openers)}); expected at most one"
        )

    opener = openers[0]
    closer = closers[0]

    if closer.start() < opener.end():
        raise ContentParseError(f"parse_channels: {label} closer appears before opener")

    inner = raw[opener.end() : closer.start()]
    residual = raw[: opener.start()] + raw[closer.end() :]
    return residual, inner


def parse_channels(raw: str, tag: str) -> tuple[str, str | None]:
    """Split raw model output into ``(public_channel, private_channel)``.

    Two-pass whitelist strip:

    1. The configured ``tag`` block (if present) carries the author's
       intended private reasoning → ``private_channel``.
    2. Native ``<think>``/``<thinking>`` blocks are ALWAYS stripped from
       the public view — even when the configured tag is different —
       and their content is DISCARDED. Native reasoning tokens are a
       third-party artifact of the model, not author-intended private
       commentary, and surfacing them anywhere (opponent view, private
       channel, rubric) would leak model-internal state that the pack
       author did not opt into.

    When the configured tag aliases the native tag (``tag`` ∈
    {"think","thinking"}), one pass covers both: the block's content
    IS the author's private channel.

    Contract: zero or one block per whitelist entry. Malformed markup
    (unclosed / stray / multiple / nested) in EITHER whitelist entry
    raises ``ContentParseError``, which ``apply_action`` quarantines
    on the utterance rather than aborting the rollout.
    """
    configured_alt = (
        _NATIVE_THINK_ALT if tag in ("think", "thinking") else re.escape(tag)
    )

    residual, configured_inner = _extract_one_block(raw, configured_alt, f"<{tag}>")

    # Second pass only when the configured tag is not already the native
    # alias — otherwise pass 1 has already consumed any native block.
    if configured_alt != _NATIVE_THINK_ALT:
        residual, _discarded_native = _extract_one_block(
            residual, _NATIVE_THINK_ALT, "<think>/<thinking>"
        )
    # Native-think contents are intentionally dropped on the floor.

    public = residual.strip()
    private_stripped = (
        configured_inner.strip() if configured_inner is not None else None
    )
    return public, (private_stripped or None)


def apply_action(
    state: KernelState,
    program: SlotProgram,
    member_id: str,
    raw_content: str,
    token_count: int,
    *,
    think_tag: str = "thinking",
) -> ActionResult:
    """Pure reducer. Raises KernelProtocolError on protocol violations.

    ``raw_content`` is split into public/private channels via
    ``parse_channels`` exactly once here; the resulting ``Utterance``
    carries all three channels and downstream consumers never re-parse.
    Callers wishing to merge provider-side reasoning (OpenAI
    ``reasoning_content`` / Anthropic ``thinking_blocks``) into the
    private channel must enrich ``raw_content`` themselves by wrapping
    that reasoning in ``<{think_tag}>...</{think_tag}>`` before calling
    here — keeping ``raw_content`` as the single source of truth for
    both ``public_channel`` and the "full" opponent view.
    """
    slot = (
        state._active_slot
        if state._active_slot is not None
        else program.current_slot(state)
    )

    if slot is None:
        raise KernelProtocolError("No active slot — episode is finished")

    if member_id not in slot.agents:
        raise KernelProtocolError(
            f"Member {member_id!r} is not scheduled for slot {slot.slot_id} "
            f"(expected one of {slot.agents})"
        )

    if member_id in state.pending:
        raise KernelProtocolError(
            f"Member {member_id!r} already submitted for slot {slot.slot_id}"
        )

    # Quarantine parse failures on model output: one agent's formatting
    # slip must not DoS the whole episode. Kernel-state violations (wrong
    # agent, duplicate, finished) are raised above and still abort.
    try:
        public, private = parse_channels(raw_content, think_tag)
        parse_error: str | None = None
    except ContentParseError as exc:
        public = ""
        private = None
        parse_error = str(exc)

    # Quarantine empty-public commits. A reasoning-mode model that spends
    # its full token budget on reasoning_content emits content="" and
    # parses cleanly into ("", maybe-private). Committing it lets the
    # schedule advance with no anchor for "what did this agent say",
    # which downstream prompts then re-render as a lone <thinking> block
    # — confusing the next own-turn into thinking it's still in the
    # previous phase. Treat the absence of a public utterance as a
    # protocol violation, same as a malformed parse: trainer masks the
    # tokens, opponent renderer skips attribution. private_channel is
    # preserved for telemetry but never reaches an opponent's view.
    if parse_error is None and not public.strip():
        parse_error = "empty public channel (model emitted no visible answer)"

    utterance = Utterance(
        member_id=member_id,
        slot_id=slot.slot_id,
        phase=slot.phase,
        raw_content=raw_content,
        public_channel=public,
        private_channel=private,
        token_count=token_count,
        parse_error=parse_error,
    )

    # Sequential: commit immediately
    if len(slot.agents) == 1:
        return ActionResult(
            new_state=replace(
                state,
                slot_index=state.slot_index + 1,
                transcript=state.transcript + (utterance,),
            ),
            committed=(utterance,),
        )

    # Simultaneous: buffer until all agents submit
    new_pending = {**state.pending, member_id: utterance}

    if len(new_pending) == len(slot.agents):
        committed = tuple(new_pending[agent] for agent in slot.agents)
        return ActionResult(
            new_state=replace(
                state,
                slot_index=state.slot_index + 1,
                transcript=state.transcript + committed,
                pending=MappingProxyType({}),
                _active_slot=None,
            ),
            committed=committed,
        )

    return ActionResult(
        new_state=replace(
            state,
            pending=MappingProxyType(new_pending),
            _active_slot=slot,
        ),
        committed=(),
    )
