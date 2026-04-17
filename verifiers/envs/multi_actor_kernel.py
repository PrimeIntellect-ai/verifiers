"""Pure-functional multi-actor episode kernel."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from verifiers.errors import KernelProtocolError


@dataclass(frozen=True)
class TurnSlot:
    """One step in the schedule.

    len(actors) == 1 → sequential (commit immediately).
    len(actors) > 1  → simultaneous (barrier until all submit).
    """

    slot_id: int
    actors: tuple[str, ...]
    phase: str = ""

    def __post_init__(self) -> None:
        if not self.actors:
            raise ValueError("TurnSlot.actors must be non-empty")
        if len(self.actors) != len(set(self.actors)):
            raise ValueError(f"TurnSlot.actors contains duplicates: {self.actors}")


@dataclass(frozen=True)
class Utterance:
    """Structured actor output committed to the transcript.

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

# ``think`` and ``thinking`` are aliased: either tag name matches both
# ``<think>`` and ``<thinking>`` so models can use either form without
# silently failing the parse. Attribute syntax on opener/closer is allowed.
_ALIAS_PATTERN = r"think(?:ing)?"


def _tag_pattern(tag: str) -> str:
    if tag in ("think", "thinking"):
        return _ALIAS_PATTERN
    return re.escape(tag)


def parse_channels(raw: str, tag: str) -> tuple[str, str | None]:
    """Split raw model output into ``(public_channel, private_channel)``.

    Contract:
      - exactly zero or one ``<tag>...</tag>`` block is legal;
      - no nested tags, no unclosed opener, no stray closer;
      - multiple blocks → protocol error.

    ``public_channel`` is ``raw`` with the block removed and
    leading/trailing whitespace stripped. ``private_channel`` is the
    block's inner content stripped, or ``None`` if no block was present.

    Raises:
        KernelProtocolError: malformed channel markup.
    """
    pat = _tag_pattern(tag)
    opener_re = re.compile(rf"<{pat}\b[^>]*>", re.IGNORECASE)
    closer_re = re.compile(rf"</{pat}\s*>", re.IGNORECASE)

    openers = list(opener_re.finditer(raw))
    closers = list(closer_re.finditer(raw))

    if not openers and not closers:
        return raw.strip(), None

    if len(openers) != len(closers):
        raise KernelProtocolError(
            f"parse_channels: unbalanced <{tag}> markup "
            f"({len(openers)} opener(s), {len(closers)} closer(s))"
        )

    if len(openers) > 1:
        # Disambiguate nested vs. sequential multiple blocks by asking
        # whether the second opener begins before the first closer ends.
        if openers[1].start() < closers[0].end():
            raise KernelProtocolError(
                f"parse_channels: nested <{tag}> tags are not allowed"
            )
        raise KernelProtocolError(
            f"parse_channels: multiple <{tag}> blocks found ({len(openers)}); "
            "expected at most one"
        )

    opener = openers[0]
    closer = closers[0]

    if closer.start() < opener.end():
        raise KernelProtocolError(
            f"parse_channels: </{tag}> appears before <{tag}> opener"
        )

    inner = raw[opener.end() : closer.start()]

    public = (raw[: opener.start()] + raw[closer.end() :]).strip()
    private = inner.strip()
    return public, (private or None)


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
    """
    slot = state._active_slot if state._active_slot is not None else program.current_slot(state)

    if slot is None:
        raise KernelProtocolError("No active slot — episode is finished")

    if member_id not in slot.actors:
        raise KernelProtocolError(
            f"Member {member_id!r} is not scheduled for slot {slot.slot_id} "
            f"(expected one of {slot.actors})"
        )

    if member_id in state.pending:
        raise KernelProtocolError(
            f"Member {member_id!r} already submitted for slot {slot.slot_id}"
        )

    public, private = parse_channels(raw_content, think_tag)

    utterance = Utterance(
        member_id=member_id,
        slot_id=slot.slot_id,
        phase=slot.phase,
        raw_content=raw_content,
        public_channel=public,
        private_channel=private,
        token_count=token_count,
    )

    # Sequential: commit immediately
    if len(slot.actors) == 1:
        return ActionResult(
            new_state=replace(
                state,
                slot_index=state.slot_index + 1,
                transcript=state.transcript + (utterance,),
            ),
            committed=(utterance,),
        )

    # Simultaneous: buffer until all actors submit
    new_pending = {**state.pending, member_id: utterance}

    if len(new_pending) == len(slot.actors):
        committed = tuple(new_pending[actor] for actor in slot.actors)
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
