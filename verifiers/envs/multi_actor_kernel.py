"""Pure-functional multi-actor episode kernel."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable


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
    member_id: str
    slot_id: int
    phase: str
    content: Any
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


def apply_action(
    state: KernelState,
    program: SlotProgram,
    member_id: str,
    content: Any,
    token_count: int,
) -> ActionResult:
    """Pure reducer. Raises ValueError on protocol violations."""
    slot = state._active_slot if state._active_slot is not None else program.current_slot(state)

    if slot is None:
        raise ValueError("No active slot — episode is finished")

    if member_id not in slot.actors:
        raise ValueError(
            f"Member {member_id!r} is not scheduled for slot {slot.slot_id} "
            f"(expected one of {slot.actors})"
        )

    if member_id in state.pending:
        raise ValueError(
            f"Member {member_id!r} already submitted for slot {slot.slot_id}"
        )

    utterance = Utterance(
        member_id=member_id,
        slot_id=slot.slot_id,
        phase=slot.phase,
        content=content,
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
