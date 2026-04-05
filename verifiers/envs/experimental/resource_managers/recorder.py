"""Simple event recording for debugging sandbox operations."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol, Self


class EventType(Enum):
    """Types of events that can be recorded."""
    COMMAND_START = auto()
    COMMAND_SUCCESS = auto()
    COMMAND_ERROR = auto()


@dataclass(slots=True)
class RecordedEvent:
    """Base class for recorded events."""
    event_type: EventType
    timestamp: float = field(default_factory=time.time)
    resource_id: str | None = None
    rollout_id: str | None = None
    turn_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CommandEvent(RecordedEvent):
    """Records a command execution."""
    command: str = ""
    working_dir: str | None = None
    stdout: str | None = None
    stderr: str | None = None
    exit_code: int | None = None
    duration_seconds: float | None = None

    @classmethod
    def success(
        cls,
        command: str,
        stdout: str,
        stderr: str | None = None,
        exit_code: int = 0,
        duration_seconds: float | None = None,
        **kwargs: Any,
    ) -> Self:
        """Create a command success event."""
        return cls(
            event_type=EventType.COMMAND_SUCCESS,
            command=command,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            duration_seconds=duration_seconds,
            **kwargs,
        )

    @classmethod
    def error(
        cls,
        command: str,
        error_message: str,
        duration_seconds: float | None = None,
        **kwargs: Any,
    ) -> Self:
        """Create a command error event."""
        return cls(
            event_type=EventType.COMMAND_ERROR,
            command=command,
            duration_seconds=duration_seconds,
            metadata={"error": error_message},
            **kwargs,
        )


class Recorder(Protocol):
    """Protocol for event recorders."""

    def record(self, event: RecordedEvent) -> None:
        """Record a single event."""
        ...

    def get_events(
        self,
        rollout_id: str | None = None,
        resource_id: str | None = None,
    ) -> list[RecordedEvent]:
        """Query recorded events with optional filters."""
        ...

    def clear(self, rollout_id: str | None = None) -> None:
        """Clear recorded events."""
        ...


class InMemoryRecorder:
    """Simple in-memory recorder for debugging."""

    def __init__(self, max_events: int | None = None):
        self.events: list[RecordedEvent] = []
        self.max_events = max_events

    def record(self, event: RecordedEvent) -> None:
        self.events.append(event)
        if self.max_events and len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

    def get_events(
        self,
        rollout_id: str | None = None,
        resource_id: str | None = None,
    ) -> list[RecordedEvent]:
        result = self.events
        if rollout_id is not None:
            result = [e for e in result if e.rollout_id == rollout_id]
        if resource_id is not None:
            result = [e for e in result if e.resource_id == resource_id]
        return result

    def get_commands(self, rollout_id: str | None = None) -> list[CommandEvent]:
        """Get command events for a rollout."""
        return [e for e in self.get_events(rollout_id=rollout_id) if isinstance(e, CommandEvent)]

    def clear(self, rollout_id: str | None = None) -> None:
        if rollout_id is None:
            self.events.clear()
        else:
            self.events = [e for e in self.events if e.rollout_id != rollout_id]


class NullRecorder:
    """No-op recorder (default)."""

    def record(self, event: RecordedEvent) -> None:
        pass

    def get_events(self, **kwargs: Any) -> list[RecordedEvent]:
        return []

    def clear(self, rollout_id: str | None = None) -> None:
        pass
