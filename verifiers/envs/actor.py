"""
Actor: A trainable entity with distinct identity (system prompt) in multi-agent environments.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from verifiers.types import State


@dataclass
class Actor:
    """A trainable actor with distinct system prompt. Set is_trainable=False for frozen actors."""

    id: str
    system_prompt: str = ""
    max_tokens: int = 4096
    is_trainable: bool = True
    sampling_args: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Actor):
            return self.id == other.id
        return False

    def __repr__(self) -> str:
        trainable_str = "trainable" if self.is_trainable else "frozen"
        return f"Actor(id={self.id!r}, {trainable_str})"

    def get_system_message(self) -> dict[str, str] | None:
        """Return system message dict or None if no system prompt."""
        if self.system_prompt:
            return {"role": "system", "content": self.system_prompt}
        return None

    def merge_sampling_args(self, base_args: dict[str, Any]) -> dict[str, Any]:
        """Merge actor's sampling args with base args (actor takes precedence)."""
        merged = dict(base_args)
        merged.update(self.sampling_args)
        if self.max_tokens:
            merged["max_tokens"] = self.max_tokens
        return merged

    def filter_state(self, state: "State") -> "State":
        """Filter state to what this actor can see. Override for hidden info games."""
        return state
