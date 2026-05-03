from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, cast

from .config import import_config_ref, resolve_config_object
from .utils.trajectory_utils import completion_from_trajectory

UserScope = Literal["rollout", "group", "global"]


def state_transcript(
    state: Mapping[str, object], transcript: Sequence[object] | None = None
) -> list[object]:
    if transcript is not None:
        return list(transcript)
    prompt = state.get("prompt")
    completion = state.get("completion")
    if isinstance(prompt, list) and isinstance(completion, list):
        return [*prompt, *completion]
    if isinstance(completion, list):
        return list(completion)
    trajectory = state.get("trajectory")
    if isinstance(trajectory, Sequence) and not isinstance(trajectory, str):
        return completion_from_trajectory(
            cast(Sequence[Mapping[str, object]], trajectory)
        )
    return []


@dataclass(frozen=True)
class User:
    fn: Callable[..., object]
    scope: UserScope = "rollout"
    bindings: Mapping[str, object] = field(default_factory=dict)
    objects: Mapping[str, object] = field(default_factory=dict)
    sandbox: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if self.scope not in {"rollout", "group", "global"}:
            raise ValueError("User scope must be 'rollout', 'group', or 'global'.")
        bindings = {"transcript": state_transcript, **dict(self.bindings)}
        object.__setattr__(self, "bindings", bindings)


def normalize_user(value: object | None) -> User | None:
    value = resolve_config_object(value) if value is not None else None
    if value is None or isinstance(value, User):
        return value
    if isinstance(value, Mapping):
        return user_from_mapping(cast(Mapping[str, object], value))
    if callable(value):
        return User(value)
    raise TypeError("User must be a callable, User, import ref, or mapping.")


def user_from_mapping(spec: Mapping[str, object]) -> User:
    unknown_keys = set(spec) - {"fn", "scope", "bindings", "objects", "sandbox"}
    if unknown_keys:
        unknown = ", ".join(sorted(unknown_keys))
        raise ValueError(f"User config has unknown keys: {unknown}.")
    fn = spec.get("fn")
    if isinstance(fn, str):
        fn = import_config_ref(fn)
    if not callable(fn):
        raise TypeError("User config requires callable fn.")
    scope = spec.get("scope") or "rollout"
    if not isinstance(scope, str):
        raise TypeError("User scope must be a string.")
    bindings = spec.get("bindings") or {}
    if not isinstance(bindings, Mapping):
        raise TypeError("User bindings must be a mapping.")
    objects = spec.get("objects") or {}
    if not isinstance(objects, Mapping):
        raise TypeError("User objects must be a mapping.")
    sandbox = spec.get("sandbox")
    if sandbox is not None and not isinstance(sandbox, Mapping):
        raise TypeError("User sandbox must be a mapping.")
    return User(
        fn=fn,
        scope=cast(UserScope, scope),
        bindings=cast(Mapping[str, object], bindings),
        objects={
            str(key): resolve_config_object(value) for key, value in objects.items()
        },
        sandbox=cast(Mapping[str, object] | None, sandbox),
    )
