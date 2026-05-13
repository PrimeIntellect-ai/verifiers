from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, cast

from .config import UserConfig, import_config_ref, resolve_config_object
from .types import ObjectSpecs
from .utils.binding_utils import BindingMap, normalize_binding_map
from .utils.binding_utils import normalize_object_map
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
    bindings: BindingMap = field(default_factory=dict)
    objects: ObjectSpecs = field(default_factory=dict)
    sandbox: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if self.scope not in {"rollout", "group", "global"}:
            raise ValueError("User scope must be 'rollout', 'group', or 'global'.")
        bindings = normalize_binding_map(
            self.bindings, "User bindings", key_style="arg"
        )
        try:
            parameters = inspect.signature(self.fn).parameters
        except (TypeError, ValueError):
            parameters = {}
        if "transcript" in parameters:
            bindings.setdefault("transcript", state_transcript)
        object.__setattr__(self, "bindings", bindings)
        object.__setattr__(
            self, "objects", normalize_object_map(self.objects, "User objects")
        )


def normalize_user(value: object | None) -> User | None:
    value = resolve_config_object(value) if value is not None else None
    if value is None or isinstance(value, User):
        return value
    if isinstance(value, UserConfig):
        return user_from_mapping(value.model_dump(exclude_none=True))
    if isinstance(value, Mapping):
        return user_from_mapping(cast(Mapping[str, object], value))
    if callable(value):
        return User(value)
    raise TypeError("User must be a callable, User, import ref, or mapping.")


def user_from_mapping(spec: Mapping[str, object]) -> User:
    config = UserConfig.from_config(spec)
    fn = config.fn
    if isinstance(fn, str):
        fn = import_config_ref(fn)
    if not callable(fn):
        raise TypeError("User config requires callable fn.")
    return User(
        fn=fn,
        scope=cast(UserScope, config.scope),
        bindings=config.bindings,
        objects={
            str(key): resolve_config_object(value)
            for key, value in config.objects.items()
        },
        sandbox=config.sandbox.model_dump(exclude_none=True)
        if config.sandbox is not None
        else None,
    )
