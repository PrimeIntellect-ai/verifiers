import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, cast

from .config import UserConfig, import_config_ref, resolve_config_object
from .utils.binding_utils import BindingMap, normalize_binding_map
from .utils.binding_utils import normalize_object_map
from .utils.trajectory_utils import completion_from_trajectory
from .types import ConfigMap, Handler, Objects, PromptMessage

UserScope = Literal["rollout", "group", "global"]


def state_transcript(
    state: ConfigMap, transcript: Sequence[PromptMessage] | None = None
) -> list[PromptMessage]:
    if transcript is not None:
        return list(transcript)
    prompt = state.get("prompt")
    completion = state.get("completion")
    if isinstance(prompt, list) and isinstance(completion, list):
        return [
            *cast(list[PromptMessage], prompt),
            *cast(list[PromptMessage], completion),
        ]
    if isinstance(completion, list):
        return list(cast(list[PromptMessage], completion))
    trajectory = state.get("trajectory")
    if isinstance(trajectory, Sequence) and not isinstance(trajectory, str):
        return completion_from_trajectory(cast(Sequence[ConfigMap], trajectory))
    return []


@dataclass(frozen=True)
class User:
    fn: Handler
    scope: UserScope = "rollout"
    bindings: BindingMap = field(default_factory=dict)
    objects: Objects = field(default_factory=dict)
    sandbox: ConfigMap | None = None

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
        return user_from_mapping(cast(ConfigMap, value))
    if callable(value):
        return User(value)
    raise TypeError("User must be a callable, User, import ref, or mapping.")


def user_from_mapping(spec: ConfigMap) -> User:
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
