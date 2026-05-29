from collections.abc import Sequence
from typing import Generic, Literal, TypeVar, cast

from .config import Config, resolve_config_object
from .sandbox import SandboxConfig, sandbox_config_mapping
from .utils.binding_utils import BindingMap, normalize_binding_map
from .utils.binding_utils import normalize_object_map
from .utils.config_utils import (
    coerce_config,
    config_type_from_class,
    registered_config_type,
    register_config_type,
)
from .utils.trajectory_utils import completion_from_trajectory
from .types import ConfigMap, Handler, Objects, PromptMessage

UserScope = Literal["rollout", "group", "global"]


class UserConfig(Config):
    scope: UserScope = "rollout"
    bindings: BindingMap = {}
    objects: dict[str, str] = {}
    sandbox: SandboxConfig | None = None


def state_messages(
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


ConfigT = TypeVar("ConfigT", bound=UserConfig)
user_type_registry: dict[type[UserConfig], type["User"]] = {}


class User(Generic[ConfigT]):
    config: ConfigT
    scope: UserScope
    bindings: BindingMap
    objects: Objects
    sandbox: ConfigMap | None
    get_response: Handler

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        config_type = config_type_from_class(
            cls,
            inherited=False,
            owner_base=User,
            config_base=UserConfig,
        )
        if config_type is not None:
            register_config_type(cls, config_type)
            user_type_registry[cast(type[UserConfig], config_type)] = cls

    def __init__(
        self,
        *,
        config: object = None,
    ):
        config_type = registered_config_type(type(self), UserConfig)
        self.config = cast(ConfigT, coerce_config(config_type, config))
        if self.config.scope not in {"rollout", "group", "global"}:
            raise ValueError("User scope must be 'rollout', 'group', or 'global'.")
        bindings = normalize_binding_map(
            self.config.bindings, "User bindings", key_style="arg"
        )
        if "messages" in bindings:
            raise ValueError("User messages are provided directly to get_response.")
        self.scope = self.config.scope
        self.bindings = bindings
        self.objects = normalize_object_map(
            cast(
                Objects,
                {
                    str(key): resolve_config_object(value)
                    for key, value in self.config.objects.items()
                },
            ),
            "User objects",
        )
        self.sandbox = sandbox_config_mapping(self.config.sandbox, fill_defaults=False)


def normalize_user(value: object | None) -> User | None:
    if value is None:
        return None
    if isinstance(value, UserConfig):
        return user_from_config(value)
    raise TypeError("User must be a UserConfig.")


def user_from_config(config: UserConfig) -> User:
    for config_type in type(config).__mro__:
        if not issubclass(config_type, UserConfig):
            continue
        user_type = user_type_registry.get(config_type)
        if user_type is not None:
            return user_type(config=config)
    raise TypeError(f"No User subclass is registered for {type(config).__name__}.")
