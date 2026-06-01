from collections.abc import Sequence
from typing import TYPE_CHECKING, Generic, Literal, TypeVar, cast, final

from verifiers.types import Message, UserMessage
from verifiers.utils.message_utils import normalize_messages

from .artifact import Artifacts, ArtifactsConfig
from .config import Config, ConfigSource
from .sandbox import SandboxConfig
from .utils.binding_utils import (
    BindingSources,
    BindingsConfig,
    ObjectsConfig,
)
from .utils.config_utils import (
    coerce_config,
    config_type_from_class,
    registered_config_type,
    register_config_type,
)
from .utils.trajectory_utils import completion_from_trajectory
from .state import State
from .types import JsonData, Objects, PromptMessage

if TYPE_CHECKING:
    from .task import Task

UserScope = Literal["rollout", "group", "global"]


class UserConfig(Config):
    scope: UserScope = "rollout"
    bindings: BindingsConfig = BindingsConfig()
    objects: ObjectsConfig = ObjectsConfig()
    artifacts: ArtifactsConfig = ArtifactsConfig()
    sandbox: SandboxConfig | None = None


def state_messages(
    state: State, transcript: Sequence[PromptMessage] | None = None
) -> list[Message]:
    if transcript is not None:
        return normalize_messages(transcript, field_name="user.transcript")
    prompt = state.get("prompt")
    completion = state.get("completion")
    if isinstance(prompt, list) and isinstance(completion, list):
        return normalize_messages(
            [
                *cast(list[PromptMessage], prompt),
                *cast(list[PromptMessage], completion),
            ],
            field_name="state.messages",
        )
    if isinstance(completion, list):
        return normalize_messages(
            cast(list[PromptMessage], completion), field_name="state.completion"
        )
    trajectory = state.get("trajectory")
    if isinstance(trajectory, Sequence) and not isinstance(trajectory, str):
        return normalize_messages(
            completion_from_trajectory(cast(Sequence[JsonData], trajectory)),
            field_name="state.trajectory",
        )
    return []


ConfigT = TypeVar("ConfigT", bound=UserConfig)
user_type_registry: dict[type[UserConfig], type["User"]] = {}


class User(Generic[ConfigT]):
    config: ConfigT
    scope: UserScope
    bindings: BindingSources
    objects: Objects
    artifacts: Artifacts
    sandbox: SandboxConfig | None

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

    @final
    def __init__(
        self,
        *,
        config: ConfigSource = None,
    ):
        config_type = registered_config_type(type(self), UserConfig)
        self.config = cast(ConfigT, coerce_config(config_type, config))
        if self.config.scope not in {"rollout", "group", "global"}:
            raise ValueError("User scope must be 'rollout', 'group', or 'global'.")
        bindings = self.config.bindings.entries("User bindings", key_style="arg")
        if "messages" in bindings:
            raise ValueError("User messages are provided directly to get_response.")
        self.scope = self.config.scope
        self.bindings = bindings
        self.objects = self.load_objects(self.config.objects)
        self.artifacts = self.load_artifacts(self.config.artifacts)
        self.sandbox = self.config.sandbox

    def load_objects(self, config: ObjectsConfig) -> Objects:
        return config.objects("user.objects")

    def load_artifacts(self, config: ArtifactsConfig) -> Artifacts:
        return config.artifacts("user.artifacts")

    async def get_object(self, name: str, task: "Task", state: State) -> object:
        return await state._runtime().resolve_owner_object(self, name, task, state)

    async def get_response(
        self, task: "Task", state: State, messages: list[Message]
    ) -> list[UserMessage]:
        return []


def user_from_config(config: UserConfig) -> User:
    for config_type in type(config).__mro__:
        if not issubclass(config_type, UserConfig):
            continue
        user_type = user_type_registry.get(config_type)
        if user_type is not None:
            return user_type(config=config)
    raise TypeError(f"No User subclass is registered for {type(config).__name__}.")
