from collections.abc import Iterable
from typing import TYPE_CHECKING, Callable, Generic, TypeVar

from ..config import LifecycleConfig
from ..artifact import Artifacts, ArtifactsConfig
from ..toolset import Toolset, Toolsets, collect_toolsets, normalize_toolset_collection
from ..types import Handler, Objects
from ..user import User, UserConfig, user_from_config
from .binding_utils import BindingSources, ObjectsConfig
from .config_callable_utils import CallableKind, merge_config_handler_map

if TYPE_CHECKING:
    from ..state import State
    from ..task import Task


_HANDLER_KINDS: tuple[CallableKind, ...] = (
    "stop",
    "setup",
    "update",
    "metric",
    "reward",
    "advantage",
    "cleanup",
    "teardown",
)

ConfigT = TypeVar("ConfigT", bound=LifecycleConfig)


class RuntimeOwnerMixin(Generic[ConfigT]):
    config: ConfigT
    toolsets: list[Toolset]
    named_toolsets: dict[str, Toolset]
    stops: list[Handler]
    setups: list[Handler]
    updates: list[Handler]
    metrics: list[Handler]
    rewards: list[Handler]
    advantages: list[Handler]
    cleanups: list[Handler]
    teardowns: list[Handler]
    bindings: BindingSources
    objects: Objects
    artifacts: Artifacts
    runtime_refresh: Callable[[], None] | None

    def load_user(self, config: UserConfig) -> User:
        return user_from_config(config)

    def load_toolsets(self, config: ConfigT) -> Toolsets:
        return None

    def load_objects(self, config: ObjectsConfig) -> Objects:
        return config.objects(f"{type(self).__name__}.objects")

    def load_artifacts(self, config: ArtifactsConfig) -> Artifacts:
        return config.artifacts(f"{type(self).__name__}.artifacts")

    async def get_object(self, name: str, task: "Task", state: "State") -> object:
        return await state._runtime().resolve_owner_object(self, name, task, state)

    def initialize_runtime_refresh(self) -> None:
        self.runtime_refresh = None

    def initialize_runtime_user(self, user: UserConfig | None) -> None:
        self.user = None if user is None else self.load_user(user)

    def initialize_runtime_toolsets(self, config: ConfigT, toolsets: object) -> None:
        self.toolsets, self.named_toolsets = collect_toolsets(
            self.load_toolsets(config), toolsets
        )

    def initialize_runtime_handlers(self) -> None:
        defaults: dict[CallableKind, Iterable[Handler]] = {
            kind: () for kind in _HANDLER_KINDS
        }
        handlers = merge_config_handler_map(defaults, self.config)
        self.stops = handlers["stop"]
        self.setups = handlers["setup"]
        self.updates = handlers["update"]
        self.metrics = handlers["metric"]
        self.rewards = handlers["reward"]
        self.advantages = handlers["advantage"]
        self.cleanups = handlers["cleanup"]
        self.teardowns = handlers["teardown"]

    def refresh_runtime(self) -> None:
        if self.runtime_refresh is not None:
            self.runtime_refresh()

    def add_metric(self, fn: Handler) -> None:
        self.metrics.append(fn)
        self.refresh_runtime()

    def add_reward(self, fn: Handler) -> None:
        self.rewards.append(fn)
        self.refresh_runtime()

    def add_advantage(self, fn: Handler) -> None:
        self.advantages.append(fn)
        self.refresh_runtime()

    def add_toolset(self, toolset: object) -> None:
        toolsets, named_toolsets = normalize_toolset_collection(toolset)
        duplicate = set(self.named_toolsets) & set(named_toolsets)
        if duplicate:
            raise ValueError(f"Toolsets are defined twice: {sorted(duplicate)}.")
        self.toolsets.extend(toolsets)
        self.named_toolsets.update(named_toolsets)
        self.refresh_runtime()

    def add_stop(self, fn: Handler) -> None:
        self.stops.append(fn)
        self.refresh_runtime()

    def add_setup(self, fn: Handler) -> None:
        self.setups.append(fn)
        self.refresh_runtime()

    def add_update(self, fn: Handler) -> None:
        self.updates.append(fn)
        self.refresh_runtime()

    def add_cleanup(self, fn: Handler) -> None:
        self.cleanups.append(fn)
        self.refresh_runtime()

    def add_teardown(self, fn: Handler) -> None:
        self.teardowns.append(fn)
        self.refresh_runtime()
